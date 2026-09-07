"""
signals.py — Lógica de señales de la estrategia de Retorno a la Media (v2).

Módulo PURO: no importa streamlit ni pinta nada. Se puede usar desde la app,
desde el backtest o desde un script/cron sin arrancar la interfaz.

Cambios respecto a la v1 que vivía dentro de app.py:
  1. AND estricto: las 4 condiciones son obligatorias (se acabó el 75/100).
  2. El R/R mínimo actúa como puerta: si no llega, no hay señal.
  3. Stop por ATR (homogéneo entre activos) en lugar de por la mecha.
  4. Filtro de régimen de volatilidad (Nagel 2012): la reversión paga cuando
     la volatilidad está alta, no cuando está muerta.
  5. Vela de rechazo con exigencia de mecha sobre el RANGO, no solo sobre el
     cuerpo (un doji con cuerpo ~0 pasaba trivialmente el test antiguo).
  6. Todo vectorizado sobre el histórico completo -> apto para backtest.
  7. La tendencia admite un estado 'neutral' real, con banda de tolerancia
     alrededor de la SMA200.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ======================================================================
# CONFIGURACIÓN
# ======================================================================

CONFIG_V2 = {
    # --- Indicadores ---
    'rsi_period': 13,
    'rsi_oversold': 30.0,
    'rsi_overbought': 70.0,
    'bb_period': 30,
    'bb_std': 2.0,
    'atr_period': 14,

    # --- Tendencia ---
    'sma_trend': 200,
    'sma_tolerancia': 0.02,      # ±2% alrededor de la SMA200 = zona neutral
    'monthly_lookback': 3,
    'monthly_min_agree': 2,

    # --- Vela de rechazo ---
    'wick_body_ratio': 1.5,      # mecha >= cuerpo * ratio
    'wick_range_min': 0.33,      # mecha >= rango * esto  (mata los dojis)

    # --- Filtros de exclusión ---
    'max_range_atr': 3.5,        # v1 tenía 2.5: filtraba las mejores señales
    'min_bandwidth_pct': 20.0,   # percentil mínimo de ancho de bandas
    'bandwidth_window': 100,
    'volume_breakout_mult': 2.0,
    'breakout_dist_std': 0.5,

    # --- Régimen de volatilidad (Nagel 2012) ---
    'usar_filtro_vol': True,
    'vol_window': 20,            # ventana de volatilidad realizada
    'vol_lookback': 252,         # ventana del percentil
    'vol_min_pct': 40.0,         # solo operar por encima de este percentil

    # --- Liquidez ---
    'min_volumen_medio': 0.0,    # volumen medio 20d en unidades monetarias

    # --- Gestión ---
    'stop_mode': 'atr',          # 'atr' | 'mecha'
    'atr_stop_mult': 2.0,
    'stop_buffer': 0.001,        # solo para stop_mode='mecha'
    'min_rr': 1.5,               # PUERTA: sin esto no hay señal
    'objetivo_dinamico': True,   # el objetivo sigue a la banda media viva
    'max_barras': 10,            # salida temporal
    'partial_at_middle': 0.5,   # fracción que se cierra en la banda media
    'risk_per_trade': 0.01,
    'max_position_pct': 0.25,    # tope de exposición por posición

    # --- Dirección ---
    'permitir_long': True,
    'permitir_short': False,     # ver Ni et al. (2020): en banda superior
                                 # funciona mejor momentum que contrarian
}


# ======================================================================
# UTILIDADES
# ======================================================================

def normalizar_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja un DataFrame con columnas Open/High/Low/Close/Volume, venga de
    yf.download (posible MultiIndex) o de Ticker.history().
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance devuelve (campo, ticker) o (ticker, campo)
        nivel0 = set(out.columns.get_level_values(0))
        if {'Open', 'Close'} & nivel0:
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    out.columns = [str(c).split('_')[0].strip().title() for c in out.columns]
    requeridas = ['Open', 'High', 'Low', 'Close']
    faltan = [c for c in requeridas if c not in out.columns]
    if faltan:
        raise ValueError(f"Faltan columnas OHLC: {faltan}")
    if 'Volume' not in out.columns:
        out['Volume'] = np.nan
    out = out[requeridas + ['Volume']]
    return out.dropna(subset=requeridas)


def _percentil_movil(serie: pd.Series, ventana: int) -> pd.Series:
    """Percentil (0-100) del valor actual dentro de su ventana móvil."""
    return serie.rolling(ventana, min_periods=max(20, ventana // 4)).rank(pct=True) * 100.0


# ======================================================================
# INDICADORES
# ======================================================================

def rsi_wilder(close: pd.Series, period: int = 13) -> pd.Series:
    """RSI de Wilder (suavizado exponencial alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def bollinger(close: pd.Series, period: int = 30, n_std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger con bandwidth y %B."""
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return pd.DataFrame({
        'bb_mid': mid,
        'bb_up': upper,
        'bb_low': lower,
        'bb_std': sd,
        'bb_width': (2 * n_std * sd) / mid * 100.0,
        'bb_pct_b': (close - lower) / (upper - lower),
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR con suavizado de Wilder."""
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _freq_fin_mes() -> str:
    return "ME" if pd.__version__ >= "2.2" else "M"


def tendencia_mensual(df: pd.DataFrame, lookback: int = 3, min_agree: int = 2) -> pd.Series:
    """
    Estructura mensual de máximos/mínimos. Solo usa meses YA cerrados.
    Devuelve serie diaria con {'alcista','bajista','neutral'}.
    """
    m = df.resample(_freq_fin_mes()).agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    hh = (m['High'].diff() > 0).astype(int)
    hl = (m['Low'].diff() > 0).astype(int)
    lh = (m['High'].diff() < 0).astype(int)
    ll = (m['Low'].diff() < 0).astype(int)

    up = (hh.rolling(lookback).sum() >= min_agree) & (hl.rolling(lookback).sum() >= min_agree)
    down = (lh.rolling(lookback).sum() >= min_agree) & (ll.rolling(lookback).sum() >= min_agree)

    struct = pd.Series('neutral', index=m.index, dtype=object)
    struct[up] = 'alcista'
    struct[down] = 'bajista'
    struct = struct.shift(1)  # anti look-ahead: solo el último mes CERRADO
    return struct.reindex(df.index, method='ffill').fillna('neutral')


def tendencia_completa(df: pd.DataFrame, sma_period: int = 200,
                       tolerancia: float = 0.02) -> pd.Series:
    """
    Estructura mensual + posición vs SMA200 con banda de tolerancia.

    A diferencia de la v1, aquí 'neutral' SÍ existe: si el precio está dentro
    de ±tolerancia de la SMA200 y la estructura mensual no manda, la tendencia
    es neutral (y por tanto se permite operar en ambos sentidos).
    """
    struct = tendencia_mensual(df)
    sma = df['Close'].rolling(sma_period).mean()
    ratio = df['Close'] / sma - 1.0

    encima = ratio > tolerancia
    debajo = ratio < -tolerancia

    trend = pd.Series('neutral', index=df.index, dtype=object)
    trend[(struct == 'alcista') | ((struct == 'neutral') & encima)] = 'alcista'
    trend[(struct == 'bajista') | ((struct == 'neutral') & debajo)] = 'bajista'
    trend[sma.isna()] = 'neutral'
    return trend


def velas_rechazo(df: pd.DataFrame, wick_body_ratio: float = 1.5,
                  wick_range_min: float = 0.33) -> pd.Series:
    """
    Detección VECTORIZADA de velas de rechazo.

    Doble exigencia (la v1 solo tenía la primera):
      - mecha >= cuerpo * wick_body_ratio
      - mecha >= rango * wick_range_min   <- esto elimina los dojis, que con
        cuerpo ~0 pasaban el primer test de forma trivial.
    """
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    body = (c - o).abs()
    rango = (h - l).replace(0.0, np.nan)
    mecha_sup = h - pd.concat([o, c], axis=1).max(axis=1)
    mecha_inf = pd.concat([o, c], axis=1).min(axis=1) - l

    alcista = (
        (mecha_inf >= body * wick_body_ratio)
        & (mecha_inf >= rango * wick_range_min)
        & (c > o)
    )
    bajista = (
        (mecha_sup >= body * wick_body_ratio)
        & (mecha_sup >= rango * wick_range_min)
        & (c < o)
    )

    out = pd.Series(None, index=df.index, dtype=object)
    out[alcista.fillna(False)] = 'alcista'
    out[bajista.fillna(False)] = 'bajista'
    return out


def regimen_volatilidad(close: pd.Series, window: int = 20,
                        lookback: int = 252, externa: pd.Series | None = None) -> pd.DataFrame:
    """
    Régimen de volatilidad. Si se pasa `externa` (p.ej. el cierre del ^VIX),
    se usa esa serie; si no, volatilidad realizada anualizada del propio activo.

    Nagel (2012, RFS): el retorno esperado de la provisión de liquidez —que es
    lo que cobra una estrategia de reversión— sube fuertemente con el VIX.
    """
    if externa is not None:
        base = externa.reindex(close.index).ffill()
        etiqueta = 'externa'
    else:
        ret = np.log(close / close.shift(1))
        base = ret.rolling(window).std() * np.sqrt(252) * 100.0
        etiqueta = 'realizada'
    return pd.DataFrame({
        'vol': base,
        'vol_pct': _percentil_movil(base, lookback),
        'vol_fuente': etiqueta,
    })


# ======================================================================
# FILTROS DE EXCLUSIÓN
# ======================================================================

def filtros_exclusion(df: pd.DataFrame, bb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Filtros anti-breakout, vectorizados. True = PASA el filtro.
    """
    a = atr(df, cfg['atr_period'])
    rango_vela = df['High'] - df['Low']
    ratio_atr = rango_vela / a.replace(0.0, np.nan)
    f_atr = ~(ratio_atr > cfg['max_range_atr'])

    bw_pct = _percentil_movil(bb['bb_width'], cfg['bandwidth_window'])
    f_bw = ~(bw_pct < cfg['min_bandwidth_pct'])

    if df['Volume'].notna().any():
        vol_ma = df['Volume'].rolling(20).mean()
        vol_ratio = df['Volume'] / vol_ma.replace(0.0, np.nan)
        sd = bb['bb_std'].replace(0.0, np.nan)
        dist_up = (df['Close'] - bb['bb_up']) / sd
        dist_low = (bb['bb_low'] - df['Close']) / sd
        lejos = (dist_up > cfg['breakout_dist_std']) | (dist_low > cfg['breakout_dist_std'])
        f_vol = ~((vol_ratio > cfg['volume_breakout_mult']) & lejos)
        vol_euros = (df['Close'] * df['Volume']).rolling(20).mean()
        f_liq = vol_euros >= cfg['min_volumen_medio'] if cfg['min_volumen_medio'] > 0 else True
    else:
        vol_ratio = pd.Series(np.nan, index=df.index)
        f_vol = pd.Series(True, index=df.index)
        f_liq = pd.Series(True, index=df.index)

    out = pd.DataFrame({
        'f_rango_atr': f_atr.fillna(True),
        'f_bandwidth': f_bw.fillna(True),
        'f_volumen': pd.Series(f_vol, index=df.index).fillna(True),
        'f_liquidez': pd.Series(f_liq, index=df.index).fillna(True),
        'ratio_atr': ratio_atr,
        'bw_pct': bw_pct,
        'vol_ratio': vol_ratio,
    })
    out['filtros_ok'] = (
        out['f_rango_atr'] & out['f_bandwidth'] & out['f_volumen'] & out['f_liquidez']
    )
    return out


# ======================================================================
# GENERACIÓN DE SEÑALES
# ======================================================================

def generar_senales(df: pd.DataFrame, cfg: dict | None = None,
                    serie_vol_externa: pd.Series | None = None) -> pd.DataFrame:
    """
    Devuelve un DataFrame indexado por fecha con toda la información de la
    estrategia barra a barra. La columna 'senal' vale 'LONG', 'SHORT' o ''.

    IMPORTANTE: la señal de la barra t se conoce AL CIERRE de t. La entrada
    debe ejecutarse en la apertura de t+1 (de eso se encarga backtest.py).
    """
    cfg = {**CONFIG_V2, **(cfg or {})}
    df = normalizar_ohlc(df)

    close = df['Close']
    rsi = rsi_wilder(close, cfg['rsi_period'])
    bb = bollinger(close, cfg['bb_period'], cfg['bb_std'])
    a = atr(df, cfg['atr_period'])
    trend = tendencia_completa(df, cfg['sma_trend'], cfg['sma_tolerancia'])
    rechazo = velas_rechazo(df, cfg['wick_body_ratio'], cfg['wick_range_min'])
    filtros = filtros_exclusion(df, bb, cfg)
    vol = regimen_volatilidad(close, cfg['vol_window'], cfg['vol_lookback'], serie_vol_externa)

    out = pd.concat([df, bb, filtros, vol], axis=1)
    out['rsi'] = rsi
    out['atr'] = a
    out['sma200'] = close.rolling(cfg['sma_trend']).mean()
    out['tendencia'] = trend
    out['rechazo'] = rechazo

    # --- Filtro de régimen de volatilidad ---
    if cfg['usar_filtro_vol']:
        f_vol_reg = out['vol_pct'] >= cfg['vol_min_pct']
        out['f_regimen_vol'] = f_vol_reg.fillna(False)
    else:
        out['f_regimen_vol'] = True

    ok = out['filtros_ok'] & out['f_regimen_vol']

    # --- Las 4 condiciones, en AND ESTRICTO ---
    c1_long = close <= out['bb_low']
    c2_long = rsi <= cfg['rsi_oversold']
    c3_long = rechazo == 'alcista'
    c4_long = trend != 'bajista'
    cand_long = c1_long & c2_long & c3_long & c4_long & ok

    c1_short = close >= out['bb_up']
    c2_short = rsi >= cfg['rsi_overbought']
    c3_short = rechazo == 'bajista'
    c4_short = trend != 'alcista'
    cand_short = c1_short & c2_short & c3_short & c4_short & ok

    if not cfg['permitir_long']:
        cand_long = pd.Series(False, index=out.index)
    if not cfg['permitir_short']:
        cand_short = pd.Series(False, index=out.index)

    for nombre, serie in [
        ('c1_banda_long', c1_long), ('c2_rsi_long', c2_long),
        ('c3_rechazo_long', c3_long), ('c4_tendencia_long', c4_long),
        ('c1_banda_short', c1_short), ('c2_rsi_short', c2_short),
        ('c3_rechazo_short', c3_short), ('c4_tendencia_short', c4_short),
    ]:
        out[nombre] = serie.fillna(False)

    # --- Niveles: stop, objetivos, R/R ---
    if cfg['stop_mode'] == 'atr':
        stop_long = close - cfg['atr_stop_mult'] * a
        stop_short = close + cfg['atr_stop_mult'] * a
    else:
        stop_long = df['Low'] * (1 - cfg['stop_buffer'])
        stop_short = df['High'] * (1 + cfg['stop_buffer'])

    riesgo_long = (close - stop_long).replace(0.0, np.nan)
    riesgo_short = (stop_short - close).replace(0.0, np.nan)
    rr1_long = (out['bb_mid'] - close) / riesgo_long
    rr1_short = (close - out['bb_mid']) / riesgo_short

    out['stop_ref'] = np.where(cand_short, stop_short, stop_long)
    out['tp1_ref'] = out['bb_mid']
    out['tp2_ref'] = np.where(cand_short, out['bb_low'], out['bb_up'])
    out['rr1'] = np.where(cand_short, rr1_short, rr1_long)

    # --- PUERTA de R/R: sin ratio suficiente NO hay señal ---
    puerta_rr_long = rr1_long >= cfg['min_rr']
    puerta_rr_short = rr1_short >= cfg['min_rr']
    out['pasa_rr'] = np.where(cand_short, puerta_rr_short.fillna(False),
                              puerta_rr_long.fillna(False))

    long_final = cand_long & puerta_rr_long.fillna(False)
    short_final = cand_short & puerta_rr_short.fillna(False)

    out['senal'] = ''
    out.loc[long_final, 'senal'] = 'LONG'
    out.loc[short_final, 'senal'] = 'SHORT'

    # --- Señal "casi": para el panel de vigilancia, no para operar ---
    n_long = (out['c1_banda_long'].astype(int) + out['c2_rsi_long'].astype(int)
              + out['c3_rechazo_long'].astype(int) + out['c4_tendencia_long'].astype(int))
    n_short = (out['c1_banda_short'].astype(int) + out['c2_rsi_short'].astype(int)
               + out['c3_rechazo_short'].astype(int) + out['c4_tendencia_short'].astype(int))
    out['n_cond_long'] = n_long
    out['n_cond_short'] = n_short
    out['vigilar'] = ''
    out.loc[(n_long == 3) & (out['senal'] == ''), 'vigilar'] = 'VIGILAR_LONG'
    out.loc[(n_short == 3) & (out['senal'] == ''), 'vigilar'] = 'VIGILAR_SHORT'

    return out


def explicar_barra(fila: pd.Series, cfg: dict | None = None) -> list[str]:
    """Lista legible del estado de las 4 condiciones + filtros de una barra."""
    cfg = {**CONFIG_V2, **(cfg or {})}
    lado = 'short' if fila.get('senal') == 'SHORT' or fila.get('n_cond_short', 0) > fila.get('n_cond_long', 0) else 'long'
    marca = lambda b: "✅" if bool(b) else "❌"
    etiquetas = {
        'long': [
            (f"Precio ({fila['Close']:.2f}) ≤ banda inferior ({fila['bb_low']:.2f})", fila['c1_banda_long']),
            (f"RSI({cfg['rsi_period']}) = {fila['rsi']:.1f} ≤ {cfg['rsi_oversold']:.0f}", fila['c2_rsi_long']),
            ("Vela de rechazo alcista (martillo)", fila['c3_rechazo_long']),
            (f"Tendencia {fila['tendencia']} (no bajista)", fila['c4_tendencia_long']),
        ],
        'short': [
            (f"Precio ({fila['Close']:.2f}) ≥ banda superior ({fila['bb_up']:.2f})", fila['c1_banda_short']),
            (f"RSI({cfg['rsi_period']}) = {fila['rsi']:.1f} ≥ {cfg['rsi_overbought']:.0f}", fila['c2_rsi_short']),
            ("Vela de rechazo bajista (estrella fugaz)", fila['c3_rechazo_short']),
            (f"Tendencia {fila['tendencia']} (no alcista)", fila['c4_tendencia_short']),
        ],
    }[lado]

    lineas = [f"{marca(v)} {t}" for t, v in etiquetas]
    lineas.append(f"{marca(fila['f_rango_atr'])} Rango vela {fila['ratio_atr']:.2f}x ATR (máx {cfg['max_range_atr']})")
    lineas.append(f"{marca(fila['f_bandwidth'])} Bandwidth en percentil {fila['bw_pct']:.0f} (mín {cfg['min_bandwidth_pct']:.0f})")
    lineas.append(f"{marca(fila['f_volumen'])} Sin ruptura por volumen")
    if cfg['usar_filtro_vol']:
        lineas.append(f"{marca(fila['f_regimen_vol'])} Régimen de volatilidad en percentil {fila['vol_pct']:.0f} (mín {cfg['vol_min_pct']:.0f})")
    rr = fila['rr1']
    lineas.append(f"{marca(fila['pasa_rr'])} R/R hasta banda media = {rr:.2f} (mín {cfg['min_rr']})")
    return lineas
