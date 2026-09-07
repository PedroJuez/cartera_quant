"""
backtest.py — Motor de backtest event-driven para la estrategia de Retorno a la Media.

Reglas de honestidad que implementa (y que son las que suelen faltar):
  - La señal se genera al CIERRE de t; la entrada se ejecuta en la APERTURA de t+1.
    Regalarse el cierre de la vela de señal es el error que convierte estrategias
    planas en estrategias "rentables".
  - Si en la misma barra se tocan stop y objetivo, se asume que se tocó el STOP
    (criterio pesimista: sin datos intradía no se puede saber el orden).
  - Comisiones y slippage se aplican en la entrada y en CADA salida (también en
    la parcial).
  - Salida temporal obligatoria: lo que no revierte, no revierte.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import CONFIG_V2, generar_senales

COSTES_DEFAULT = {
    'comision_pct': 0.0010,   # 0,10% del nominal
    'comision_min': 1.00,     # mínimo por operación (€)
    'slippage_pct': 0.0005,   # 0,05% de deslizamiento
}


# ======================================================================
# MOTOR
# ======================================================================

def backtest_activo(sen: pd.DataFrame, cfg: dict | None = None,
                    capital: float = 10_000.0,
                    costes: dict | None = None,
                    ticker: str = "") -> dict:
    """
    Backtest de un único activo sobre el DataFrame que devuelve generar_senales().

    Devuelve dict con 'operaciones' (DataFrame), 'equity' (Series) y 'metricas'.
    """
    cfg = {**CONFIG_V2, **(cfg or {})}
    cst = {**COSTES_DEFAULT, **(costes or {})}

    idx = sen.index
    n = len(sen)
    cash = capital
    pos = None
    operaciones = []
    equity = np.full(n, np.nan)

    o = sen['Open'].to_numpy(float)
    h = sen['High'].to_numpy(float)
    l = sen['Low'].to_numpy(float)
    c = sen['Close'].to_numpy(float)
    mid = sen['bb_mid'].to_numpy(float)
    up = sen['bb_up'].to_numpy(float)
    low = sen['bb_low'].to_numpy(float)
    senal = sen['senal'].to_numpy(object)
    stop_ref = sen['stop_ref'].to_numpy(float)

    def comision(nominal: float) -> float:
        return max(cst['comision_min'], abs(nominal) * cst['comision_pct'])

    def cerrar(precio_bruto: float, fraccion: float, motivo: str, i: int):
        """Cierra una fracción de la posición viva y contabiliza."""
        nonlocal cash, pos
        signo = 1 if pos['lado'] == 'LONG' else -1
        # el slippage siempre juega en contra al salir
        precio = precio_bruto * (1 - signo * cst['slippage_pct'])
        qty = pos['qty_viva'] * fraccion if fraccion < 1.0 else pos['qty_viva']
        nominal = qty * precio
        com = comision(nominal)
        cash += signo * nominal - com
        pos['comisiones'] += com
        pos['qty_viva'] -= qty
        pos['salidas'].append({'fecha': idx[i], 'precio': precio, 'qty': qty, 'motivo': motivo})
        if pos['qty_viva'] <= 1e-9:
            pnl = pos['valor_salida_acum'] = sum(
                s['qty'] * s['precio'] for s in pos['salidas']
            )
            bruto = signo * (pnl - pos['qty_ini'] * pos['precio_entrada'])
            neto = bruto - pos['comisiones']
            riesgo_eur = pos['qty_ini'] * pos['riesgo_unit']
            operaciones.append({
                'ticker': ticker,
                'lado': pos['lado'],
                'fecha_entrada': pos['fecha_entrada'],
                'fecha_salida': idx[i],
                'barras': pos['barras'],
                'precio_entrada': pos['precio_entrada'],
                'precio_salida_medio': pnl / pos['qty_ini'] if pos['qty_ini'] else np.nan,
                'qty': pos['qty_ini'],
                'stop_inicial': pos['stop_ini'],
                'riesgo_eur': riesgo_eur,
                'comisiones': pos['comisiones'],
                'pnl_bruto': bruto,
                'pnl_neto': neto,
                'R': neto / riesgo_eur if riesgo_eur > 0 else np.nan,
                'motivo_salida': motivo,
                'parcial': pos['parcial_hecho'],
            })
            pos = None

    for i in range(n):
        # ---------- 1. Gestión de la posición viva ----------
        if pos is not None:
            pos['barras'] += 1
            signo = 1 if pos['lado'] == 'LONG' else -1

            tp1 = mid[i] if cfg['objetivo_dinamico'] else pos['tp1_ini']
            tp2 = (up[i] if pos['lado'] == 'LONG' else low[i]) if cfg['objetivo_dinamico'] else pos['tp2_ini']

            stop_tocado = (l[i] <= pos['stop']) if pos['lado'] == 'LONG' else (h[i] >= pos['stop'])

            if stop_tocado:
                # Gap: si la apertura ya está pasada del stop, se ejecuta en la apertura
                if pos['lado'] == 'LONG':
                    precio_stop = min(pos['stop'], o[i])
                else:
                    precio_stop = max(pos['stop'], o[i])
                cerrar(precio_stop, 1.0, 'stop' if not pos['parcial_hecho'] else 'stop_be', i)

            elif not pos['parcial_hecho'] and np.isfinite(tp1) and (
                (h[i] >= tp1) if pos['lado'] == 'LONG' else (l[i] <= tp1)
            ):
                cerrar(tp1, cfg['partial_at_middle'] if 'partial_at_middle' in cfg else 0.5,
                       'objetivo_parcial', i)
                if pos is not None:
                    pos['parcial_hecho'] = True
                    pos['stop'] = pos['precio_entrada']  # breakeven

            elif pos['parcial_hecho'] and np.isfinite(tp2) and (
                (h[i] >= tp2) if pos['lado'] == 'LONG' else (l[i] <= tp2)
            ):
                cerrar(tp2, 1.0, 'objetivo_extendido', i)

            elif pos['barras'] >= cfg['max_barras']:
                cerrar(c[i], 1.0, 'salida_temporal', i)

        # ---------- 2. Marca a mercado ----------
        if pos is not None:
            signo = 1 if pos['lado'] == 'LONG' else -1
            equity[i] = cash + signo * pos['qty_viva'] * c[i]
        else:
            equity[i] = cash

        # ---------- 3. ¿Hay señal para entrar MAÑANA? ----------
        if pos is None and i + 1 < n and senal[i] in ('LONG', 'SHORT'):
            lado = senal[i]
            signo = 1 if lado == 'LONG' else -1
            entrada = o[i + 1] * (1 + signo * cst['slippage_pct'])
            stop = stop_ref[i]
            riesgo_unit = abs(entrada - stop)
            if riesgo_unit > 1e-9 and np.isfinite(entrada) and np.isfinite(stop):
                riesgo_cash = equity[i] * cfg['risk_per_trade']
                qty = riesgo_cash / riesgo_unit
                nominal = qty * entrada
                tope = equity[i] * cfg['max_position_pct']
                if nominal > tope:
                    qty = tope / entrada
                    nominal = qty * entrada
                if qty > 0 and nominal > cst['comision_min'] * 10:
                    com = comision(nominal)
                    cash -= signo * nominal + com
                    pos = {
                        'lado': lado,
                        'fecha_entrada': idx[i + 1],
                        'precio_entrada': entrada,
                        'qty_ini': qty,
                        'qty_viva': qty,
                        'stop_ini': stop,
                        'stop': stop,
                        'riesgo_unit': riesgo_unit,
                        'tp1_ini': mid[i],
                        'tp2_ini': up[i] if lado == 'LONG' else low[i],
                        'barras': -1,          # la barra de entrada cuenta como 0
                        'parcial_hecho': False,
                        'comisiones': com,
                        'salidas': [],
                    }

    # Cierre forzoso al final de la muestra
    if pos is not None:
        cerrar(c[n - 1], 1.0, 'fin_muestra', n - 1)
        equity[n - 1] = cash

    eq = pd.Series(equity, index=idx, name='equity').ffill()
    ops = pd.DataFrame(operaciones)
    return {
        'operaciones': ops,
        'equity': eq,
        'metricas': calcular_metricas(ops, eq, capital),
        'ticker': ticker,
    }


# ======================================================================
# MÉTRICAS
# ======================================================================

def calcular_metricas(ops: pd.DataFrame, eq: pd.Series, capital: float) -> dict:
    """Métricas de la curva de resultados y de la distribución de operaciones."""
    m = {
        'n_operaciones': int(len(ops)),
        'capital_inicial': capital,
        'capital_final': float(eq.iloc[-1]) if len(eq) else capital,
    }
    m['rentabilidad_total_pct'] = (m['capital_final'] / capital - 1) * 100

    dias = max((eq.index[-1] - eq.index[0]).days, 1) if len(eq) > 1 else 1
    anios = dias / 365.25
    m['anios'] = anios
    m['cagr_pct'] = ((m['capital_final'] / capital) ** (1 / anios) - 1) * 100 if anios > 0 and m['capital_final'] > 0 else np.nan

    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    m['sharpe'] = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 2 and ret.std() > 0 else np.nan
    neg = ret[ret < 0]
    m['sortino'] = float(ret.mean() / neg.std() * np.sqrt(252)) if len(neg) > 2 and neg.std() > 0 else np.nan
    m['volatilidad_pct'] = float(ret.std() * np.sqrt(252) * 100) if len(ret) > 2 else np.nan

    pico = eq.cummax()
    dd = eq / pico - 1
    m['max_drawdown_pct'] = float(dd.min() * 100) if len(dd) else np.nan
    m['mar'] = m['cagr_pct'] / abs(m['max_drawdown_pct']) if m.get('max_drawdown_pct') and m['max_drawdown_pct'] != 0 else np.nan
    m['drawdown_serie'] = dd

    if len(ops) == 0:
        m.update({'win_rate_pct': np.nan, 'profit_factor': np.nan, 'expectancy_R': np.nan,
                  'expectancy_eur': np.nan, 'R_medio_ganadora': np.nan, 'R_medio_perdedora': np.nan,
                  'barras_medias': np.nan, 'comisiones_totales': 0.0, 'exposicion_pct': 0.0})
        return m

    ganadoras = ops[ops['pnl_neto'] > 0]
    perdedoras = ops[ops['pnl_neto'] <= 0]
    m['win_rate_pct'] = len(ganadoras) / len(ops) * 100
    bruto_g = ganadoras['pnl_neto'].sum()
    bruto_p = abs(perdedoras['pnl_neto'].sum())
    m['profit_factor'] = float(bruto_g / bruto_p) if bruto_p > 0 else np.inf
    m['expectancy_R'] = float(ops['R'].mean())
    m['expectancy_eur'] = float(ops['pnl_neto'].mean())
    m['R_medio_ganadora'] = float(ganadoras['R'].mean()) if len(ganadoras) else np.nan
    m['R_medio_perdedora'] = float(perdedoras['R'].mean()) if len(perdedoras) else np.nan
    m['mejor_R'] = float(ops['R'].max())
    m['peor_R'] = float(ops['R'].min())
    m['barras_medias'] = float(ops['barras'].mean())
    m['comisiones_totales'] = float(ops['comisiones'].sum())
    m['coste_sobre_beneficio_pct'] = (
        m['comisiones_totales'] / abs(ops['pnl_bruto'].sum()) * 100
        if ops['pnl_bruto'].sum() != 0 else np.nan
    )
    m['exposicion_pct'] = float(ops['barras'].sum() / len(eq) * 100) if len(eq) else np.nan
    m['motivos_salida'] = ops['motivo_salida'].value_counts().to_dict()
    return m


# ======================================================================
# CARTERA MULTIACTIVO
# ======================================================================

def backtest_cartera(datos: dict[str, pd.DataFrame], cfg: dict | None = None,
                     capital: float = 10_000.0, costes: dict | None = None,
                     serie_vol_externa: pd.Series | None = None) -> dict:
    """
    Backtest independiente por activo con capital repartido a partes iguales.
    Es una aproximación (no compite por capital entre activos), pero da una
    lectura honesta del comportamiento agregado de la estrategia.

    `datos`: {ticker: DataFrame OHLCV}
    """
    cfg = {**CONFIG_V2, **(cfg or {})}
    if not datos:
        return {'operaciones': pd.DataFrame(), 'equity': pd.Series(dtype=float), 'metricas': {}}

    cap_activo = capital / len(datos)
    resultados, equities, todas_ops = {}, [], []

    for tk, df in datos.items():
        try:
            sen = generar_senales(df, cfg, serie_vol_externa)
            r = backtest_activo(sen, cfg, cap_activo, costes, ticker=tk)
            resultados[tk] = r
            equities.append(r['equity'].rename(tk))
            if len(r['operaciones']):
                todas_ops.append(r['operaciones'])
        except Exception as e:  # un ticker con datos corruptos no tumba la cartera
            resultados[tk] = {'error': str(e)}

    if not equities:
        return {'operaciones': pd.DataFrame(), 'equity': pd.Series(dtype=float),
                'metricas': {}, 'por_activo': resultados}

    eq_total = pd.concat(equities, axis=1).ffill().bfill().sum(axis=1)
    ops = pd.concat(todas_ops, ignore_index=True) if todas_ops else pd.DataFrame()
    return {
        'operaciones': ops.sort_values('fecha_entrada') if len(ops) else ops,
        'equity': eq_total,
        'equity_por_activo': pd.concat(equities, axis=1).ffill(),
        'metricas': calcular_metricas(ops, eq_total, capital),
        'por_activo': resultados,
    }


# ======================================================================
# BUY & HOLD DE REFERENCIA
# ======================================================================

def buy_and_hold(datos: dict[str, pd.DataFrame], capital: float = 10_000.0) -> pd.Series:
    """Curva equiponderada de comprar y mantener, para comparar."""
    curvas = []
    for tk, df in datos.items():
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cl = d['Close'].dropna()
        curvas.append((cl / cl.iloc[0]).rename(tk))
    if not curvas:
        return pd.Series(dtype=float)
    m = pd.concat(curvas, axis=1).ffill().dropna(how='all')
    return (m.mean(axis=1) * capital).rename('buy_and_hold')


# ======================================================================
# WALK-FORWARD
# ======================================================================

def walk_forward(df: pd.DataFrame, rejilla: list[dict], cfg_base: dict | None = None,
                 anios_train: int = 3, anios_test: int = 1,
                 capital: float = 10_000.0, costes: dict | None = None,
                 metrica: str = 'expectancy_R',
                 serie_vol_externa: pd.Series | None = None) -> dict:
    """
    Walk-forward anclado en ventanas móviles: optimiza en `anios_train`, opera el
    siguiente `anios_test`, avanza y repite. Es la única forma decente de saber
    si los parámetros valen algo o son sobreajuste.

    `rejilla`: lista de dicts con los parámetros a probar (solo los que cambian).
    """
    cfg_base = {**CONFIG_V2, **(cfg_base or {})}
    df = df.sort_index()
    inicio, fin = df.index[0], df.index[-1]

    tramos, ops_oos, equities_oos = [], [], []
    t0 = inicio
    while True:
        t_train_fin = t0 + pd.DateOffset(years=anios_train)
        t_test_fin = t_train_fin + pd.DateOffset(years=anios_test)
        if t_train_fin >= fin:
            break
        train = df.loc[t0:t_train_fin]
        test = df.loc[t_train_fin:min(t_test_fin, fin)]
        if len(train) < 300 or len(test) < 40:
            break

        mejor, mejor_val = None, -np.inf
        for params in rejilla:
            cfg = {**cfg_base, **params}
            try:
                sen = generar_senales(train, cfg, serie_vol_externa)
                r = backtest_activo(sen, cfg, capital, costes)
                val = r['metricas'].get(metrica, np.nan)
                if r['metricas']['n_operaciones'] >= 5 and np.isfinite(val) and val > mejor_val:
                    mejor, mejor_val = params, val
            except Exception:
                continue

        if mejor is None:
            t0 = t0 + pd.DateOffset(years=anios_test)
            continue

        cfg = {**cfg_base, **mejor}
        sen_t = generar_senales(df.loc[:min(t_test_fin, fin)], cfg, serie_vol_externa)
        sen_t = sen_t.loc[test.index[0]:test.index[-1]]
        r_oos = backtest_activo(sen_t, cfg, capital, costes)

        tramos.append({
            'train_ini': t0, 'train_fin': t_train_fin,
            'test_ini': test.index[0], 'test_fin': test.index[-1],
            'params': mejor,
            f'{metrica}_train': mejor_val,
            f'{metrica}_test': r_oos['metricas'].get(metrica, np.nan),
            'n_ops_test': r_oos['metricas']['n_operaciones'],
            'rent_test_pct': r_oos['metricas']['rentabilidad_total_pct'],
        })
        if len(r_oos['operaciones']):
            ops_oos.append(r_oos['operaciones'])
        equities_oos.append(r_oos['equity'] / capital)
        t0 = t0 + pd.DateOffset(years=anios_test)

    ops = pd.concat(ops_oos, ignore_index=True) if ops_oos else pd.DataFrame()
    if equities_oos:
        curva = pd.concat([e.pct_change().fillna(0) for e in equities_oos])
        curva = (1 + curva).cumprod() * capital
    else:
        curva = pd.Series(dtype=float)

    return {
        'tramos': pd.DataFrame(tramos),
        'operaciones_oos': ops,
        'equity_oos': curva,
        'metricas_oos': calcular_metricas(ops, curva, capital) if len(curva) else {},
    }


def rejilla_por_defecto() -> list[dict]:
    """Rejilla pequeña y sensata. Cuantos menos parámetros libres, mejor."""
    out = []
    for rsi_p in (9, 13, 21):
        for bb_p in (20, 30):
            for atr_m in (1.5, 2.0, 3.0):
                out.append({'rsi_period': rsi_p, 'bb_period': bb_p, 'atr_stop_mult': atr_m})
    return out
