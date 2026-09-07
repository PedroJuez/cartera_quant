"""
adaptador_v1.py — Puente entre el motor v2 y el resto de app.py.

PROBLEMA QUE RESUELVE
---------------------
`analizar_retorno_media_completo()` no vive sola: la consumen cuatro sitios de
app.py, y dos de ellos son los modos "🎯 Recomendación compra/venta" y
"🌍 Análisis por Región". Esos modos esperan un diccionario con una forma muy
concreta y, sobre todo, un vocabulario de señal concreto:

    'LONG' | 'SHORT' | 'VIGILAR_LONG' | 'VIGILAR_SHORT' | 'NEUTRAL' | 'FILTRO_ACTIVO'

porque `ordenar_por_senal()` ordena el desplegable de regiones buscando esas
cadenas dentro del texto. El motor v2 devuelve otro formato (un DataFrame con
columnas 'senal' y 'vigilar'), así que enchufarlo directamente rompería el
escáner regional.

Este módulo traduce. Expone una función con la MISMA firma y la MISMA forma de
salida que la v1, pero calculada con la lógica v2 (AND estricto, puerta de R/R,
stop por ATR, filtro de régimen de volatilidad, vela de rechazo sin dojis).

USO
---
Una sola línea en app.py, después de la definición original:

    from adaptador_v1 import analizar_retorno_media_completo_v2 as _rm_v2
    MOTOR_V2 = True   # ponlo en False para volver a la v1 al instante

    _rm_v1_original = analizar_retorno_media_completo
    def analizar_retorno_media_completo(df, cfg=None, capital=100000):
        if MOTOR_V2:
            return _rm_v2(df, cfg, capital)
        return _rm_v1_original(df, cfg, capital)

A partir de ahí, los cuatro puntos de consumo (Señales de Trading, escáner
regional, detalle de acción por región y comparador) usan el motor nuevo sin
que haya que tocar ni una línea más.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import CONFIG_V2, generar_senales, normalizar_ohlc

# Mínimo de barras para que el análisis tenga sentido. Con SMA200 + percentiles
# de volatilidad a 252 sesiones, un histórico de "1y" deja casi todo en NaN.
MIN_BARRAS = 260


def _traducir_config(cfg_v1: dict | None) -> dict:
    """
    Convierte la ESTRATEGIA_CONFIG de la v1 (o un dict parcial) en config v2.
    Las claves que comparten nombre pasan tal cual; el resto se ignora.
    """
    cfg = dict(CONFIG_V2)
    if not cfg_v1:
        return cfg
    for k, v in cfg_v1.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def _position_sizing(capital: float, entrada: float, stop: float,
                     risk_pct: float, max_position_pct: float) -> dict:
    riesgo_unit = abs(entrada - stop)
    if riesgo_unit < 1e-9 or not np.isfinite(riesgo_unit):
        return {'qty': 0, 'risk_cash': 0, 'position_value': 0,
                'risk_per_unit': 0, 'pct_capital': 0}
    risk_cash = capital * risk_pct
    qty = risk_cash / riesgo_unit
    position_value = qty * entrada
    max_position = capital * max_position_pct
    if position_value > max_position:
        qty = max_position / entrada
        position_value = qty * entrada
        risk_cash = qty * riesgo_unit
    return {
        'qty': qty,
        'risk_cash': risk_cash,
        'position_value': position_value,
        'risk_per_unit': riesgo_unit,
        'pct_capital': position_value / capital * 100 if capital > 0 else 0,
    }


def analizar_retorno_media_completo_v2(df: pd.DataFrame, cfg: dict | None = None,
                                       capital: float = 100_000,
                                       serie_vol_externa: pd.Series | None = None) -> dict:
    """
    Motor v2 con la salida de la v1. Compatible con los cuatro puntos de
    consumo de app.py sin modificar ninguno.
    """
    c = _traducir_config(cfg)

    resultado = {
        'señal': 'NEUTRAL',
        'fuerza': 0,
        'condiciones': [],
        'filtros_ok': True,
        'detalles': {},
        'position_sizing': {},
        'niveles': {},
        'motor': 'v2',
    }

    # --- Validación de datos ---
    try:
        d = normalizar_ohlc(df)
    except Exception as e:
        resultado['condiciones'] = [f"Datos no válidos: {e}"]
        return resultado

    if len(d) < MIN_BARRAS:
        resultado['condiciones'] = [
            f"Datos insuficientes ({len(d)}/{MIN_BARRAS} barras). "
            f"El motor v2 necesita SMA200 y percentiles a 252 sesiones: "
            f"usa un histórico de 2 años o más."
        ]
        return resultado

    sen = generar_senales(d, c, serie_vol_externa)
    u = sen.iloc[-1]

    # --- Traducción del vocabulario de señal ---
    if u['senal'] == 'LONG':
        señal = 'LONG'
    elif u['senal'] == 'SHORT':
        señal = 'SHORT'
    elif not bool(u['filtros_ok']) or not bool(u['f_regimen_vol']):
        señal = 'FILTRO_ACTIVO'
    elif u['vigilar'] == 'VIGILAR_LONG':
        señal = 'VIGILAR_LONG'
    elif u['vigilar'] == 'VIGILAR_SHORT':
        señal = 'VIGILAR_SHORT'
    else:
        señal = 'NEUTRAL'
    resultado['señal'] = señal

    # --- Fuerza 0-100, comparable con la escala de la v1 ---
    lado_long = int(u['n_cond_long']) >= int(u['n_cond_short'])
    n_cond = int(u['n_cond_long'] if lado_long else u['n_cond_short'])
    fuerza = n_cond * 25
    if señal in ('LONG', 'SHORT'):
        fuerza = 100
    elif not bool(u['filtros_ok']) or not bool(u['f_regimen_vol']):
        fuerza = 0
    resultado['fuerza'] = fuerza

    # --- Condiciones legibles ---
    marca = lambda b: "✅" if bool(b) else "❌"
    if lado_long:
        cond = [
            f"{marca(u['c1_banda_long'])} Precio en/bajo banda inferior",
            f"{marca(u['c2_rsi_long'])} RSI en sobreventa ({u['rsi']:.1f})",
            f"{marca(u['c3_rechazo_long'])} Vela de rechazo alcista (martillo)",
            f"{marca(u['c4_tendencia_long'])} Tendencia {u['tendencia']} (no bajista)",
        ]
    else:
        cond = [
            f"{marca(u['c1_banda_short'])} Precio en/sobre banda superior",
            f"{marca(u['c2_rsi_short'])} RSI en sobrecompra ({u['rsi']:.1f})",
            f"{marca(u['c3_rechazo_short'])} Vela de rechazo bajista (estrella fugaz)",
            f"{marca(u['c4_tendencia_short'])} Tendencia {u['tendencia']} (no alcista)",
        ]
    if not bool(u['f_rango_atr']):
        cond.append(f"❌ Filtro: vela de rango extremo ({u['ratio_atr']:.2f}x ATR)")
    if not bool(u['f_bandwidth']):
        cond.append(f"❌ Filtro: bandas estrechas (percentil {u['bw_pct']:.0f})")
    if not bool(u['f_volumen']):
        cond.append("❌ Filtro: ruptura con volumen")
    if not bool(u['f_regimen_vol']):
        cond.append(f"❌ Filtro: volatilidad baja (percentil {u['vol_pct']:.0f})")
    if not bool(u['pasa_rr']) and n_cond == 4:
        cond.append(f"❌ R/R insuficiente ({u['rr1']:.2f} < {c['min_rr']})")
    resultado['condiciones'] = cond

    # --- Filtros con la forma exacta de la v1 ---
    filtros_v1 = {
        'rango_atr': bool(u['f_rango_atr']),
        'bandwidth': bool(u['f_bandwidth']),
        'volume_breakout': bool(u['f_volumen']),
        'regimen_vol': bool(u['f_regimen_vol']),   # nuevo, no molesta a la v1
        'detalles': {
            'rango_atr': f"{u['ratio_atr']:.2f}x ATR" if np.isfinite(u['ratio_atr']) else "n/d",
            'bandwidth': f"{u['bb_width']:.1f}% (percentil {u['bw_pct']:.0f})"
                         if np.isfinite(u['bw_pct']) else "n/d",
            'volumen': f"{u['vol_ratio']:.1f}x media" if np.isfinite(u['vol_ratio']) else "n/d",
            'regimen_vol': f"percentil {u['vol_pct']:.0f}" if np.isfinite(u['vol_pct']) else "n/d",
        },
    }
    resultado['filtros_ok'] = bool(u['filtros_ok']) and bool(u['f_regimen_vol'])

    # --- Detalles con las MISMAS claves que la v1 ---
    tipo_rechazo = u['rechazo'] if isinstance(u['rechazo'], str) else None
    resultado['detalles'] = {
        'precio': float(u['Close']),
        'rsi': float(u['rsi']),
        'bb_upper': float(u['bb_up']),
        'bb_mid': float(u['bb_mid']),
        'bb_lower': float(u['bb_low']),
        'bb_pct_b': float(u['bb_pct_b']) * 100,
        'bb_width': float(u['bb_width']),
        'tendencia': str(u['tendencia']),
        'tendencia_mensual': str(u['tendencia']),
        'sma200': float(u['sma200']) if np.isfinite(u['sma200']) else np.nan,
        'atr': float(u['atr']) if np.isfinite(u['atr']) else np.nan,
        'tipo_rechazo': tipo_rechazo,
        'filtros': filtros_v1,
        # extras del motor v2 (la v1 los ignora sin problema)
        'vol_pct': float(u['vol_pct']) if np.isfinite(u['vol_pct']) else np.nan,
        'n_cond_long': int(u['n_cond_long']),
        'n_cond_short': int(u['n_cond_short']),
    }

    # --- Niveles, solo si hay algo que operar o vigilar ---
    if señal in ('LONG', 'VIGILAR_LONG', 'SHORT', 'VIGILAR_SHORT'):
        es_long = señal.endswith('LONG')
        precio = float(u['Close'])
        atr_v = float(u['atr']) if np.isfinite(u['atr']) else 0.0

        if c['stop_mode'] == 'atr' and atr_v > 0:
            stop = precio - c['atr_stop_mult'] * atr_v if es_long else precio + c['atr_stop_mult'] * atr_v
        else:
            stop = (float(u['Low']) * (1 - c['stop_buffer']) if es_long
                    else float(u['High']) * (1 + c['stop_buffer']))

        obj1 = float(u['bb_mid'])
        obj2 = float(u['bb_up']) if es_long else float(u['bb_low'])
        riesgo = abs(precio - stop)

        if riesgo > 1e-9:
            rr1 = (obj1 - precio) / riesgo if es_long else (precio - obj1) / riesgo
            rr2 = (obj2 - precio) / riesgo if es_long else (precio - obj2) / riesgo
        else:
            rr1 = rr2 = 0.0

        resultado['niveles'] = {
            'entrada': precio,
            'stop': stop,
            'objetivo_parcial': obj1,
            'objetivo_extendido': obj2,
            'riesgo': riesgo,
            'beneficio_parcial': abs(obj1 - precio),
            'beneficio_extendido': abs(obj2 - precio),
            'rr_parcial': rr1,
            'rr_extendido': rr2,
            'pasa_min_rr': rr1 >= c['min_rr'],
        }
        resultado['position_sizing'] = _position_sizing(
            capital, precio, stop, c['risk_per_trade'], c['max_position_pct']
        )

    return resultado


def analizar_retorno_media_v2(hist_df, rsi_period=13, bb_period=30, bb_std=2):
    """Equivalente del wrapper corto de la v1, por si lo necesitas."""
    cfg = {'rsi_period': rsi_period, 'bb_period': bb_period, 'bb_std': bb_std}
    r = analizar_retorno_media_completo_v2(hist_df, cfg)
    det = dict(r['detalles'])
    det['señal'] = r['señal']
    det['razon'] = r['condiciones']
    if r['niveles']:
        det['objetivo'] = r['niveles'].get('objetivo_parcial')
        det['stop'] = r['niveles'].get('stop')
        det['riesgo_beneficio'] = r['niveles'].get('rr_parcial', 0)
    det['bb_position'] = det.get('bb_pct_b', 50)
    return r['señal'], det
