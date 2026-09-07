"""
seleccion_tactica.py — El puente entre la estrategia de reversión y la cartera.

QUÉ RESUELVE
------------
En "Comparador de Activos → Calculadora de Cartera", app.py ordena los candidatos
solo por `score` (línea 4400) y la señal de retorno a la media acaba en un
`st.caption` decorativo (línea 4502). Se paga el cómputo y se tira el resultado.

Este módulo usa esa señal para tres cosas distintas, y conviene no confundirlas:

  1. SELECCIÓN  — un score táctico que combina el score de calidad con el estado
     de reversión, para desempatar entre candidatos parecidos.
  2. ENTRADA    — un plan de compra por activo: comprar ya, escalonar, o esperar
     a la banda inferior con fecha límite. Los pesos de Markowitz NO se tocan.
  3. REBALANCEO — cuando el umbral de desviación salta, comprar preferentemente
     lo que está barato dentro de sus bandas.

PRINCIPIO DE DISEÑO
-------------------
La reversión a la media NO decide QUÉ comprar ni CUÁNTO. Eso lo deciden el score
fundamental y el optimizador de Markowitz, que es lo correcto para medio y largo
plazo. La reversión decide CUÁNDO. Mezclar las dos cosas sería sustituir una
tesis de inversión por una señal técnica de 10 días, y eso es un error de
categoría.

Por eso `peso_timing` es 0.20 por defecto y no debería subir mucho más: es un
desempate, no un criterio.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import CONFIG_V2, generar_senales, normalizar_ohlc

# ======================================================================
# CONFIGURACIÓN DE SEÑALES PARA HORIZONTE DE CARTERA
# ----------------------------------------------------------------------
# NO se usa CONFIG_V2 aquí, y es deliberado.
#
# CONFIG_V2 está calibrada para operaciones de ~10 días: Bollinger de 30
# sesiones, RSI de 13. Esa ventana responde a la pregunta "¿está barato
# respecto a las últimas seis semanas?", que es la correcta para un trade
# corto y la EQUIVOCADA para una posición que se va a mantener meses.
#
# Para el puente se usan ventanas más lentas, coherentes con el horizonte
# de una cartera de Markowitz:
#   - Bollinger(55) ~ tres meses de referencia
#   - RSI(21)       ~ un mes
#   - SIN filtro de régimen de volatilidad: la prima de liquidez de Nagel
#     es un fenómeno de días; aplicarla a una decisión a seis meses no
#     tiene fundamento y solo bloquearía entradas de forma arbitraria.
# ======================================================================

CONFIG_SENALES_CARTERA = {
    'bb_period': 55,
    'bb_std': 2.0,
    'rsi_period': 21,
    'rsi_oversold': 35.0,
    'rsi_overbought': 65.0,
    'sma_trend': 200,
    'usar_filtro_vol': False,
    'max_range_atr': 4.0,
    'min_bandwidth_pct': 0.0,
}

CONFIG_TACTICO = {
    # --- Selección ---
    'peso_timing': 0.20,          # cuánto pesa el timing frente al score de calidad
    'penalizar_caros': True,      # restar puntos a lo que está pegado a la banda superior

    # --- Zonas ---
    'zona_barata_pct_b': 30.0,    # %B por debajo = zona de compra
    'zona_cara_pct_b': 70.0,      # %B por encima = zona de espera

    # --- Plan de entrada ---
    'dias_max_espera': 15,        # tras esto se compra igual, sin excusas
    'tramos_escalonado': 3,       # nº de compras parciales en zona intermedia

    # --- Rebalanceo ---
    'umbral_rebalanceo': 0.05,
}


# ======================================================================
# 1. ESTADO TÁCTICO DE UN ACTIVO
# ======================================================================

def estado_tactico(df: pd.DataFrame, cfg_senales: dict | None = None,
                   cfg: dict | None = None) -> dict:
    """
    Fotografía del estado de reversión de un activo hoy.
    Devuelve un dict con la señal, la posición en las bandas y un score de
    timing 0-100 (100 = momento inmejorable para entrar largo).
    """
    c = {**CONFIG_TACTICO, **(cfg or {})}
    cs = {**CONFIG_V2, **CONFIG_SENALES_CARTERA, **(cfg_senales or {})}

    vacio = {
        'senal': 'NEUTRAL', 'pct_b': 50.0, 'rsi': 50.0, 'tendencia': 'neutral',
        'score_timing': 50.0, 'zona': 'sin datos', 'precio': np.nan,
        'bb_low': np.nan, 'bb_mid': np.nan, 'bb_up': np.nan,
        'dist_banda_inf_pct': np.nan, 'valido': False,
    }

    try:
        d = normalizar_ohlc(df)
    except Exception:
        return vacio
    if len(d) < max(cs['sma_trend'], 260):
        return vacio

    try:
        sen = generar_senales(d, cs)
    except Exception:
        return vacio

    u = sen.iloc[-1]
    pct_b = float(u['bb_pct_b']) * 100 if np.isfinite(u['bb_pct_b']) else 50.0
    pct_b = float(np.clip(pct_b, -50, 150))
    senal = u['senal'] or u['vigilar'] or 'NEUTRAL'

    # --- Score de timing ---
    # Base: cuanto más abajo en las bandas, mejor momento para comprar.
    base = float(np.clip(100.0 - pct_b, 0, 100))

    bonus = 0.0
    if senal == 'LONG':
        bonus += 15.0
    elif senal == 'VIGILAR_LONG':
        bonus += 8.0
    elif senal == 'SHORT' and c['penalizar_caros']:
        bonus -= 15.0
    elif senal == 'VIGILAR_SHORT' and c['penalizar_caros']:
        bonus -= 8.0

    # Una tendencia principal bajista es mala noticia para una compra de cartera,
    # por muy barato que esté dentro de sus bandas.
    if u['tendencia'] == 'bajista':
        bonus -= 10.0

    score_timing = float(np.clip(base + bonus, 0, 100))

    if pct_b <= c['zona_barata_pct_b']:
        zona = 'barata'
    elif pct_b >= c['zona_cara_pct_b']:
        zona = 'cara'
    else:
        zona = 'intermedia'

    bb_low = float(u['bb_low'])
    precio = float(u['Close'])

    return {
        'senal': senal,
        'pct_b': pct_b,
        'rsi': float(u['rsi']) if np.isfinite(u['rsi']) else 50.0,
        'tendencia': str(u['tendencia']),
        'score_timing': score_timing,
        'zona': zona,
        'precio': precio,
        'bb_low': bb_low,
        'bb_mid': float(u['bb_mid']),
        'bb_up': float(u['bb_up']),
        'dist_banda_inf_pct': (precio / bb_low - 1) * 100 if bb_low > 0 else np.nan,
        'vol_pct': float(u['vol_pct']) if np.isfinite(u['vol_pct']) else np.nan,
        'valido': True,
    }


# ======================================================================
# 2. SELECCIÓN: score táctico
# ======================================================================

def aplicar_score_tactico(candidatos: list[dict], estados: dict[str, dict],
                          cfg: dict | None = None) -> list[dict]:
    """
    Añade 'score_timing' y 'score_tactico' a cada candidato y los devuelve
    ordenados por score_tactico descendente.

    `candidatos`: la lista `todas_acciones` de app.py (dicts con 'ticker' y 'score').
    `estados`:    {ticker: estado_tactico(...)}

    El score de calidad sigue mandando: con peso_timing=0.20, el timing solo
    puede mover un candidato unos pocos puestos, que es justo lo que queremos.
    """
    c = {**CONFIG_TACTICO, **(cfg or {})}
    w = float(np.clip(c['peso_timing'], 0.0, 0.5))

    out = []
    for a in candidatos:
        e = estados.get(a['ticker'])
        st_timing = e['score_timing'] if (e and e['valido']) else 50.0
        b = dict(a)
        b['score_timing'] = st_timing
        b['score_tactico'] = a.get('score', 50.0) * (1 - w) + st_timing * w
        b['zona'] = e['zona'] if (e and e['valido']) else 'sin datos'
        b['pct_b'] = e['pct_b'] if (e and e['valido']) else np.nan
        out.append(b)

    return sorted(out, key=lambda x: x['score_tactico'], reverse=True)


# ======================================================================
# 3. ENTRADA: plan de compra
# ======================================================================

def plan_de_entrada(pesos: dict[str, float], estados: dict[str, dict],
                    importe_total: float, cfg: dict | None = None,
                    fecha_ref: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Convierte los pesos de Markowitz en un plan de ejecución.

    Los pesos NO se modifican. Lo único que se decide aquí es el CUÁNDO:

      - zona barata o señal LONG viva -> comprar el 100% ahora
      - zona intermedia               -> escalonar en N tramos
      - zona cara                     -> esperar a la banda inferior,
                                         con fecha límite; llegada esa fecha se
                                         compra igual (el coste de estar fuera
                                         del mercado supera al de entrar caro)
    """
    c = {**CONFIG_TACTICO, **(cfg or {})}
    fecha_ref = fecha_ref or pd.Timestamp.today().normalize()
    limite = fecha_ref + pd.Timedelta(days=int(c['dias_max_espera']) * 7 / 5)

    filas = []
    for tk, peso in pesos.items():
        if peso <= 0:
            continue
        e = estados.get(tk)
        importe = importe_total * peso

        if not e or not e['valido']:
            filas.append({
                'Ticker': tk, 'Peso': peso, 'Importe (€)': importe,
                'Zona': 'sin datos', '%B': np.nan, 'Señal': '—',
                'Acción': 'COMPRAR AHORA', 'Ahora (€)': importe,
                'Pendiente (€)': 0.0, 'Precio objetivo': np.nan,
                'Fecha límite': '—',
                'Motivo': 'Sin datos suficientes: no se retrasa la compra.',
            })
            continue

        if e['zona'] == 'barata' or e['senal'] == 'LONG':
            accion, ahora, pendiente = 'COMPRAR AHORA', importe, 0.0
            objetivo = np.nan
            fecha_lim = '—'
            motivo = (f"Señal LONG viva." if e['senal'] == 'LONG'
                      else f"%B en {e['pct_b']:.0f}: ya está en la parte baja de sus bandas.")
        elif e['zona'] == 'cara':
            accion = 'ESPERAR'
            ahora = 0.0
            pendiente = importe
            objetivo = e['bb_low']
            fecha_lim = limite.strftime('%d/%m/%Y')
            motivo = (f"%B en {e['pct_b']:.0f}: pegado a la banda superior. "
                      f"Esperar a {objetivo:,.2f} o comprar igual el {fecha_lim}.")
        else:
            n = max(int(c['tramos_escalonado']), 1)
            accion = f'ESCALONAR en {n}'
            ahora = importe / n
            pendiente = importe - ahora
            objetivo = e['bb_mid']
            fecha_lim = limite.strftime('%d/%m/%Y')
            motivo = (f"%B en {e['pct_b']:.0f}: zona intermedia. Primer tramo ya, "
                      f"el resto si cae hacia {objetivo:,.2f} o el {fecha_lim}.")

        filas.append({
            'Ticker': tk, 'Peso': peso, 'Importe (€)': importe,
            'Zona': e['zona'], '%B': e['pct_b'], 'Señal': e['senal'],
            'Acción': accion, 'Ahora (€)': ahora, 'Pendiente (€)': pendiente,
            'Precio objetivo': objetivo, 'Fecha límite': fecha_lim,
            'Motivo': motivo,
        })

    df = pd.DataFrame(filas)
    if len(df):
        df = df.sort_values('Importe (€)', ascending=False).reset_index(drop=True)
    return df


# ======================================================================
# 4. REBALANCEO TÁCTICO
# ======================================================================

def rebalanceo_tactico(precios_actuales: dict[str, float], n_acciones: dict[str, float],
                       pesos_objetivo: dict[str, float], estados: dict[str, dict],
                       cfg: dict | None = None) -> pd.DataFrame:
    """
    Versión con timing del `rebalance()` de portfolio.py.

    Marca qué operaciones son urgentes (desviación alta) y cuáles pueden esperar
    a un mejor precio: comprar lo que está caro dentro de sus bandas puede
    aplazarse; vender lo que está barato, también.
    """
    c = {**CONFIG_TACTICO, **(cfg or {})}
    tickers = [t for t in pesos_objetivo if t in precios_actuales and t in n_acciones]
    if not tickers:
        return pd.DataFrame()

    valores = {t: precios_actuales[t] * n_acciones[t] for t in tickers}
    total = sum(valores.values())
    if total <= 0:
        return pd.DataFrame()

    filas = []
    for t in tickers:
        peso_actual = valores[t] / total
        peso_obj = pesos_objetivo[t]
        desv = peso_actual - peso_obj
        dif_valor = (peso_obj - peso_actual) * total
        dif_acciones = dif_valor / precios_actuales[t]
        e = estados.get(t)
        zona = e['zona'] if (e and e['valido']) else 'sin datos'

        if abs(desv) < c['umbral_rebalanceo']:
            accion, prioridad = 'MANTENER', 'baja'
            nota = f"Desviación {desv*100:+.1f}%, por debajo del umbral."
        elif dif_valor > 0:  # hay que comprar
            if zona == 'cara':
                accion, prioridad = 'COMPRAR (aplazable)', 'media'
                nota = f"Toca comprar pero %B={e['pct_b']:.0f}. Esperar a {e['bb_mid']:,.2f}."
            else:
                accion, prioridad = 'COMPRAR', 'alta'
                nota = f"Toca comprar y el precio acompaña (zona {zona})."
        else:  # hay que vender
            if zona == 'barata':
                accion, prioridad = 'VENDER (aplazable)', 'media'
                nota = f"Toca vender pero %B={e['pct_b']:.0f}: vender en mínimos es caro."
            else:
                accion, prioridad = 'VENDER', 'alta'
                nota = f"Toca vender y el precio acompaña (zona {zona})."

        filas.append({
            'Ticker': t,
            'Peso actual': peso_actual,
            'Peso objetivo': peso_obj,
            'Desviación': desv,
            'Acciones a operar': dif_acciones,
            'Importe (€)': dif_valor,
            'Zona': zona,
            'Acción': accion,
            'Prioridad': prioridad,
            'Nota': nota,
        })

    orden = {'alta': 0, 'media': 1, 'baja': 2}
    return (pd.DataFrame(filas)
            .sort_values(['Prioridad', 'Importe (€)'],
                         key=lambda s: s.map(orden) if s.name == 'Prioridad' else s.abs(),
                         ascending=[True, False])
            .reset_index(drop=True))


# ======================================================================
# 5. ¿SIRVE DE ALGO? — medición honesta del puente
# ======================================================================

def backtest_timing(df: pd.DataFrame, cfg_senales: dict | None = None,
                    cfg: dict | None = None, paso: int = 5,
                    horizonte_fwd: int = 126) -> dict:
    """
    Mide históricamente si esperar mejora el precio de entrada, o si es un cuento.

    Para cada fecha de la muestra (cada `paso` sesiones) compara dos políticas:

      A) COMPRAR YA         -> se compra al cierre de esa sesión.
      B) ESPERAR A LA BANDA -> se espera hasta `dias_max_espera` sesiones a que
                               %B baje del umbral; si ocurre, se compra ahí; si
                               no, se compra al cierre de la sesión límite.

    Reporta la mejora media en el precio de entrada Y —esto es lo importante— la
    rentabilidad a `horizonte_fwd` sesiones desde cada entrada. Un precio de
    entrada mejor que no mejora el resultado a 6 meses no sirve para nada: puede
    ser simplemente que estás comprando activos que siguen cayendo.
    """
    c = {**CONFIG_TACTICO, **(cfg or {})}
    cs = {**CONFIG_V2, **CONFIG_SENALES_CARTERA, **(cfg_senales or {})}
    d = normalizar_ohlc(df)
    if len(d) < max(cs['sma_trend'], 300) + horizonte_fwd:
        return {'n': 0, 'error': 'Histórico insuficiente'}

    sen = generar_senales(d, cs)
    close = sen['Close'].to_numpy(float)
    pct_b = (sen['bb_pct_b'] * 100).to_numpy(float)
    n = len(sen)
    espera = int(c['dias_max_espera'])
    umbral = c['zona_barata_pct_b']

    inicio = max(cs['sma_trend'], 260)
    filas = []
    for i in range(inicio, n - espera - horizonte_fwd, max(paso, 1)):
        p_ya = close[i]
        # política de espera
        j_compra, espero = i + espera, True
        for j in range(i + 1, i + espera + 1):
            if np.isfinite(pct_b[j]) and pct_b[j] <= umbral:
                j_compra = j
                break
        else:
            espero = False  # nunca tocó el umbral: se compró en la fecha límite
        p_esp = close[j_compra]

        filas.append({
            'i': i,
            'fecha': sen.index[i],
            'pct_b_inicial': pct_b[i],
            'precio_ya': p_ya,
            'precio_esperando': p_esp,
            'mejora_precio_pct': (p_ya / p_esp - 1) * 100,   # >0 = esperar salió mejor
            'toco_banda': espero,
            'dias_esperados': j_compra - i,
            'fwd_ya_pct': (close[i + horizonte_fwd] / p_ya - 1) * 100,
            'fwd_esperando_pct': (close[j_compra + horizonte_fwd] / p_esp - 1) * 100
            if j_compra + horizonte_fwd < n else np.nan,
        })

    r = pd.DataFrame(filas)
    if r.empty:
        return {'n': 0, 'error': 'Sin muestras'}

    # Solo tiene sentido evaluar el puente donde realmente cambiaría algo:
    # cuando el activo NO estaba ya barato el día de la decisión.
    relevantes = r[r['pct_b_inicial'] > umbral]

    def _m(s):
        s = s.dropna()
        return float(s.mean()) if len(s) else np.nan

    resumen = {
        'n': int(len(r)),
        'n_relevantes': int(len(relevantes)),
        'pct_veces_toco_banda': float(relevantes['toco_banda'].mean() * 100) if len(relevantes) else np.nan,
        'dias_espera_medios': _m(relevantes['dias_esperados']),
        'mejora_precio_media_pct': _m(relevantes['mejora_precio_pct']),
        'mejora_precio_mediana_pct': float(relevantes['mejora_precio_pct'].median()) if len(relevantes) else np.nan,
        'pct_veces_mejor_precio': float((relevantes['mejora_precio_pct'] > 0).mean() * 100) if len(relevantes) else np.nan,
        'fwd_comprando_ya_pct': _m(relevantes['fwd_ya_pct']),
        'fwd_esperando_pct': _m(relevantes['fwd_esperando_pct']),
        'horizonte_fwd_sesiones': horizonte_fwd,
    }
    resumen['ventaja_fwd_pp'] = (
        resumen['fwd_esperando_pct'] - resumen['fwd_comprando_ya_pct']
        if np.isfinite(resumen['fwd_esperando_pct']) and np.isfinite(resumen['fwd_comprando_ya_pct'])
        else np.nan
    )
    resumen['detalle'] = r
    resumen['veredicto'] = _veredicto(resumen)
    return resumen


def _veredicto(res: dict) -> str:
    v = res.get('ventaja_fwd_pp')
    m = res.get('mejora_precio_media_pct')
    if not np.isfinite(v) or not np.isfinite(m):
        return "Datos insuficientes para concluir."
    if v > 1.0 and m > 0:
        return ("Esperar a la banda inferior mejora el precio de entrada Y la "
                "rentabilidad posterior. El puente aporta valor en este activo.")
    if m > 0 and v <= 1.0:
        return ("Esperar mejora el precio de entrada pero NO la rentabilidad "
                "posterior de forma apreciable. Estás comprando más barato algo "
                "que sigue barato: el puente es cosmético aquí.")
    if m <= 0:
        return ("Esperar EMPEORA el precio medio de entrada: el activo tiende a "
                "irse hacia arriba mientras esperas. En este activo conviene "
                "comprar sin más al asignar la cartera.")
    return "Resultado ambiguo: amplía la muestra antes de concluir."


def backtest_timing_universo(datos: dict[str, pd.DataFrame], cfg_senales: dict | None = None,
                             cfg: dict | None = None, paso: int = 5,
                             horizonte_fwd: int = 126) -> pd.DataFrame:
    """Ejecuta `backtest_timing` sobre varios activos y resume en una tabla."""
    filas = []
    for tk, df in datos.items():
        r = backtest_timing(df, cfg_senales, cfg, paso, horizonte_fwd)
        if r.get('n', 0) == 0:
            continue
        filas.append({
            'Ticker': tk,
            'Muestras': r['n_relevantes'],
            '% tocó banda': r['pct_veces_toco_banda'],
            'Días espera': r['dias_espera_medios'],
            'Mejora precio (%)': r['mejora_precio_media_pct'],
            '% veces mejor': r['pct_veces_mejor_precio'],
            f"Fwd comprando ya (%)": r['fwd_comprando_ya_pct'],
            f"Fwd esperando (%)": r['fwd_esperando_pct'],
            'Ventaja (pp)': r['ventaja_fwd_pp'],
        })
    return pd.DataFrame(filas).sort_values('Ventaja (pp)', ascending=False).reset_index(drop=True) \
        if filas else pd.DataFrame()
