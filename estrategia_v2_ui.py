"""
estrategia_v2_ui.py — Interfaz Streamlit de la Estrategia v2 (Retorno a la Media).

Se engancha a app.py con dos líneas (ver INTEGRACION.md). Toda la lógica vive en
signals.py y backtest.py; aquí solo hay pintura.

Pestañas:
  1. Señal actual   — checklist de las 4 condiciones + niveles + gráfico
  2. Backtest       — equity vs buy & hold, drawdown, distribución de R, operaciones
  3. Walk-forward   — validación fuera de muestra, tramo a tramo
  4. Escáner        — la estrategia sobre una lista de activos, señales vivas

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import yfinance as yf

from signals import CONFIG_V2, generar_senales, normalizar_ohlc, explicar_barra
from backtest import (
    COSTES_DEFAULT, backtest_activo, backtest_cartera, buy_and_hold,
    walk_forward, rejilla_por_defecto,
)

# Paleta coherente con el resto de la app
C_PRECIO = '#1f2937'
C_BANDA = '#7c3aed'
C_MEDIA = '#f59e0b'
C_LONG = '#16a34a'
C_SHORT = '#dc2626'
C_EQ = '#2563eb'
C_BH = '#9ca3af'


# ======================================================================
# DESCARGA
# ======================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_ohlc(ticker: str, periodo: str = "8y") -> pd.DataFrame | None:
    try:
        d = yf.Ticker(ticker).history(period=periodo, auto_adjust=True)
        if d is None or d.empty:
            return None
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return normalizar_ohlc(d)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_vix(periodo: str = "8y") -> pd.Series | None:
    try:
        d = yf.Ticker("^VIX").history(period=periodo, auto_adjust=False)
        if d is None or d.empty:
            return None
        s = d['Close']
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


# ======================================================================
# PANEL DE PARÁMETROS
# ======================================================================

def panel_parametros() -> tuple[dict, dict]:
    """Devuelve (cfg, costes) leídos de la barra lateral."""
    cfg = dict(CONFIG_V2)
    st.sidebar.subheader("⚙️ Parámetros de la estrategia")

    with st.sidebar.expander("Indicadores", expanded=False):
        cfg['rsi_period'] = st.number_input("Periodo RSI", 5, 50, cfg['rsi_period'])
        cfg['rsi_oversold'] = st.slider("RSI sobreventa", 10.0, 45.0, cfg['rsi_oversold'], 1.0)
        cfg['rsi_overbought'] = st.slider("RSI sobrecompra", 55.0, 90.0, cfg['rsi_overbought'], 1.0)
        cfg['bb_period'] = st.number_input("Periodo Bollinger", 10, 100, cfg['bb_period'])
        cfg['bb_std'] = st.slider("Desviaciones Bollinger", 1.0, 3.5, cfg['bb_std'], 0.1)

    with st.sidebar.expander("Tendencia y rechazo", expanded=False):
        cfg['sma_trend'] = st.number_input("SMA de tendencia", 50, 300, cfg['sma_trend'])
        cfg['sma_tolerancia'] = st.slider("Banda neutral alrededor de la SMA (%)", 0.0, 10.0,
                                          cfg['sma_tolerancia'] * 100, 0.5) / 100
        cfg['wick_body_ratio'] = st.slider("Mecha / cuerpo mínimo", 0.5, 3.0, cfg['wick_body_ratio'], 0.1)
        cfg['wick_range_min'] = st.slider("Mecha / rango mínimo", 0.0, 0.7, cfg['wick_range_min'], 0.01)

    with st.sidebar.expander("Filtros", expanded=False):
        cfg['max_range_atr'] = st.slider("Rango máximo de la vela (× ATR)", 1.5, 6.0,
                                         cfg['max_range_atr'], 0.1)
        cfg['min_bandwidth_pct'] = st.slider("Percentil mínimo de bandwidth", 0.0, 60.0,
                                             cfg['min_bandwidth_pct'], 5.0)
        cfg['usar_filtro_vol'] = st.checkbox("Filtro de régimen de volatilidad (Nagel 2012)",
                                             cfg['usar_filtro_vol'])
        cfg['vol_min_pct'] = st.slider("Percentil mínimo de volatilidad", 0.0, 90.0,
                                       cfg['vol_min_pct'], 5.0,
                                       disabled=not cfg['usar_filtro_vol'])
        cfg['_usar_vix'] = st.checkbox("Usar el VIX en vez de la volatilidad del activo",
                                       value=False, disabled=not cfg['usar_filtro_vol'])

    with st.sidebar.expander("Gestión de la operación", expanded=True):
        cfg['stop_mode'] = st.radio("Tipo de stop", ['atr', 'mecha'],
                                    index=0 if cfg['stop_mode'] == 'atr' else 1,
                                    format_func=lambda x: "ATR (recomendado)" if x == 'atr' else "Mínimo de la vela")
        cfg['atr_stop_mult'] = st.slider("Múltiplo de ATR para el stop", 0.5, 5.0,
                                         cfg['atr_stop_mult'], 0.1,
                                         disabled=cfg['stop_mode'] != 'atr')
        cfg['min_rr'] = st.slider("R/R mínimo (puerta de entrada)", 0.5, 4.0, cfg['min_rr'], 0.1)
        cfg['max_barras'] = st.number_input("Salida temporal (barras)", 3, 60, cfg['max_barras'])
        cfg['objetivo_dinamico'] = st.checkbox("Objetivo dinámico (sigue a la banda media)",
                                               cfg['objetivo_dinamico'])
        cfg['risk_per_trade'] = st.slider("Riesgo por operación (%)", 0.1, 5.0,
                                          cfg['risk_per_trade'] * 100, 0.1) / 100
        cfg['max_position_pct'] = st.slider("Exposición máxima por posición (%)", 5.0, 100.0,
                                            cfg['max_position_pct'] * 100, 5.0) / 100
        cfg['permitir_short'] = st.checkbox("Permitir cortos", cfg['permitir_short'],
                                            help="Ni et al. (2020): en la banda superior suele "
                                                 "funcionar mejor momentum que la operativa contraria.")

    st.sidebar.subheader("💸 Costes")
    costes = dict(COSTES_DEFAULT)
    with st.sidebar.expander("Comisiones y slippage", expanded=True):
        costes['comision_pct'] = st.number_input("Comisión (% del nominal)", 0.0, 2.0,
                                                 COSTES_DEFAULT['comision_pct'] * 100, 0.01,
                                                 format="%.3f") / 100
        costes['comision_min'] = st.number_input("Comisión mínima (€)", 0.0, 30.0,
                                                 COSTES_DEFAULT['comision_min'], 0.5)
        costes['slippage_pct'] = st.number_input("Slippage (%)", 0.0, 1.0,
                                                 COSTES_DEFAULT['slippage_pct'] * 100, 0.01,
                                                 format="%.3f") / 100
    return cfg, costes


# ======================================================================
# GRÁFICOS
# ======================================================================

def grafico_precio_senales(sen: pd.DataFrame, cfg: dict, ticker: str,
                           meses: int = 24) -> plt.Figure:
    """Precio + Bollinger + SMA200 + marcas de señal, RSI y régimen de volatilidad."""
    d = sen.iloc[-meses * 21:] if len(sen) > meses * 21 else sen

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax = axes[0]
    ax.plot(d.index, d['Close'], color=C_PRECIO, lw=1.3, label='Cierre', zorder=3)
    ax.plot(d.index, d['bb_up'], color=C_BANDA, lw=0.9, ls='--', alpha=.8, label=f"BB({cfg['bb_period']}) sup")
    ax.plot(d.index, d['bb_mid'], color=C_MEDIA, lw=1.1, label='Banda media (objetivo)')
    ax.plot(d.index, d['bb_low'], color=C_BANDA, lw=0.9, ls='--', alpha=.8, label='BB inf')
    ax.fill_between(d.index, d['bb_low'], d['bb_up'], color=C_BANDA, alpha=.06)
    if d['sma200'].notna().any():
        ax.plot(d.index, d['sma200'], color='#0891b2', lw=1.0, alpha=.9,
                label=f"SMA{cfg['sma_trend']}")

    longs = d[d['senal'] == 'LONG']
    shorts = d[d['senal'] == 'SHORT']
    if len(longs):
        ax.scatter(longs.index, longs['Low'] * 0.985, marker='^', s=110,
                   color=C_LONG, edgecolor='white', lw=.8, zorder=5, label='Señal LONG')
    if len(shorts):
        ax.scatter(shorts.index, shorts['High'] * 1.015, marker='v', s=110,
                   color=C_SHORT, edgecolor='white', lw=.8, zorder=5, label='Señal SHORT')
    vig = d[d['vigilar'] != '']
    if len(vig):
        ax.scatter(vig.index, vig['Low'] * 0.99, marker='o', s=18,
                   facecolor='none', edgecolor='#94a3b8', lw=.8, zorder=4, label='3 de 4 (vigilar)')

    ax.set_title(f"{ticker} · Estrategia de Retorno a la Media v2", fontsize=13, weight='bold')
    ax.set_ylabel("Precio")
    ax.legend(loc='upper left', fontsize=8, ncol=3, framealpha=.9)
    ax.grid(alpha=.25)

    ax2 = axes[1]
    ax2.plot(d.index, d['rsi'], color=C_BANDA, lw=1.2)
    ax2.axhline(cfg['rsi_overbought'], color=C_SHORT, ls='--', lw=1, alpha=.8)
    ax2.axhline(cfg['rsi_oversold'], color=C_LONG, ls='--', lw=1, alpha=.8)
    ax2.axhline(50, color='#9ca3af', ls=':', lw=.8)
    ax2.fill_between(d.index, cfg['rsi_oversold'], cfg['rsi_overbought'], color='#e5e7eb', alpha=.35)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI({cfg['rsi_period']})")
    ax2.grid(alpha=.25)

    ax3 = axes[2]
    ax3.plot(d.index, d['vol_pct'], color='#0f766e', lw=1.2, label='Percentil de volatilidad')
    if cfg['usar_filtro_vol']:
        ax3.axhline(cfg['vol_min_pct'], color=C_SHORT, ls='--', lw=1)
        ax3.fill_between(d.index, 0, cfg['vol_min_pct'], color=C_SHORT, alpha=.07)
        ax3.text(d.index[0], cfg['vol_min_pct'] + 3, "zona sin operativa", fontsize=7, color=C_SHORT)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Vol (pct)")
    ax3.grid(alpha=.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    fig.tight_layout()
    return fig


def grafico_equity(eq: pd.Series, bh: pd.Series | None, dd: pd.Series | None,
                   titulo: str = "Curva de resultados") -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    ax = axes[0]
    ax.plot(eq.index, eq.values, color=C_EQ, lw=1.6, label='Estrategia v2')
    if bh is not None and len(bh):
        b = bh.reindex(eq.index).ffill()
        ax.plot(b.index, b.values, color=C_BH, lw=1.3, ls='--', label='Comprar y mantener')
    ax.set_title(titulo, fontsize=12, weight='bold')
    ax.set_ylabel("Capital (€)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    if dd is None:
        pico = eq.cummax()
        dd = eq / pico - 1
    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values * 100, 0, color=C_SHORT, alpha=.35)
    ax2.plot(dd.index, dd.values * 100, color=C_SHORT, lw=.9)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    fig.tight_layout()
    return fig


def grafico_distribucion_R(ops: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    r = ops['R'].dropna()

    ax = axes[0]
    colores = [C_LONG if x > 0 else C_SHORT for x in r]
    ax.bar(range(len(r)), r.values, color=colores, alpha=.85)
    ax.axhline(0, color='#374151', lw=1)
    ax.axhline(r.mean(), color=C_EQ, ls='--', lw=1.2,
               label=f"Esperanza = {r.mean():+.2f} R")
    ax.set_title("Resultado de cada operación (múltiplos de R)", fontsize=11, weight='bold')
    ax.set_xlabel("Operación")
    ax.set_ylabel("R")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2, axis='y')

    ax2 = axes[1]
    acum = r.cumsum()
    ax2.plot(range(len(acum)), acum.values, color=C_EQ, lw=1.6)
    ax2.fill_between(range(len(acum)), acum.values, 0, color=C_EQ, alpha=.12)
    ax2.axhline(0, color='#374151', lw=1)
    ax2.set_title("R acumulada", fontsize=11, weight='bold')
    ax2.set_xlabel("Operación")
    ax2.set_ylabel("R acumulada")
    ax2.grid(alpha=.2)

    fig.tight_layout()
    return fig


def grafico_rentabilidad_mensual(eq: pd.Series) -> plt.Figure | None:
    """Barras de rentabilidad mensual de la estrategia."""
    if len(eq) < 40:
        return None
    freq = "ME" if pd.__version__ >= "2.2" else "M"
    mensual = eq.resample(freq).last().pct_change().dropna() * 100
    if mensual.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, 3.4))
    colores = [C_LONG if x > 0 else C_SHORT for x in mensual]
    ax.bar(mensual.index, mensual.values, width=20, color=colores, alpha=.85)
    ax.axhline(0, color='#374151', lw=1)
    ax.set_title("Rentabilidad mensual (%)", fontsize=11, weight='bold')
    ax.set_ylabel("%")
    ax.grid(alpha=.2, axis='y')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    fig.tight_layout()
    return fig


# ======================================================================
# BLOQUES DE PRESENTACIÓN
# ======================================================================

def mostrar_metricas(m: dict):
    c = st.columns(5)
    c[0].metric("Operaciones", m.get('n_operaciones', 0))
    c[1].metric("Rentabilidad", f"{m.get('rentabilidad_total_pct', float('nan')):.1f}%")
    c[2].metric("CAGR", f"{m.get('cagr_pct', float('nan')):.1f}%")
    c[3].metric("Máx. drawdown", f"{m.get('max_drawdown_pct', float('nan')):.1f}%")
    c[4].metric("Sharpe", f"{m.get('sharpe', float('nan')):.2f}")

    c = st.columns(5)
    c[0].metric("Aciertos", f"{m.get('win_rate_pct', float('nan')):.1f}%")
    pf = m.get('profit_factor', float('nan'))
    c[1].metric("Profit factor", "∞" if pf == np.inf else f"{pf:.2f}")
    c[2].metric("Esperanza", f"{m.get('expectancy_R', float('nan')):+.3f} R")
    c[3].metric("Esperanza (€)", f"{m.get('expectancy_eur', float('nan')):+.2f} €")
    c[4].metric("Barras medias", f"{m.get('barras_medias', float('nan')):.1f}")

    coste = m.get('coste_sobre_beneficio_pct')
    if coste is not None and np.isfinite(coste):
        aviso = st.error if coste > 40 else (st.warning if coste > 20 else st.info)
        aviso(f"💸 Las comisiones ({m.get('comisiones_totales', 0):.2f} €) se comen el "
              f"**{coste:.1f}%** del beneficio bruto. Por encima del 30-40% la estrategia "
              f"solo trabaja para el bróker.")

    if m.get('n_operaciones', 0) < 30:
        st.warning(f"⚠️ Solo {m.get('n_operaciones', 0)} operaciones. Por debajo de ~30 las "
                   f"métricas no son estadísticamente informativas: no tomes decisiones con esto.")


def mostrar_checklist(sen: pd.DataFrame, cfg: dict):
    ult = sen.iloc[-1]
    senal = ult['senal'] or ult['vigilar'] or 'SIN SEÑAL'

    if ult['senal'] == 'LONG':
        st.success(f"### 🟢 SEÑAL LONG · {sen.index[-1]:%d/%m/%Y}")
    elif ult['senal'] == 'SHORT':
        st.error(f"### 🔴 SEÑAL SHORT · {sen.index[-1]:%d/%m/%Y}")
    elif ult['vigilar']:
        st.warning(f"### 🟡 {ult['vigilar']} · 3 de 4 condiciones · {sen.index[-1]:%d/%m/%Y}")
    else:
        st.info(f"### ⚪ Sin señal · {sen.index[-1]:%d/%m/%Y}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Condiciones y filtros**")
        for linea in explicar_barra(ult, cfg):
            st.markdown(f"- {linea}")

    with col2:
        st.markdown("**Niveles**")
        entrada = ult['Close']
        stop = ult['stop_ref']
        tp1 = ult['tp1_ref']
        tp2 = ult['tp2_ref']
        riesgo = abs(entrada - stop)
        st.markdown(
            f"""
| Concepto | Valor |
|---|---|
| Referencia (cierre) | {entrada:,.2f} |
| Stop ({cfg['stop_mode']}) | {stop:,.2f} |
| Riesgo por unidad | {riesgo:,.2f} |
| Objetivo parcial (banda media) | {tp1:,.2f} |
| Objetivo extendido | {tp2:,.2f} |
| R/R hasta banda media | {ult['rr1']:.2f} |
"""
        )
        st.caption("La entrada se ejecuta en la **apertura de la sesión siguiente**, "
                   "no a este cierre. El backtest asume exactamente eso.")


# ======================================================================
# PUNTO DE ENTRADA
# ======================================================================

def render_estrategia_v2(tickers_disponibles: list[str] | None = None):
    """Renderiza el modo completo. Llamar desde app.py."""
    st.title("🧪 Estrategia de Retorno a la Media · v2")
    st.caption("AND estricto de las 4 condiciones · puerta de R/R · stop por ATR · "
               "filtro de régimen de volatilidad · entrada en la apertura siguiente · "
               "costes y slippage incluidos.")

    cfg, costes = panel_parametros()

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
    with col_b:
        periodo = st.selectbox("Histórico", ["3y", "5y", "8y", "10y", "max"], index=2)
    with col_c:
        capital = st.number_input("Capital (€)", 500, 1_000_000, 10_000, 500)

    if not ticker:
        st.stop()

    with st.spinner(f"Descargando {ticker}…"):
        df = descargar_ohlc(ticker, periodo)
    if df is None or len(df) < max(cfg['sma_trend'], 300):
        st.error(f"No hay datos suficientes para {ticker} (mínimo ~{max(cfg['sma_trend'], 300)} sesiones).")
        st.stop()

    serie_vol = None
    if cfg['usar_filtro_vol'] and cfg.get('_usar_vix'):
        serie_vol = descargar_vix(periodo)
        if serie_vol is None:
            st.warning("No se pudo descargar el ^VIX; se usa la volatilidad realizada del activo.")

    sen = generar_senales(df, cfg, serie_vol)

    t1, t2, t3, t4 = st.tabs([
        "🎯 Señal actual", "📊 Backtest", "🔬 Walk-forward", "🔎 Escáner",
    ])

    # ---------------- 1. SEÑAL ACTUAL ----------------
    with t1:
        mostrar_checklist(sen, cfg)
        st.divider()
        meses = st.slider("Ventana del gráfico (meses)", 6, 60, 24, 3, key="v2_meses")
        st.pyplot(grafico_precio_senales(sen, cfg, ticker, meses), clear_figure=True)

        hist = sen[sen['senal'] != ''][['Close', 'rsi', 'bb_low', 'bb_mid', 'bb_up',
                                        'tendencia', 'senal', 'stop_ref', 'rr1', 'vol_pct']]
        st.markdown(f"**Señales históricas: {len(hist)}** en {len(sen)} sesiones "
                    f"({len(hist) / len(sen) * 100:.2f}% de las barras)")
        if len(hist):
            st.dataframe(hist.tail(30).round(2), width='stretch')
        else:
            st.info("Sin señales con esta configuración. Con el AND estricto es normal que un "
                    "solo activo dé muy pocas: la estrategia está pensada para barrer un "
                    "universo amplio (pestaña Escáner).")

    # ---------------- 2. BACKTEST ----------------
    with t2:
        st.markdown("#### Backtest sobre el histórico completo")
        st.caption("In-sample: sirve para descartar, no para validar. Para validar, "
                   "la pestaña de walk-forward.")
        res = backtest_activo(sen, cfg, capital, costes, ticker=ticker)
        m = res['metricas']
        mostrar_metricas(m)

        if m['n_operaciones'] > 0:
            bh = buy_and_hold({ticker: df}, capital)
            st.pyplot(grafico_equity(res['equity'], bh, m.get('drawdown_serie'),
                                     f"{ticker} · Estrategia v2 vs Comprar y mantener"),
                      clear_figure=True)
            st.pyplot(grafico_distribucion_R(res['operaciones']), clear_figure=True)
            fig_m = grafico_rentabilidad_mensual(res['equity'])
            if fig_m is not None:
                st.pyplot(fig_m, clear_figure=True)

            st.markdown("**Motivos de salida**")
            mot = pd.Series(m.get('motivos_salida', {})).rename("operaciones")
            st.dataframe(mot.to_frame(), width='stretch')

            st.markdown("**Operaciones**")
            cols = ['fecha_entrada', 'fecha_salida', 'lado', 'barras', 'precio_entrada',
                    'precio_salida_medio', 'stop_inicial', 'pnl_neto', 'R',
                    'comisiones', 'motivo_salida', 'parcial']
            st.dataframe(res['operaciones'][cols].round(3), width='stretch')
            st.download_button("⬇️ Descargar operaciones (CSV)",
                               res['operaciones'].to_csv(index=False).encode('utf-8'),
                               f"operaciones_{ticker}.csv", "text/csv")
        else:
            st.info("Sin operaciones que analizar con esta configuración.")

    # ---------------- 3. WALK-FORWARD ----------------
    with t3:
        st.markdown("#### Validación fuera de muestra")
        st.markdown(
            "Optimiza en una ventana, opera la siguiente, avanza y repite. Es lo único que "
            "distingue una estrategia de un ajuste bonito a la historia. Rink (2023) muestra "
            "que las reglas técnicas que mejor funcionan en muestra rinden **peor** que comprar "
            "y mantener fuera de ella."
        )
        c1, c2, c3 = st.columns(3)
        anios_train = c1.number_input("Años de entrenamiento", 2, 8, 3)
        anios_test = c2.number_input("Años de prueba", 1, 3, 1)
        metrica = c3.selectbox("Métrica a optimizar",
                               ['expectancy_R', 'profit_factor', 'cagr_pct', 'sharpe'])

        if st.button("▶️ Ejecutar walk-forward", type="primary"):
            with st.spinner("Optimizando ventana a ventana… puede tardar un minuto."):
                wf = walk_forward(df, rejilla_por_defecto(), cfg, int(anios_train),
                                  int(anios_test), capital, costes, metrica, serie_vol)
            if len(wf['tramos']) == 0:
                st.warning("No se han podido formar tramos con suficientes operaciones. "
                           "Amplía el histórico o relaja los umbrales.")
            else:
                st.dataframe(wf['tramos'], width='stretch')
                mo = wf['metricas_oos']
                if mo:
                    st.markdown("#### Resultado agregado FUERA DE MUESTRA")
                    mostrar_metricas(mo)
                    st.pyplot(grafico_equity(wf['equity_oos'], None, None,
                                             "Curva fuera de muestra (encadenada)"),
                              clear_figure=True)
                if len(wf['operaciones_oos']):
                    st.pyplot(grafico_distribucion_R(wf['operaciones_oos']), clear_figure=True)
                st.info("Si la esperanza en prueba es mucho peor que en entrenamiento tramo tras "
                        "tramo, los parámetros son ruido. Reduce el número de parámetros libres.")

    # ---------------- 4. ESCÁNER ----------------
    with t4:
        st.markdown("#### La estrategia sobre un universo de activos")
        st.caption("Con el AND estricto un solo activo da pocas señales al año. El valor "
                   "está en barrer muchos activos y quedarse con los que hoy cumplen.")
        por_defecto = tickers_disponibles or [
            "SPY", "QQQ", "IWM", "EEM", "EFA",
            "AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "KO", "PG",
            "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "REP.MC",
        ]
        seleccion = st.multiselect("Activos", por_defecto, default=por_defecto[:10])
        modo_bt = st.checkbox("Backtestear todo el universo (más lento)", value=False)

        if st.button("🔎 Escanear", type="primary") and seleccion:
            filas, datos = [], {}
            barra = st.progress(0.0)
            for k, tk in enumerate(seleccion):
                d = descargar_ohlc(tk, periodo)
                barra.progress((k + 1) / len(seleccion))
                if d is None or len(d) < max(cfg['sma_trend'], 300):
                    continue
                datos[tk] = d
                s = generar_senales(d, cfg, serie_vol)
                u = s.iloc[-1]
                filas.append({
                    'Ticker': tk,
                    'Señal': u['senal'] or u['vigilar'] or '—',
                    'Cond. LONG': f"{int(u['n_cond_long'])}/4",
                    'Cond. SHORT': f"{int(u['n_cond_short'])}/4",
                    'Cierre': round(float(u['Close']), 2),
                    'RSI': round(float(u['rsi']), 1),
                    '%B': round(float(u['bb_pct_b']) * 100, 1),
                    'Tendencia': u['tendencia'],
                    'Vol (pct)': round(float(u['vol_pct']), 0) if np.isfinite(u['vol_pct']) else np.nan,
                    'R/R': round(float(u['rr1']), 2) if np.isfinite(u['rr1']) else np.nan,
                    'Filtros': "OK" if bool(u['filtros_ok']) else "bloqueado",
                    'Señales 5a': int((s['senal'] != '').sum()),
                })
            barra.empty()

            if filas:
                tabla = pd.DataFrame(filas)
                orden = {'LONG': 0, 'SHORT': 1, 'VIGILAR_LONG': 2, 'VIGILAR_SHORT': 3, '—': 4}
                tabla = tabla.sort_values('Señal', key=lambda s: s.map(orden).fillna(9))
                st.dataframe(tabla, width='stretch', hide_index=True)
                vivas = tabla[tabla['Señal'].isin(['LONG', 'SHORT'])]
                if len(vivas):
                    st.success(f"🎯 {len(vivas)} señal(es) viva(s) hoy: "
                               f"{', '.join(vivas['Ticker'])}")
                else:
                    st.info("Ninguna señal viva hoy en este universo.")

                if modo_bt and datos:
                    st.divider()
                    st.markdown("#### Backtest agregado del universo")
                    with st.spinner("Backtesteando…"):
                        rc = backtest_cartera(datos, cfg, capital, costes, serie_vol)
                    if rc['metricas']:
                        mostrar_metricas(rc['metricas'])
                        st.pyplot(grafico_equity(rc['equity'], buy_and_hold(datos, capital),
                                                 None, "Universo · Estrategia v2 vs Comprar y mantener"),
                                  clear_figure=True)
                        if len(rc['operaciones']):
                            st.pyplot(grafico_distribucion_R(rc['operaciones']), clear_figure=True)
                            st.dataframe(rc['operaciones'].round(3), width='stretch')
            else:
                st.warning("Ningún activo con datos suficientes.")
