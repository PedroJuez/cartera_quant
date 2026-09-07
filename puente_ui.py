"""
puente_ui.py — Paneles Streamlit del puente táctico.

Se insertan dentro del modo "📈 Comparador de Activos → ⚖️ Calculadora de Cartera",
después de que Markowitz haya calculado los pesos. No sustituyen nada: añaden.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from seleccion_tactica import (
    CONFIG_TACTICO, estado_tactico, aplicar_score_tactico, plan_de_entrada,
    rebalanceo_tactico, backtest_timing_universo,
)

C_OK = '#16a34a'
C_ESP = '#f59e0b'
C_NO = '#dc2626'


# ======================================================================
# CONFIGURACIÓN EN LA BARRA LATERAL
# ======================================================================

def panel_config_tactico() -> dict:
    """Controles del puente. Llamar una vez, devuelve el cfg."""
    cfg = dict(CONFIG_TACTICO)
    st.sidebar.subheader("🌉 Puente táctico")
    with st.sidebar.expander("Timing de entradas", expanded=False):
        cfg['_activo'] = st.checkbox("Usar la señal de reversión en la selección", value=True)
        cfg['peso_timing'] = st.slider(
            "Peso del timing en la selección (%)", 0, 50,
            int(CONFIG_TACTICO['peso_timing'] * 100), 5,
            help="El score de calidad debe seguir mandando. Por encima del 30% "
                 "estarías dejando que una señal de 10 días decida tu cartera."
        ) / 100
        cfg['zona_barata_pct_b'] = st.slider("Zona barata (%B por debajo de)", 5, 50,
                                             int(CONFIG_TACTICO['zona_barata_pct_b']), 5)
        cfg['zona_cara_pct_b'] = st.slider("Zona cara (%B por encima de)", 50, 95,
                                           int(CONFIG_TACTICO['zona_cara_pct_b']), 5)
        cfg['dias_max_espera'] = st.number_input("Días máximos de espera", 3, 60,
                                                 CONFIG_TACTICO['dias_max_espera'])
        cfg['tramos_escalonado'] = st.number_input("Tramos del escalonado", 2, 6,
                                                   CONFIG_TACTICO['tramos_escalonado'])
    return cfg


# ======================================================================
# ESTADOS
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _estado_cacheado(_df: pd.DataFrame, ticker: str, cfg_key: str) -> dict:
    return estado_tactico(_df)


def calcular_estados(ohlc: dict[str, pd.DataFrame], cfg: dict | None = None) -> dict[str, dict]:
    """Estado táctico de cada activo, con barra de progreso."""
    estados, barra = {}, st.progress(0.0)
    items = list(ohlc.items())
    for i, (tk, df) in enumerate(items):
        try:
            estados[tk] = estado_tactico(df, cfg=cfg)
        except Exception:
            estados[tk] = {'valido': False, 'zona': 'sin datos', 'senal': 'NEUTRAL',
                           'pct_b': np.nan, 'score_timing': 50.0}
        barra.progress((i + 1) / max(len(items), 1))
    barra.empty()
    return estados


# ======================================================================
# PANEL 1 — EFECTO EN LA SELECCIÓN
# ======================================================================

def panel_seleccion(candidatos_originales: list[dict], candidatos_tacticos: list[dict],
                    n_final: int):
    """Muestra qué cambia el timing en la selección. Transparencia total."""
    st.markdown("#### 🌉 Efecto del timing en la selección")

    orden_base = [c['ticker'] for c in sorted(candidatos_originales,
                                              key=lambda x: x.get('score', 0), reverse=True)]
    orden_tact = [c['ticker'] for c in candidatos_tacticos]

    entran = set(orden_tact[:n_final]) - set(orden_base[:n_final])
    salen = set(orden_base[:n_final]) - set(orden_tact[:n_final])

    if not entran and not salen:
        st.info("El timing no cambia la selección: los mismos valores entran en la cartera. "
                "Solo cambia el plan de ejecución.")
    else:
        c1, c2 = st.columns(2)
        c1.success(f"**Entran por timing:** {', '.join(sorted(entran)) or '—'}")
        c2.warning(f"**Salen:** {', '.join(sorted(salen)) or '—'}")

    tabla = pd.DataFrame([{
        'Ticker': c['ticker'],
        'Score calidad': round(c.get('score', np.nan), 1),
        'Score timing': round(c.get('score_timing', np.nan), 1),
        'Score táctico': round(c.get('score_tactico', np.nan), 1),
        'Zona': c.get('zona', '—'),
        '%B': round(c.get('pct_b', np.nan), 1) if np.isfinite(c.get('pct_b', np.nan)) else None,
        'En cartera': '✅' if c['ticker'] in orden_tact[:n_final] else '',
    } for c in candidatos_tacticos])
    st.dataframe(tabla, width='stretch', hide_index=True)


# ======================================================================
# PANEL 2 — PLAN DE ENTRADA
# ======================================================================

def panel_plan_entrada(pesos: dict[str, float], estados: dict[str, dict],
                       importe: float, cfg: dict | None = None):
    st.markdown("#### 📋 Plan de entrada")
    st.caption("Los pesos de Markowitz no se tocan. Lo único que decide este panel "
               "es **cuándo** se ejecuta cada compra.")

    plan = plan_de_entrada(pesos, estados, importe, cfg)
    if plan.empty:
        st.info("Sin posiciones que planificar.")
        return plan

    ahora = plan['Ahora (€)'].sum()
    pend = plan['Pendiente (€)'].sum()
    c = st.columns(3)
    c[0].metric("A comprar ya", f"{ahora:,.0f} €", f"{ahora / importe * 100:.0f}% del total")
    c[1].metric("Pendiente de precio", f"{pend:,.0f} €", f"{pend / importe * 100:.0f}% del total")
    c[2].metric("Posiciones", len(plan))

    def _color(fila):
        if 'AHORA' in fila['Acción']:
            col = C_OK
        elif 'ESPERAR' in fila['Acción']:
            col = C_NO
        else:
            col = C_ESP
        return [f'color: {col}' if c == 'Acción' else '' for c in fila.index]

    vista = plan[['Ticker', 'Peso', 'Importe (€)', 'Zona', '%B', 'Señal',
                  'Acción', 'Ahora (€)', 'Pendiente (€)', 'Precio objetivo', 'Fecha límite']].copy()
    vista['Peso'] = (vista['Peso'] * 100).round(1).astype(str) + '%'
    st.dataframe(vista.style.apply(_color, axis=1).format({
        'Importe (€)': '{:,.0f}', 'Ahora (€)': '{:,.0f}', 'Pendiente (€)': '{:,.0f}',
        '%B': '{:.0f}', 'Precio objetivo': '{:,.2f}',
    }, na_rep='—'), width='stretch', hide_index=True)

    with st.expander("Por qué cada decisión"):
        for _, f in plan.iterrows():
            st.markdown(f"- **{f['Ticker']}** — {f['Motivo']}")

    st.warning("**La fecha límite no es negociable.** Si llega y el precio no ha "
               "bajado, se compra igual. Quedarse fuera del mercado esperando un "
               "precio que no llega cuesta históricamente más que entrar caro.")

    st.download_button("⬇️ Descargar plan (CSV)",
                       plan.to_csv(index=False).encode('utf-8'),
                       "plan_entrada.csv", "text/csv")
    return plan


# ======================================================================
# PANEL 3 — REBALANCEO
# ======================================================================

def panel_rebalanceo(precios: dict[str, float], n_acciones: dict[str, float],
                     pesos_objetivo: dict[str, float], estados: dict[str, dict],
                     cfg: dict | None = None):
    st.markdown("#### ♻️ Rebalanceo con timing")
    reb = rebalanceo_tactico(precios, n_acciones, pesos_objetivo, estados, cfg)
    if reb.empty:
        st.info("Nada que rebalancear.")
        return
    vista = reb.copy()
    for col in ['Peso actual', 'Peso objetivo', 'Desviación']:
        vista[col] = (vista[col] * 100).round(1).astype(str) + '%'
    st.dataframe(vista[['Ticker', 'Peso actual', 'Peso objetivo', 'Desviación',
                        'Acciones a operar', 'Importe (€)', 'Zona', 'Acción',
                        'Prioridad', 'Nota']].round(2),
                 width='stretch', hide_index=True)
    st.caption("«Aplazable» significa que la desviación justifica operar pero el "
               "precio no acompaña: comprar en la banda superior o vender en la "
               "inferior es tirar dinero si la desviación no es urgente.")


# ======================================================================
# PANEL 4 — ¿SIRVE DE ALGO?
# ======================================================================

def panel_validacion(ohlc: dict[str, pd.DataFrame], cfg: dict | None = None):
    """La parte incómoda: medir si esperar mejora algo o es una ilusión."""
    st.markdown("#### 🔬 ¿El puente aporta valor? (validación histórica)")
    st.markdown(
        "Compara dos políticas sobre todo el histórico: **comprar el día que lo "
        "decide la cartera** frente a **esperar a la banda inferior** con fecha "
        "límite. Mide dos cosas: si mejora el precio de entrada y —lo que "
        "realmente importa— si mejora la rentabilidad a 6 meses desde esa entrada. "
        "Un precio de entrada mejor que no mejora el resultado posterior significa "
        "que estás comprando más barato algo que sigue cayendo."
    )
    horizonte = st.select_slider("Horizonte de medición (sesiones)",
                                 [63, 126, 189, 252], value=126,
                                 format_func=lambda x: f"{x} (~{x//21} meses)")

    if not st.button("▶️ Validar el puente", type="primary"):
        return

    with st.spinner("Recorriendo el histórico de cada activo…"):
        tabla = backtest_timing_universo(ohlc, cfg=cfg, horizonte_fwd=int(horizonte))

    if tabla.empty:
        st.warning("No hay histórico suficiente. Se necesitan al menos "
                   f"{260 + int(horizonte)} sesiones por activo.")
        return

    st.dataframe(tabla.round(2), width='stretch', hide_index=True)

    ventaja = tabla['Ventaja (pp)'].dropna()
    if len(ventaja):
        media = float(ventaja.mean())
        favorables = int((ventaja > 0).sum())
        c = st.columns(3)
        c[0].metric("Ventaja media", f"{media:+.2f} pp")
        c[1].metric("Activos favorables", f"{favorables}/{len(ventaja)}")
        c[2].metric("Mejora de precio media", f"{tabla['Mejora precio (%)'].mean():+.2f}%")

        fig, ax = plt.subplots(figsize=(11, 3.6))
        colores = [C_OK if v > 0 else C_NO for v in tabla['Ventaja (pp)']]
        ax.bar(tabla['Ticker'], tabla['Ventaja (pp)'], color=colores, alpha=.85)
        ax.axhline(0, color='#374151', lw=1)
        ax.axhline(media, color='#2563eb', ls='--', lw=1.2, label=f"Media {media:+.2f} pp")
        ax.set_ylabel("Ventaja de esperar (pp)")
        ax.set_title(f"Esperar a la banda vs comprar ya · rentabilidad a {horizonte} sesiones",
                     fontsize=11, weight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2, axis='y')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        if media > 1.0 and favorables > len(ventaja) * 0.6:
            st.success("El puente aporta valor en este universo: esperar mejora el "
                       "resultado posterior de forma consistente.")
        elif media > 0:
            st.warning("Ventaja pequeña y desigual. El puente no hace daño, pero no "
                       "esperes que mueva la aguja. Si la ventaja media está por "
                       "debajo de tus costes de transacción, es ruido.")
        else:
            st.error("En este universo esperar **empeora** el resultado. Lo honesto "
                     "es bajar el peso del timing a 0 y comprar al asignar la cartera. "
                     "El dato manda sobre la intuición.")
