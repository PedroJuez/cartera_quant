import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio import (
    load_prices, 
    optimal_portfolio, 
    monte_carlo, 
    rebalance,
    risk_metrics,
    efficient_frontier,
    compute_statistics
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cartera Óptima Monte Carlo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

TICKERS = ["AAPL", "MSFT", "BNP.PA", "NVO"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Parámetros")

st.sidebar.subheader("Inversión")
investment = st.sidebar.number_input(
    "Inversión total (€)", 
    min_value=100, 
    max_value=1_000_000, 
    value=10_000, 
    step=500
)

st.sidebar.subheader("Simulación")
months = st.sidebar.slider("Horizonte (meses)", 1, 24, 6)
n_sim = st.sidebar.select_slider(
    "Número de simulaciones",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000
)

st.sidebar.subheader("Optimización")
rf = st.sidebar.slider(
    "Tasa libre de riesgo (%)", 
    0.0, 10.0, 3.0, 0.25
) / 100

optimization_method = st.sidebar.radio(
    "Método de optimización",
    ["scipy", "montecarlo"],
    format_func=lambda x: "Analítico (Scipy)" if x == "scipy" else "Monte Carlo"
)

st.sidebar.subheader("Rebalanceo")
rebalance_threshold = st.sidebar.slider(
    "Umbral de rebalanceo (%)", 
    1, 20, 5
) / 100

# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------
@st.cache_data
def get_prices():
    return load_prices()

try:
    prices = get_prices()
except FileNotFoundError:
    st.error("❌ No se encontró el archivo de datos. Ejecuta primero `download_data.py`")
    st.stop()

# --------------------------------------------------
# TÍTULO Y DESCRIPCIÓN
# --------------------------------------------------
st.title("📊 Cartera Óptima con Simulación Monte Carlo")

st.markdown("""
Esta aplicación encuentra la cartera con **máximo ratio de Sharpe** y simula 
su comportamiento futuro usando el modelo Geométrico Browniano (GBM) con 
correlaciones entre activos.
""")

# --------------------------------------------------
# TABS PRINCIPALES
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cartera Óptima", 
    "🎲 Simulación Monte Carlo",
    "⚖️ Rebalanceo",
    "📉 Frontera Eficiente"
])

# --------------------------------------------------
# TAB 1: CARTERA ÓPTIMA
# --------------------------------------------------
with tab1:
    st.subheader("Optimización de Cartera")
    
    # Calcular cartera óptima
    best = optimal_portfolio(prices, rf, method=optimization_method)
    weights = best["Weights"]
    
    # Mostrar pesos y asignación
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pesos Óptimos")
        allocation_df = pd.DataFrame({
            'Activo': TICKERS,
            'Peso (%)': [f"{w:.1%}" for w in weights],
            'Inversión (€)': [f"{investment * w:,.2f}" for w in weights],
            'Acciones': [f"{investment * w / prices[t].iloc[-1]:.2f}" for t, w in zip(TICKERS, weights)]
        })
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Distribución")
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(TICKERS)))
        wedges, texts, autotexts = ax.pie(
            weights, 
            labels=TICKERS, 
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(TICKERS)
        )
        ax.set_title("Asignación de Activos")
        st.pyplot(fig)
    
    # Métricas
    st.markdown("#### Métricas Anualizadas")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Retorno Esperado", f"{best['Return']:.2%}")
    m2.metric("Volatilidad", f"{best['Vol']:.2%}")
    m3.metric("Ratio de Sharpe", f"{best['Sharpe']:.2f}")
    
    # Estadísticas de activos individuales
    st.markdown("#### Estadísticas de Activos Individuales")
    log_returns, mu, cov = compute_statistics(prices)
    
    stats_df = pd.DataFrame({
        'Activo': TICKERS,
        'Retorno Anual': [f"{mu[t]:.2%}" for t in TICKERS],
        'Volatilidad': [f"{np.sqrt(cov.loc[t, t]):.2%}" for t in TICKERS],
        'Último Precio': [f"${prices[t].iloc[-1]:.2f}" for t in TICKERS]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Matriz de correlación
    st.markdown("#### Matriz de Correlación")
    corr = log_returns.corr()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(TICKERS)))
    ax.set_yticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS)
    ax.set_yticklabels(TICKERS)
    
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
    
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlación de Retornos")
    st.pyplot(fig)

# --------------------------------------------------
# TAB 2: SIMULACIÓN MONTE CARLO
# --------------------------------------------------
with tab2:
    st.subheader("Simulación Monte Carlo")
    
    # Asegurar que tenemos los pesos
    if 'weights' not in dir():
        best = optimal_portfolio(prices, rf, method=optimization_method)
        weights = best["Weights"]
    
    # Ejecutar simulación
    with st.spinner(f"Ejecutando {n_sim:,} simulaciones..."):
        sim_results = monte_carlo(
            prices,
            weights,
            investment,
            n_days=21 * months,
            n_sim=n_sim
        )
    
    returns = sim_results['returns']
    final_values = sim_results['final_values']
    
    # Métricas de riesgo
    metrics = risk_metrics(returns)
    
    st.markdown(f"#### Distribución de Retornos a {months} meses")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(returns, bins=75, density=True, alpha=0.7, color='steelblue')
        
        # Colorear pérdidas en rojo
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('indianred')
        
        # Líneas de referencia
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even')
        ax.axvline(metrics['VaR'], color='red', linestyle='--', linewidth=2, 
                   label=f'VaR 95%: {metrics["VaR"]:.2%}')
        ax.axvline(metrics['mean'], color='green', linestyle='--', linewidth=2,
                   label=f'Media: {metrics["mean"]:.2%}')
        
        ax.set_xlabel('Retorno')
        ax.set_ylabel('Densidad')
        ax.set_title(f'Distribución de Retornos ({n_sim:,} simulaciones)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Métricas de Riesgo**")
        
        st.metric("Retorno Medio", f"{metrics['mean']:.2%}")
        st.metric("Desviación Estándar", f"{metrics['std']:.2%}")
        st.metric("Prob. de Pérdida", f"{metrics['prob_loss']:.1%}")
        st.metric("VaR (95%)", f"{metrics['VaR']:.2%}")
        st.metric("CVaR (Expected Shortfall)", f"{metrics['CVaR']:.2%}")
    
    # Tabla de percentiles
    st.markdown("#### Escenarios por Percentiles")
    
    scenarios = pd.DataFrame({
        'Escenario': ['Muy Pesimista (5%)', 'Pesimista (25%)', 'Mediana (50%)', 
                      'Optimista (75%)', 'Muy Optimista (95%)'],
        'Retorno': [f"{metrics['percentile_5']:.2%}", f"{metrics['percentile_25']:.2%}",
                    f"{metrics['percentile_50']:.2%}", f"{metrics['percentile_75']:.2%}",
                    f"{metrics['percentile_95']:.2%}"],
        'Valor Final (€)': [
            f"{investment * (1 + metrics['percentile_5']):,.2f}",
            f"{investment * (1 + metrics['percentile_25']):,.2f}",
            f"{investment * (1 + metrics['percentile_50']):,.2f}",
            f"{investment * (1 + metrics['percentile_75']):,.2f}",
            f"{investment * (1 + metrics['percentile_95']):,.2f}"
        ],
        'Ganancia/Pérdida (€)': [
            f"{investment * metrics['percentile_5']:,.2f}",
            f"{investment * metrics['percentile_25']:,.2f}",
            f"{investment * metrics['percentile_50']:,.2f}",
            f"{investment * metrics['percentile_75']:,.2f}",
            f"{investment * metrics['percentile_95']:,.2f}"
        ]
    })
    
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    
    # Trayectorias de ejemplo (si están disponibles)
    if 'paths' in sim_results and sim_results['paths'] is not None:
        st.markdown("#### Trayectorias de Ejemplo")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        paths = sim_results['paths']
        days = np.arange(paths.shape[1])
        
        for i in range(min(50, len(paths))):
            color = 'green' if paths[i, -1] > investment else 'red'
            ax.plot(days, paths[i], alpha=0.3, color=color, linewidth=0.5)
        
        ax.axhline(investment, color='blue', linestyle='--', label='Inversión inicial')
        ax.set_xlabel('Días')
        ax.set_ylabel('Valor de la Cartera (€)')
        ax.set_title('Trayectorias Simuladas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# --------------------------------------------------
# TAB 3: REBALANCEO
# --------------------------------------------------
with tab3:
    st.subheader("Análisis de Rebalanceo")
    
    # Asegurar que tenemos los pesos
    if 'weights' not in dir():
        best = optimal_portfolio(prices, rf, method=optimization_method)
        weights = best["Weights"]
    
    # Calcular número de acciones inicial
    S0 = prices.iloc[-1].values
    amounts_invested = investment * weights
    n_shares = amounts_invested / S0
    
    # Verificar rebalanceo
    rebalance_needed, current_weights, deviations = rebalance(
        prices, weights, n_shares, rebalance_threshold
    )
    
    if rebalance_needed:
        st.warning("⚠️ **Rebalanceo recomendado** - Algunas posiciones exceden el umbral")
    else:
        st.success("✅ **No es necesario rebalancear** - Todas las posiciones dentro del umbral")
    
    # Tabla de comparación
    st.markdown("#### Comparación de Pesos")
    
    comparison_df = pd.DataFrame({
        'Activo': TICKERS,
        'Peso Actual': [f"{w:.2%}" for w in current_weights],
        'Peso Objetivo': [f"{w:.2%}" for w in weights],
        'Desviación': [f"{d:.2%}" for d in deviations],
        'Estado': ['🔴 Excede umbral' if d > rebalance_threshold else '🟢 OK' 
                   for d in deviations]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(TICKERS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current_weights * 100, width, label='Actual', color='steelblue')
    bars2 = ax.bar(x + width/2, weights * 100, width, label='Objetivo', color='lightgreen')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Peso (%)')
    ax.set_title('Pesos Actuales vs Objetivo')
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir umbral visual
    for i, (cw, tw) in enumerate(zip(current_weights, weights)):
        ax.fill_between(
            [i - width, i + width], 
            (tw - rebalance_threshold) * 100, 
            (tw + rebalance_threshold) * 100,
            alpha=0.2, color='gray'
        )
    
    st.pyplot(fig)
    
    # Explicación
    st.markdown(f"""
    **Nota:** La zona gris representa el rango aceptable (±{rebalance_threshold:.0%} del objetivo).
    Los pesos que salgan de esta zona disparan la recomendación de rebalanceo.
    """)

# --------------------------------------------------
# TAB 4: FRONTERA EFICIENTE
# --------------------------------------------------
with tab4:
    st.subheader("Frontera Eficiente")
    
    with st.spinner("Calculando frontera eficiente..."):
        frontier = efficient_frontier(prices, rf, n_points=100)
    
    # Asegurar que tenemos los pesos
    if 'best' not in dir():
        best = optimal_portfolio(prices, rf, method=optimization_method)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Frontera eficiente
    ax.plot(
        frontier['Vol'] * 100, 
        frontier['Return'] * 100, 
        'b-', 
        linewidth=2, 
        label='Frontera Eficiente'
    )
    
    # Cartera óptima
    ax.scatter(
        best['Vol'] * 100, 
        best['Return'] * 100, 
        marker='*', 
        s=300, 
        c='red', 
        label=f'Cartera Óptima (Sharpe={best["Sharpe"]:.2f})'
    )
    
    # Activos individuales
    log_returns, mu, cov = compute_statistics(prices)
    for ticker in TICKERS:
        ax.scatter(
            np.sqrt(cov.loc[ticker, ticker]) * 100,
            mu[ticker] * 100,
            marker='o',
            s=100,
            label=ticker
        )
    
    # Capital Market Line
    sharpe_opt = best['Sharpe']
    x_cml = np.linspace(0, frontier['Vol'].max() * 100 * 1.2, 100)
    y_cml = rf * 100 + sharpe_opt * x_cml
    ax.plot(x_cml, y_cml, 'r--', alpha=0.5, label='Capital Market Line')
    
    ax.set_xlabel('Volatilidad (%)')
    ax.set_ylabel('Retorno Esperado (%)')
    ax.set_title('Frontera Eficiente de Markowitz')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretación:**
    - La **frontera eficiente** (línea azul) muestra las carteras con máximo retorno para cada nivel de riesgo.
    - La **estrella roja** es la cartera con máximo Sharpe ratio (mejor relación retorno/riesgo).
    - La **línea roja discontinua** es la Capital Market Line, que conecta el activo libre de riesgo con la cartera óptima.
    - Los **puntos** individuales muestran la posición de cada activo.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<small>
<b>Disclaimer:</b> Esta herramienta es únicamente para fines educativos. 
Los resultados pasados no garantizan rendimientos futuros. 
Consulte con un asesor financiero antes de tomar decisiones de inversión.
</small>
""", unsafe_allow_html=True)
