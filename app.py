"""
Cartera Óptima con Simulación Monte Carlo + Análisis Fundamental
=================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cartera Óptima Monte Carlo", 
    layout="wide",
    page_icon="📊"
)

# --------------------------------------------------
# FUNCIONES DE DATOS
# --------------------------------------------------
@st.cache_data(ttl=3600)
def descargar_datos(tickers, periodo="5y"):
    """Descarga datos de Yahoo Finance."""
    try:
        data = yf.download(tickers, period=periodo, progress=False, auto_adjust=True)
        if len(tickers) == 1:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            prices = data['Close']
        return prices.dropna()
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None


@st.cache_data(ttl=3600)
def obtener_info_accion(ticker):
    """Obtiene información fundamental de una acción."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        # Calcular P/FCF si hay datos
        market_cap = info.get('marketCap', 0)
        free_cash_flow = info.get('freeCashflow', 0)
        p_fcf = market_cap / free_cash_flow if free_cash_flow and free_cash_flow > 0 else None
        
        return {
            'info': info,
            'history': hist,
            'p_fcf': p_fcf
        }
    except Exception as e:
        return None


def formatear_numero(num, decimales=2):
    """Formatea números grandes."""
    if num is None:
        return "N/A"
    if abs(num) >= 1e12:
        return f"{num/1e12:.{decimales}f}T"
    elif abs(num) >= 1e9:
        return f"{num/1e9:.{decimales}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{decimales}f}M"
    else:
        return f"{num:,.{decimales}f}"


# --------------------------------------------------
# FUNCIONES DE ANÁLISIS
# --------------------------------------------------
def compute_statistics(prices):
    """Calcula retornos logarítmicos, media y covarianza anualizados."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * 252
    cov = log_returns.cov() * 252
    return log_returns, mu, cov


def optimal_portfolio(prices, rf=0.02):
    """Encuentra la cartera con máximo Sharpe ratio."""
    log_returns, mu, cov = compute_statistics(prices)
    n_assets = len(prices.columns)
    
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret - rf) / vol
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    w0 = np.ones(n_assets) / n_assets
    
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf) / vol
    
    return {'Return': ret, 'Vol': vol, 'Sharpe': sharpe, 'Weights': weights}


def monte_carlo(prices, weights, investment, n_days=21, n_sim=5000):
    """Simula el valor futuro de la cartera usando GBM correlacionado."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu_d = log_returns.mean().values
    cov_d = log_returns.cov().values
    
    try:
        L = np.linalg.cholesky(cov_d)
    except:
        L = np.eye(len(weights))
    
    S0 = prices.iloc[-1].values
    amounts_invested = investment * weights
    n_shares = amounts_invested / S0
    V0 = investment
    
    final_values = []
    
    for _ in range(n_sim):
        prices_sim = S0.copy()
        for _ in range(n_days):
            Z = np.random.standard_normal(len(weights))
            correlated_Z = L @ Z
            prices_sim = prices_sim * np.exp((mu_d - 0.5 * np.diag(cov_d)) + correlated_Z)
        final_values.append(np.dot(prices_sim, n_shares))
    
    final_values = np.array(final_values)
    returns = (final_values - V0) / V0
    
    return {'returns': returns, 'final_values': final_values, 'V0': V0}


def risk_metrics(returns, confidence=0.95):
    """Calcula métricas de riesgo."""
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
    
    return {
        'VaR': var, 'CVaR': cvar,
        'prob_loss': (returns < 0).mean(),
        'percentile_5': np.percentile(returns, 5),
        'percentile_25': np.percentile(returns, 25),
        'percentile_50': np.percentile(returns, 50),
        'percentile_75': np.percentile(returns, 75),
        'percentile_95': np.percentile(returns, 95),
        'mean': returns.mean(), 'std': returns.std()
    }


def efficient_frontier(prices, rf=0.02, n_points=50):
    """Calcula la frontera eficiente."""
    log_returns, mu, cov = compute_statistics(prices)
    n_assets = len(prices.columns)
    
    min_ret, max_ret = mu.min(), mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier = []
    for target in target_returns:
        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(portfolio_vol, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = result.fun
            sharpe = (target - rf) / vol if vol > 0 else 0
            frontier.append({'Return': target, 'Vol': vol, 'Sharpe': sharpe, 'Weights': result.x})
    
    return pd.DataFrame(frontier)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Parámetros")

# Selector de tickers
st.sidebar.subheader("📈 Activos")

tickers_populares = {
    "Tech US": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Europa": ["BNP.PA", "SAP.DE", "ASML.AS", "NVO"],
    "España": ["BBVA.MC", "SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC"],
    "ETFs": ["SPY", "QQQ", "VTI", "IWM"],
    "Bancos": ["BBVA.MC", "SAN.MC", "BNP.PA", "JPM", "BAC"],
}

usar_predefinidos = st.sidebar.checkbox("Usar tickers predefinidos", value=False)

if usar_predefinidos:
    categoria = st.sidebar.selectbox("Categoría", list(tickers_populares.keys()))
    TICKERS = tickers_populares[categoria]
    st.sidebar.write(f"Tickers: {', '.join(TICKERS)}")
else:
    tickers_input = st.sidebar.text_input(
        "Introduce tickers (separados por coma)",
        value="AAPL, MSFT, BBVA.MC, NVO",
        help="Ejemplo: AAPL, MSFT, GOOGL. Para España añade .MC (ej: BBVA.MC)"
    )
    TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

periodo = st.sidebar.selectbox("Período histórico", ["1y", "2y", "3y", "5y", "10y"], index=3)

st.sidebar.markdown("---")

st.sidebar.subheader("💰 Inversión")
investment = st.sidebar.number_input("Inversión total (€)", min_value=100, max_value=1_000_000, value=10_000, step=500)

st.sidebar.subheader("📅 Simulación")
months = st.sidebar.slider("Horizonte (meses)", 1, 24, 6)
n_sim = st.sidebar.select_slider("Simulaciones", options=[1000, 5000, 10000, 25000], value=10000)

st.sidebar.subheader("📊 Optimización")
rf = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 10.0, 3.0, 0.25) / 100

# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------
st.title("📊 Cartera Óptima con Simulación Monte Carlo")

if len(TICKERS) < 2:
    st.error("Necesitas al menos 2 tickers para optimizar una cartera.")
    st.stop()

with st.spinner(f"Descargando datos de {', '.join(TICKERS)}..."):
    prices = descargar_datos(TICKERS, periodo)

if prices is None or prices.empty:
    st.error("No se pudieron descargar los datos. Verifica los tickers.")
    st.stop()

tickers_validos = [t for t in TICKERS if t in prices.columns]
if len(tickers_validos) < len(TICKERS):
    tickers_invalidos = set(TICKERS) - set(tickers_validos)
    st.warning(f"⚠️ No se encontraron datos para: {', '.join(tickers_invalidos)}")
    TICKERS = tickers_validos

if len(TICKERS) < 2:
    st.error("Necesitas al menos 2 tickers válidos.")
    st.stop()

prices = prices[TICKERS]

st.markdown(f"""
**Datos cargados:** {len(prices)} días | **Desde:** {prices.index[0].strftime('%Y-%m-%d')} | **Hasta:** {prices.index[-1].strftime('%Y-%m-%d')}
""")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Cartera Óptima", 
    "🎲 Simulación Monte Carlo", 
    "⚖️ Rebalanceo", 
    "📉 Frontera Eficiente",
    "🔍 Análisis Fundamental"
])

# --------------------------------------------------
# TAB 1: CARTERA ÓPTIMA
# --------------------------------------------------
with tab1:
    st.subheader("Optimización de Cartera")
    
    best = optimal_portfolio(prices, rf)
    weights = best["Weights"]
    
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
        wedges, texts, autotexts = ax.pie(weights, labels=TICKERS, autopct='%1.1f%%', colors=colors, explode=[0.02] * len(TICKERS))
        ax.set_title("Asignación de Activos")
        st.pyplot(fig)
    
    st.markdown("#### Métricas Anualizadas")
    m1, m2, m3 = st.columns(3)
    m1.metric("Retorno Esperado", f"{best['Return']:.2%}")
    m2.metric("Volatilidad", f"{best['Vol']:.2%}")
    m3.metric("Ratio de Sharpe", f"{best['Sharpe']:.2f}")
    
    st.markdown("#### Estadísticas de Activos Individuales")
    log_returns, mu, cov = compute_statistics(prices)
    
    stats_df = pd.DataFrame({
        'Activo': TICKERS,
        'Retorno Anual': [f"{mu[t]:.2%}" for t in TICKERS],
        'Volatilidad': [f"{np.sqrt(cov.loc[t, t]):.2%}" for t in TICKERS],
        'Último Precio': [f"${prices[t].iloc[-1]:.2f}" for t in TICKERS]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
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
    
    with st.spinner(f"Ejecutando {n_sim:,} simulaciones..."):
        sim_results = monte_carlo(prices, weights, investment, n_days=21 * months, n_sim=n_sim)
    
    returns = sim_results['returns']
    metrics = risk_metrics(returns)
    
    st.markdown(f"#### Distribución de Retornos a {months} meses")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(returns, bins=75, density=True, alpha=0.7, color='steelblue')
        
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('indianred')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even')
        ax.axvline(metrics['VaR'], color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {metrics["VaR"]:.2%}')
        ax.axvline(metrics['mean'], color='green', linestyle='--', linewidth=2, label=f'Media: {metrics["mean"]:.2%}')
        
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
    
    st.markdown("#### Escenarios por Percentiles")
    scenarios = pd.DataFrame({
        'Escenario': ['Muy Pesimista (5%)', 'Pesimista (25%)', 'Mediana (50%)', 'Optimista (75%)', 'Muy Optimista (95%)'],
        'Retorno': [f"{metrics['percentile_5']:.2%}", f"{metrics['percentile_25']:.2%}", f"{metrics['percentile_50']:.2%}", f"{metrics['percentile_75']:.2%}", f"{metrics['percentile_95']:.2%}"],
        'Valor Final (€)': [f"{investment * (1 + metrics['percentile_5']):,.2f}", f"{investment * (1 + metrics['percentile_25']):,.2f}", f"{investment * (1 + metrics['percentile_50']):,.2f}", f"{investment * (1 + metrics['percentile_75']):,.2f}", f"{investment * (1 + metrics['percentile_95']):,.2f}"],
        'Ganancia/Pérdida (€)': [f"{investment * metrics['percentile_5']:,.2f}", f"{investment * metrics['percentile_25']:,.2f}", f"{investment * metrics['percentile_50']:,.2f}", f"{investment * metrics['percentile_75']:,.2f}", f"{investment * metrics['percentile_95']:,.2f}"]
    })
    st.dataframe(scenarios, use_container_width=True, hide_index=True)

# --------------------------------------------------
# TAB 3: REBALANCEO
# --------------------------------------------------
with tab3:
    st.subheader("Análisis de Rebalanceo")
    
    rebalance_threshold = st.slider("Umbral de rebalanceo (%)", 1, 20, 5) / 100
    
    S0 = prices.iloc[-1].values
    amounts_invested = investment * weights
    n_shares = amounts_invested / S0
    
    position_values = S0 * n_shares
    total_value = position_values.sum()
    current_weights = position_values / total_value
    deviations = np.abs(current_weights - weights)
    rebalance_needed = (deviations > rebalance_threshold).any()
    
    if rebalance_needed:
        st.warning("⚠️ **Rebalanceo recomendado**")
    else:
        st.success("✅ **No es necesario rebalancear**")
    
    st.markdown("#### Comparación de Pesos")
    comparison_df = pd.DataFrame({
        'Activo': TICKERS,
        'Peso Actual': [f"{w:.2%}" for w in current_weights],
        'Peso Objetivo': [f"{w:.2%}" for w in weights],
        'Desviación': [f"{d:.2%}" for d in deviations],
        'Estado': ['🔴 Excede umbral' if d > rebalance_threshold else '🟢 OK' for d in deviations]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TICKERS))
    width = 0.35
    ax.bar(x - width/2, current_weights * 100, width, label='Actual', color='steelblue')
    ax.bar(x + width/2, weights * 100, width, label='Objetivo', color='lightgreen')
    ax.set_ylabel('Peso (%)')
    ax.set_title('Pesos Actuales vs Objetivo')
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

# --------------------------------------------------
# TAB 4: FRONTERA EFICIENTE
# --------------------------------------------------
with tab4:
    st.subheader("Frontera Eficiente")
    
    with st.spinner("Calculando frontera eficiente..."):
        frontier = efficient_frontier(prices, rf, n_points=100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frontier['Vol'] * 100, frontier['Return'] * 100, 'b-', linewidth=2, label='Frontera Eficiente')
    ax.scatter(best['Vol'] * 100, best['Return'] * 100, marker='*', s=300, c='red', label=f'Cartera Óptima (Sharpe={best["Sharpe"]:.2f})')
    
    log_returns, mu, cov = compute_statistics(prices)
    for ticker in TICKERS:
        ax.scatter(np.sqrt(cov.loc[ticker, ticker]) * 100, mu[ticker] * 100, marker='o', s=100, label=ticker)
    
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

# --------------------------------------------------
# TAB 5: ANÁLISIS FUNDAMENTAL
# --------------------------------------------------
with tab5:
    st.subheader("🔍 Análisis Fundamental")
    
    st.markdown("""
    Selecciona una acción para ver su gráfico de cotización y ratios fundamentales.
    """)
    
    # Selector de acción
    ticker_seleccionado = st.selectbox("Selecciona una acción", TICKERS)
    
    with st.spinner(f"Cargando datos de {ticker_seleccionado}..."):
        data_accion = obtener_info_accion(ticker_seleccionado)
    
    if data_accion is None:
        st.error(f"No se pudieron obtener datos para {ticker_seleccionado}")
    else:
        info = data_accion['info']
        hist = data_accion['history']
        
        # Información general
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {info.get('longName', ticker_seleccionado)}")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industria:** {info.get('industry', 'N/A')}")
            st.markdown(f"**País:** {info.get('country', 'N/A')} | **Moneda:** {info.get('currency', 'N/A')}")
            
            if info.get('longBusinessSummary'):
                with st.expander("📋 Descripción de la empresa"):
                    st.write(info.get('longBusinessSummary'))
        
        with col2:
            precio_actual = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            cambio = info.get('regularMarketChangePercent', 0)
            st.metric(
                "Precio Actual",
                f"{precio_actual:.2f} {info.get('currency', '')}",
                f"{cambio:.2f}%"
            )
        
        st.markdown("---")
        
        # Gráfico de cotización
        st.markdown("#### 📈 Cotización Histórica (1 año)")
        
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            ax.plot(hist.index, hist['Close'], 'b-', linewidth=1.5, label='Precio de Cierre')
            
            # Media móvil 50 días
            if len(hist) >= 50:
                ma50 = hist['Close'].rolling(window=50).mean()
                ax.plot(hist.index, ma50, 'orange', linewidth=1, label='MA 50', alpha=0.8)
            
            # Media móvil 200 días
            if len(hist) >= 200:
                ma200 = hist['Close'].rolling(window=200).mean()
                ax.plot(hist.index, ma200, 'red', linewidth=1, label='MA 200', alpha=0.8)
            
            ax.fill_between(hist.index, hist['Low'], hist['High'], alpha=0.1, color='blue')
            
            ax.set_xlabel('Fecha')
            ax.set_ylabel(f'Precio ({info.get("currency", "USD")})')
            ax.set_title(f'{ticker_seleccionado} - Cotización')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Formato de fechas
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Estadísticas de precio
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mínimo 52 sem", f"{info.get('fiftyTwoWeekLow', 'N/A')}")
            col2.metric("Máximo 52 sem", f"{info.get('fiftyTwoWeekHigh', 'N/A')}")
            col3.metric("Media 50 días", f"{info.get('fiftyDayAverage', 0):.2f}")
            col4.metric("Media 200 días", f"{info.get('twoHundredDayAverage', 0):.2f}")
        
        st.markdown("---")
        
        # Ratios de valoración
        st.markdown("#### 📊 Ratios de Valoración")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # PER
        per = info.get('trailingPE')
        per_forward = info.get('forwardPE')
        with col1:
            st.markdown("**PER (Precio/Beneficio)**")
            st.markdown(f"### {per:.2f}" if per else "### N/A")
            if per_forward:
                st.caption(f"Forward PER: {per_forward:.2f}")
            st.caption("Años de beneficios que pagas al comprar hoy")
        
        # EV/EBITDA
        ev_ebitda = info.get('enterpriseToEbitda')
        with col2:
            st.markdown("**EV/EBITDA**")
            st.markdown(f"### {ev_ebitda:.2f}" if ev_ebitda else "### N/A")
            st.caption("Valor empresa vs beneficio operativo")
        
        # P/FCF
        p_fcf = data_accion['p_fcf']
        with col3:
            st.markdown("**P/FCF (Precio/Flujo Caja)**")
            st.markdown(f"### {p_fcf:.2f}" if p_fcf else "### N/A")
            st.caption("Más limpio que el PER")
        
        # P/BV
        p_bv = info.get('priceToBook')
        with col4:
            st.markdown("**P/BV (Precio/Valor Contable)**")
            st.markdown(f"### {p_bv:.2f}" if p_bv else "### N/A")
            st.caption("Importante en bancos y aseguradoras")
        
        st.markdown("---")
        
        # Interpretación de ratios
        with st.expander("📖 ¿Cómo interpretar los ratios?"):
            st.markdown("""
            | Ratio | Bajo | Medio | Alto | Interpretación |
            |-------|------|-------|------|----------------|
            | **PER** | <10 | 10-20 | >25 | PER bajo puede indicar infravaloración o problemas; alto puede indicar crecimiento esperado |
            | **EV/EBITDA** | <6 | 6-12 | >15 | Útil para comparar empresas del mismo sector |
            | **P/FCF** | <10 | 10-20 | >25 | Similar al PER pero basado en caja real |
            | **P/BV** | <1 | 1-3 | >3 | P/BV < 1 puede indicar infravaloración (común en bancos) |
            
            ⚠️ **Importante:** Siempre compara ratios con empresas del mismo sector. Un PER "alto" en tecnología puede ser normal, mientras que sería preocupante en utilities.
            """)
        
        st.markdown("---")
        
        # Más métricas financieras
        st.markdown("#### 💰 Métricas Financieras")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Capitalización**")
            market_cap = info.get('marketCap', 0)
            st.markdown(f"### {formatear_numero(market_cap)}")
            
            st.markdown("**Ingresos (TTM)**")
            revenue = info.get('totalRevenue', 0)
            st.markdown(f"### {formatear_numero(revenue)}")
        
        with col2:
            st.markdown("**EBITDA**")
            ebitda = info.get('ebitda', 0)
            st.markdown(f"### {formatear_numero(ebitda)}")
            
            st.markdown("**Beneficio Neto**")
            net_income = info.get('netIncomeToCommon', 0)
            st.markdown(f"### {formatear_numero(net_income)}")
        
        with col3:
            st.markdown("**Margen Operativo**")
            op_margin = info.get('operatingMargins', 0)
            st.markdown(f"### {op_margin*100:.1f}%" if op_margin else "### N/A")
            
            st.markdown("**ROE**")
            roe = info.get('returnOnEquity', 0)
            st.markdown(f"### {roe*100:.1f}%" if roe else "### N/A")
        
        st.markdown("---")
        
        # Dividendos
        st.markdown("#### 💵 Dividendos")
        
        col1, col2, col3 = st.columns(3)
        
        div_yield = info.get('dividendYield', 0)
        div_rate = info.get('dividendRate', 0)
        payout = info.get('payoutRatio', 0)
        
        col1.metric("Rentabilidad por dividendo", f"{div_yield*100:.2f}%" if div_yield else "N/A")
        col2.metric("Dividendo anual", f"{div_rate:.2f} {info.get('currency', '')}" if div_rate else "N/A")
        col3.metric("Payout Ratio", f"{payout*100:.1f}%" if payout else "N/A")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(f"""
<small>
<b>Última actualización:</b> {prices.index[-1].strftime('%Y-%m-%d')} | 
<b>Tickers:</b> {', '.join(TICKERS)} |
<b>Disclaimer:</b> Esta herramienta es únicamente para fines educativos.
</small>
""", unsafe_allow_html=True)
