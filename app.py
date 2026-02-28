"""
Análisis de Inversiones - Cartera y Acción Individual
======================================================
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
    page_title="Análisis de Inversiones", 
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
def obtener_info_accion(ticker, periodo="1y"):
    """Obtiene información fundamental de una acción."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=periodo)
        
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


@st.cache_data(ttl=86400)
def buscar_ticker(nombre_empresa):
    """Busca tickers por nombre de empresa."""
    try:
        import requests
        # Usar la API de búsqueda de Yahoo Finance
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={nombre_empresa}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        resultados = []
        if 'quotes' in data:
            for quote in data['quotes']:
                if quote.get('quoteType') in ['EQUITY', 'ETF']:
                    resultados.append({
                        'ticker': quote.get('symbol', ''),
                        'nombre': quote.get('longname') or quote.get('shortname', ''),
                        'bolsa': quote.get('exchange', ''),
                        'tipo': quote.get('quoteType', '')
                    })
        return resultados
    except Exception as e:
        return []


# Diccionario de empresas comunes (backup)
EMPRESAS_COMUNES = {
    # España
    "Santander": "SAN.MC", "BBVA": "BBVA.MC", "Inditex": "ITX.MC", "Iberdrola": "IBE.MC",
    "Telefonica": "TEF.MC", "Repsol": "REP.MC", "Caixabank": "CABK.MC", "Naturgy": "NTGY.MC",
    "Ferrovial": "FER.MC", "Amadeus": "AMS.MC", "Aena": "AENA.MC", "Cellnex": "CLNX.MC",
    "Grifols": "GRF.MC", "Endesa": "ELE.MC", "Mapfre": "MAP.MC", "Sabadell": "SAB.MC",
    # USA Tech
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN",
    "Meta": "META", "Tesla": "TSLA", "Nvidia": "NVDA", "Netflix": "NFLX",
    "Adobe": "ADBE", "Salesforce": "CRM", "Intel": "INTC", "AMD": "AMD",
    "Paypal": "PYPL", "Uber": "UBER", "Airbnb": "ABNB", "Spotify": "SPOT",
    # USA Otros
    "Coca Cola": "KO", "Pepsi": "PEP", "McDonalds": "MCD", "Nike": "NKE",
    "Disney": "DIS", "Visa": "V", "Mastercard": "MA", "JPMorgan": "JPM",
    "Bank of America": "BAC", "Goldman Sachs": "GS", "Pfizer": "PFE", "Johnson Johnson": "JNJ",
    "Walmart": "WMT", "Costco": "COST", "Home Depot": "HD", "Procter Gamble": "PG",
    "Exxon": "XOM", "Chevron": "CVX", "Boeing": "BA", "Caterpillar": "CAT",
    # Europa
    "LVMH": "MC.PA", "Nestle": "NESN.SW", "Novartis": "NOVN.SW", "Roche": "ROG.SW",
    "ASML": "ASML.AS", "SAP": "SAP.DE", "Siemens": "SIE.DE", "Volkswagen": "VOW3.DE",
    "BMW": "BMW.DE", "Mercedes": "MBG.DE", "BNP Paribas": "BNP.PA", "Total": "TTE.PA",
    "LOreal": "OR.PA", "Airbus": "AIR.PA", "Novo Nordisk": "NVO", "Shell": "SHEL",
    # ETFs
    "SP500 ETF": "SPY", "Nasdaq ETF": "QQQ", "Dow Jones ETF": "DIA", "Russell 2000": "IWM",
    "Emerging Markets": "EEM", "Europe ETF": "VGK", "Bond ETF": "BND", "Gold ETF": "GLD",
}


# --------------------------------------------------
# FUNCIONES DE ANÁLISIS
# --------------------------------------------------
def compute_statistics(prices):
    """Calcula retornos logarítmicos, media y covarianza anualizados."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * 252
    cov = log_returns.cov() * 252
    return log_returns, mu, cov


def optimal_portfolio(prices, rf=0.02, max_weight=1.0):
    """Encuentra la cartera con máximo Sharpe ratio."""
    log_returns, mu, cov = compute_statistics(prices)
    n_assets = len(prices.columns)
    
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret - rf) / vol if vol > 0 else 0
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    w0 = np.ones(n_assets) / n_assets
    
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    
    # Normalizar por si acaso
    weights = weights / weights.sum()
    
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    
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


def efficient_frontier(prices, rf=0.02, n_points=50, max_weight=1.0):
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
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(portfolio_vol, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = result.fun
            sharpe = (target - rf) / vol if vol > 0 else 0
            frontier.append({'Return': target, 'Vol': vol, 'Sharpe': sharpe, 'Weights': result.x})
    
    return pd.DataFrame(frontier)


# --------------------------------------------------
# FUNCIONES DE ANÁLISIS TÉCNICO
# --------------------------------------------------
def calcular_rsi(prices, period=14):
    """Calcula el RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_macd(prices, fast=12, slow=26, signal=9):
    """Calcula MACD y línea de señal."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calcular_soportes_resistencias(prices, window=20):
    """Identifica soportes y resistencias usando mínimos y máximos locales."""
    rolling_min = prices.rolling(window=window, center=True).min()
    rolling_max = prices.rolling(window=window, center=True).max()
    
    # Encontrar niveles donde el precio tocó mínimos/máximos
    soportes = prices[prices == rolling_min].dropna().unique()
    resistencias = prices[prices == rolling_max].dropna().unique()
    
    # Tomar los más recientes/relevantes
    precio_actual = prices.iloc[-1]
    soportes = sorted([s for s in soportes if s < precio_actual], reverse=True)[:3]
    resistencias = sorted([r for r in resistencias if r > precio_actual])[:3]
    
    return soportes, resistencias


def calcular_volumen_tendencia(volume, window=20):
    """Analiza si el volumen está aumentando o disminuyendo."""
    if volume is None or len(volume) < window:
        return 0, "N/A"
    
    vol_ma = volume.rolling(window=window).mean()
    vol_actual = volume.iloc[-5:].mean()
    vol_anterior = vol_ma.iloc[-window]
    
    if vol_anterior > 0:
        cambio = (vol_actual - vol_anterior) / vol_anterior * 100
        tendencia = "Creciente" if cambio > 10 else "Decreciente" if cambio < -10 else "Estable"
        return cambio, tendencia
    return 0, "N/A"


# --------------------------------------------------
# FUNCIONES DE SCORING
# --------------------------------------------------
def score_fundamental(info):
    """Calcula el score fundamental (0-100)."""
    score = 0
    detalles = {}
    
    # PER (0-20 puntos)
    per = info.get('trailingPE')
    if per:
        if per < 10:
            pts = 20
        elif per < 15:
            pts = 15
        elif per < 20:
            pts = 10
        elif per < 25:
            pts = 5
        else:
            pts = 0
        score += pts
        detalles['PER'] = {'valor': f"{per:.1f}", 'puntos': pts, 'max': 20, 
                          'estado': '🟢' if pts >= 15 else '🟡' if pts >= 10 else '🔴'}
    else:
        detalles['PER'] = {'valor': 'N/A', 'puntos': 0, 'max': 20, 'estado': '⚪'}
    
    # EV/EBITDA (0-20 puntos)
    ev_ebitda = info.get('enterpriseToEbitda')
    if ev_ebitda:
        if ev_ebitda < 6:
            pts = 20
        elif ev_ebitda < 10:
            pts = 15
        elif ev_ebitda < 15:
            pts = 10
        elif ev_ebitda < 20:
            pts = 5
        else:
            pts = 0
        score += pts
        detalles['EV/EBITDA'] = {'valor': f"{ev_ebitda:.1f}", 'puntos': pts, 'max': 20,
                                 'estado': '🟢' if pts >= 15 else '🟡' if pts >= 10 else '🔴'}
    else:
        detalles['EV/EBITDA'] = {'valor': 'N/A', 'puntos': 0, 'max': 20, 'estado': '⚪'}
    
    # P/BV (0-15 puntos)
    p_bv = info.get('priceToBook')
    if p_bv:
        if p_bv < 1:
            pts = 15
        elif p_bv < 1.5:
            pts = 12
        elif p_bv < 2:
            pts = 8
        elif p_bv < 3:
            pts = 4
        else:
            pts = 0
        score += pts
        detalles['P/BV'] = {'valor': f"{p_bv:.2f}", 'puntos': pts, 'max': 15,
                           'estado': '🟢' if pts >= 12 else '🟡' if pts >= 8 else '🔴'}
    else:
        detalles['P/BV'] = {'valor': 'N/A', 'puntos': 0, 'max': 15, 'estado': '⚪'}
    
    # ROE (0-20 puntos)
    roe = info.get('returnOnEquity')
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            pts = 20
        elif roe_pct > 15:
            pts = 15
        elif roe_pct > 10:
            pts = 10
        elif roe_pct > 5:
            pts = 5
        else:
            pts = 0
        score += pts
        detalles['ROE'] = {'valor': f"{roe_pct:.1f}%", 'puntos': pts, 'max': 20,
                          'estado': '🟢' if pts >= 15 else '🟡' if pts >= 10 else '🔴'}
    else:
        detalles['ROE'] = {'valor': 'N/A', 'puntos': 0, 'max': 20, 'estado': '⚪'}
    
    # Deuda/Equity (0-15 puntos)
    debt_equity = info.get('debtToEquity')
    if debt_equity:
        if debt_equity < 50:
            pts = 15
        elif debt_equity < 100:
            pts = 12
        elif debt_equity < 150:
            pts = 8
        elif debt_equity < 200:
            pts = 4
        else:
            pts = 0
        score += pts
        detalles['Deuda/Equity'] = {'valor': f"{debt_equity:.0f}%", 'puntos': pts, 'max': 15,
                                    'estado': '🟢' if pts >= 12 else '🟡' if pts >= 8 else '🔴'}
    else:
        detalles['Deuda/Equity'] = {'valor': 'N/A', 'puntos': 0, 'max': 15, 'estado': '⚪'}
    
    # Dividend Yield (0-10 puntos)
    div_yield = info.get('dividendYield')
    if div_yield:
        div_pct = div_yield * 100
        if div_pct > 4:
            pts = 10
        elif div_pct > 3:
            pts = 8
        elif div_pct > 2:
            pts = 6
        elif div_pct > 1:
            pts = 4
        else:
            pts = 2
        score += pts
        detalles['Dividendo'] = {'valor': f"{div_pct:.2f}%", 'puntos': pts, 'max': 10,
                                 'estado': '🟢' if pts >= 8 else '🟡' if pts >= 6 else '🔴'}
    else:
        detalles['Dividendo'] = {'valor': 'N/A', 'puntos': 0, 'max': 10, 'estado': '⚪'}
    
    return score, detalles


def score_tecnico(hist):
    """Calcula el score técnico (0-100)."""
    score = 0
    detalles = {}
    
    if hist.empty or len(hist) < 50:
        return 0, {'error': 'Datos insuficientes'}
    
    close = hist['Close']
    precio_actual = close.iloc[-1]
    
    # MA50 (0-15 puntos)
    ma50 = close.rolling(window=50).mean().iloc[-1]
    if precio_actual > ma50:
        pts = 15
        estado = '🟢'
        texto = f"Precio > MA50 ({ma50:.2f})"
    else:
        pts = 0
        estado = '🔴'
        texto = f"Precio < MA50 ({ma50:.2f})"
    score += pts
    detalles['Precio vs MA50'] = {'valor': texto, 'puntos': pts, 'max': 15, 'estado': estado}
    
    # MA200 (0-15 puntos)
    if len(close) >= 200:
        ma200 = close.rolling(window=200).mean().iloc[-1]
        if precio_actual > ma200:
            pts = 15
            estado = '🟢'
            texto = f"Precio > MA200 ({ma200:.2f})"
        else:
            pts = 0
            estado = '🔴'
            texto = f"Precio < MA200 ({ma200:.2f})"
        score += pts
        detalles['Precio vs MA200'] = {'valor': texto, 'puntos': pts, 'max': 15, 'estado': estado}
    else:
        detalles['Precio vs MA200'] = {'valor': 'N/A (datos insuf.)', 'puntos': 0, 'max': 15, 'estado': '⚪'}
    
    # Golden/Death Cross (0-20 puntos)
    if len(close) >= 200:
        ma50_series = close.rolling(window=50).mean()
        ma200_series = close.rolling(window=200).mean()
        
        golden_cross = ma50_series.iloc[-1] > ma200_series.iloc[-1]
        
        # Detectar si el cruce es reciente (últimas 20 sesiones)
        cruce_reciente = False
        for i in range(-20, -1):
            if len(ma50_series) > abs(i) and len(ma200_series) > abs(i):
                if ma50_series.iloc[i-1] <= ma200_series.iloc[i-1] and ma50_series.iloc[i] > ma200_series.iloc[i]:
                    cruce_reciente = True
                    break
        
        if golden_cross:
            pts = 20 if cruce_reciente else 15
            estado = '🟢'
            texto = "Golden Cross" + (" (reciente!)" if cruce_reciente else "")
        else:
            pts = 0
            estado = '🔴'
            texto = "Death Cross"
        score += pts
        detalles['Cruce de Medias'] = {'valor': texto, 'puntos': pts, 'max': 20, 'estado': estado}
    else:
        detalles['Cruce de Medias'] = {'valor': 'N/A', 'puntos': 0, 'max': 20, 'estado': '⚪'}
    
    # RSI (0-20 puntos)
    rsi = calcular_rsi(close).iloc[-1]
    if 30 <= rsi <= 50:  # Recuperando de sobreventa
        pts = 20
        estado = '🟢'
        texto = f"{rsi:.1f} (recuperando)"
    elif 50 < rsi <= 70:  # Zona neutral-alcista
        pts = 15
        estado = '🟢'
        texto = f"{rsi:.1f} (alcista)"
    elif rsi < 30:  # Sobrevendido
        pts = 10
        estado = '🟡'
        texto = f"{rsi:.1f} (sobrevendido)"
    else:  # >70 Sobrecomprado
        pts = 0
        estado = '🔴'
        texto = f"{rsi:.1f} (sobrecomprado)"
    score += pts
    detalles['RSI'] = {'valor': texto, 'puntos': pts, 'max': 20, 'estado': estado}
    
    # MACD (0-20 puntos)
    macd, signal, histogram = calcular_macd(close)
    macd_actual = macd.iloc[-1]
    signal_actual = signal.iloc[-1]
    hist_actual = histogram.iloc[-1]
    hist_anterior = histogram.iloc[-2] if len(histogram) > 1 else 0
    
    cruce_alcista = macd_actual > signal_actual and histogram.iloc[-2] <= 0 and hist_actual > 0
    
    if cruce_alcista:
        pts = 20
        estado = '🟢'
        texto = "Cruce alcista"
    elif macd_actual > signal_actual and hist_actual > hist_anterior:
        pts = 15
        estado = '🟢'
        texto = "Tendencia alcista"
    elif macd_actual > signal_actual:
        pts = 10
        estado = '🟡'
        texto = "MACD positivo"
    else:
        pts = 0
        estado = '🔴'
        texto = "MACD negativo"
    score += pts
    detalles['MACD'] = {'valor': texto, 'puntos': pts, 'max': 20, 'estado': estado}
    
    # Volumen (0-10 puntos)
    if 'Volume' in hist.columns:
        cambio_vol, tendencia_vol = calcular_volumen_tendencia(hist['Volume'])
        if tendencia_vol == "Creciente" and precio_actual > close.iloc[-5]:
            pts = 10
            estado = '🟢'
            texto = f"Creciente (+{cambio_vol:.0f}%)"
        elif tendencia_vol == "Creciente":
            pts = 5
            estado = '🟡'
            texto = f"Creciente ({cambio_vol:+.0f}%)"
        else:
            pts = 3
            estado = '🟡'
            texto = tendencia_vol
        score += pts
        detalles['Volumen'] = {'valor': texto, 'puntos': pts, 'max': 10, 'estado': estado}
    else:
        detalles['Volumen'] = {'valor': 'N/A', 'puntos': 0, 'max': 10, 'estado': '⚪'}
    
    return score, detalles


def generar_recomendacion(score_fund, score_tech, peso_fundamental=0.5):
    """Genera recomendación final basada en scores."""
    peso_tecnico = 1 - peso_fundamental
    score_total = score_fund * peso_fundamental + score_tech * peso_tecnico
    
    if score_total >= 80:
        recomendacion = "COMPRA FUERTE"
        color = "🟢"
        explicacion = "Los indicadores fundamentales y técnicos están alineados positivamente. Buen momento para entrar."
    elif score_total >= 65:
        recomendacion = "COMPRA"
        color = "🟢"
        explicacion = "La mayoría de indicadores son favorables. Considerar entrar con precaución."
    elif score_total >= 50:
        recomendacion = "MANTENER"
        color = "🟡"
        explicacion = "Señales mixtas. Si ya tienes posición, mantén. Si no, espera mejor momento."
    elif score_total >= 35:
        recomendacion = "VENTA"
        color = "🔴"
        explicacion = "Varios indicadores negativos. Considerar reducir posición."
    else:
        recomendacion = "VENTA FUERTE"
        color = "🔴"
        explicacion = "Indicadores claramente negativos. Recomendable salir de la posición."
    
    return {
        'score_total': score_total,
        'recomendacion': recomendacion,
        'color': color,
        'explicacion': explicacion
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Parámetros")

# Buscador de tickers
with st.sidebar.expander("🔎 Buscar ticker por nombre"):
    busqueda = st.text_input("Nombre de empresa", placeholder="Ej: Inditex, Apple, BBVA...")
    
    if busqueda:
        # Primero buscar en diccionario local
        resultados_locales = [(k, v) for k, v in EMPRESAS_COMUNES.items() 
                              if busqueda.lower() in k.lower()]
        
        if resultados_locales:
            st.markdown("**Resultados:**")
            for nombre, ticker in resultados_locales[:5]:
                st.code(f"{ticker} → {nombre}")
        
        # Buscar en Yahoo Finance
        resultados_yahoo = buscar_ticker(busqueda)
        
        if resultados_yahoo:
            st.markdown("**Más resultados:**")
            for r in resultados_yahoo[:5]:
                if r['ticker'] not in [v for k, v in resultados_locales]:
                    st.code(f"{r['ticker']} → {r['nombre']} ({r['bolsa']})")
        
        if not resultados_locales and not resultados_yahoo:
            st.warning("No se encontraron resultados")
    
    st.caption("💡 España: añade .MC (ej: BBVA.MC)")
    st.caption("💡 Alemania: añade .DE (ej: BMW.DE)")
    st.caption("💡 Francia: añade .PA (ej: BNP.PA)")

st.sidebar.markdown("---")

# Modo de análisis
st.sidebar.subheader("🎯 Modo de Análisis")
modo = st.sidebar.radio(
    "¿Qué quieres analizar?",
    ["🔍 Acción individual", "🎯 Recomendación compra/venta", "📊 Cartera (2+ activos)"],
    index=0
)

st.sidebar.subheader("📈 Activos")

tickers_populares = {
    "Tech US": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Europa": ["BNP.PA", "SAP.DE", "ASML.AS", "NVO"],
    "España": ["BBVA.MC", "SAN.MC", "ITX.MC", "IBE.MC", "TEF.MC"],
    "ETFs": ["SPY", "QQQ", "VTI", "IWM"],
    "Bancos": ["BBVA.MC", "SAN.MC", "BNP.PA", "JPM", "BAC"],
}

usar_predefinidos = st.sidebar.checkbox("Usar tickers predefinidos", value=False)

if modo == "🔍 Acción individual" or modo == "🎯 Recomendación compra/venta":
    if usar_predefinidos:
        todas_acciones = []
        for cat, ticks in tickers_populares.items():
            todas_acciones.extend(ticks)
        todas_acciones = sorted(list(set(todas_acciones)))
        TICKER_INDIVIDUAL = st.sidebar.selectbox("Selecciona acción", todas_acciones)
    else:
        TICKER_INDIVIDUAL = st.sidebar.text_input(
            "Introduce ticker",
            value="BBVA.MC",
            help="Ejemplo: AAPL, MSFT, BBVA.MC"
        ).strip().upper()
    TICKERS = [TICKER_INDIVIDUAL] if TICKER_INDIVIDUAL else []
else:
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

periodo = st.sidebar.selectbox(
    "Período histórico", 
    ["5d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y"],
    index=4,
    format_func=lambda x: {
        "5d": "Última semana",
        "1mo": "1 mes",
        "3mo": "3 meses",
        "6mo": "6 meses",
        "1y": "1 año",
        "2y": "2 años",
        "3y": "3 años",
        "5y": "5 años",
        "10y": "10 años"
    }.get(x, x)
)

periodo_texto = {
    "5d": "última semana",
    "1mo": "último mes",
    "3mo": "últimos 3 meses",
    "6mo": "últimos 6 meses",
    "1y": "último año",
    "2y": "últimos 2 años",
    "3y": "últimos 3 años",
    "5y": "últimos 5 años",
    "10y": "últimos 10 años"
}.get(periodo, periodo)

st.sidebar.markdown("---")

# Parámetros para modo recomendación
if modo == "🎯 Recomendación compra/venta":
    st.sidebar.subheader("⚖️ Ponderación")
    peso_fundamental = st.sidebar.slider(
        "Peso Análisis Fundamental",
        0, 100, 50, 5,
        help="Porcentaje de peso para el análisis fundamental vs técnico"
    ) / 100

if modo == "📊 Cartera (2+ activos)":
    st.sidebar.subheader("💰 Inversión")
    investment = st.sidebar.number_input("Inversión total (€)", min_value=100, max_value=1_000_000, value=10_000, step=500)

    st.sidebar.subheader("📅 Simulación")
    months = st.sidebar.slider("Horizonte (meses)", 1, 24, 6)
    n_sim = st.sidebar.select_slider("Simulaciones", options=[1000, 5000, 10000, 25000], value=10000)

    st.sidebar.subheader("📊 Optimización")
    rf = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 10.0, 3.0, 0.25) / 100
    
    # Modo de optimización
    modo_optimizacion = st.sidebar.radio(
        "Estrategia de cartera",
        ["🎯 Máximo Sharpe (sin límites)", "🔀 Diversificación forzada"],
        index=1,
        help="Máximo Sharpe puede concentrar todo en una acción. Diversificación fuerza un reparto."
    )
    
    if modo_optimizacion == "🔀 Diversificación forzada":
        max_weight = st.sidebar.slider(
            "Peso máximo por activo (%)", 
            20, 60, 40, 5,
            help="Limita cuánto puede invertirse en un solo activo"
        ) / 100
    else:
        max_weight = 1.0  # Sin límite

# --------------------------------------------------
# CONTENIDO PRINCIPAL
# --------------------------------------------------
st.title("📊 Análisis de Inversiones")

# ==================================================
# MODO ACCIÓN INDIVIDUAL
# ==================================================
if modo == "🔍 Acción individual":
    if not TICKERS or not TICKERS[0]:
        st.error("Introduce un ticker para analizar.")
        st.stop()
    
    ticker = TICKERS[0]
    
    with st.spinner(f"Cargando datos de {ticker}..."):
        data_accion = obtener_info_accion(ticker, periodo)
    
    if data_accion is None:
        st.error(f"No se pudieron obtener datos para {ticker}. Verifica que el ticker sea correcto.")
        st.stop()
    
    info = data_accion['info']
    hist = data_accion['history']
    
    # Información general
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {info.get('longName', ticker)}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industria:** {info.get('industry', 'N/A')}")
        st.markdown(f"**País:** {info.get('country', 'N/A')} | **Moneda:** {info.get('currency', 'N/A')} | **Bolsa:** {info.get('exchange', 'N/A')}")
        
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
        
        market_cap = info.get('marketCap', 0)
        st.metric("Capitalización", formatear_numero(market_cap))
    
    st.markdown("---")
    
    # Gráfico de cotización
    st.markdown(f"### 📈 Cotización ({periodo_texto})")
    
    if not hist.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(hist.index, hist['Close'], 'b-', linewidth=1.5, label='Precio de Cierre')
        
        if len(hist) >= 50:
            ma50 = hist['Close'].rolling(window=50).mean()
            ax.plot(hist.index, ma50, 'orange', linewidth=1, label='MA 50', alpha=0.8)
        
        if len(hist) >= 200:
            ma200 = hist['Close'].rolling(window=200).mean()
            ax.plot(hist.index, ma200, 'red', linewidth=1, label='MA 200', alpha=0.8)
        
        ax.fill_between(hist.index, hist['Low'], hist['High'], alpha=0.1, color='blue')
        
        ax.set_xlabel('Fecha')
        ax.set_ylabel(f'Precio ({info.get("currency", "USD")})')
        ax.set_title(f'{ticker} - {info.get("longName", "")}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo 52 sem", f"{info.get('fiftyTwoWeekLow', 'N/A')}")
        col2.metric("Máximo 52 sem", f"{info.get('fiftyTwoWeekHigh', 'N/A')}")
        col3.metric("Media 50 días", f"{info.get('fiftyDayAverage', 0):.2f}")
        col4.metric("Media 200 días", f"{info.get('twoHundredDayAverage', 0):.2f}")
    
    st.markdown("---")
    
    # Ratios de valoración
    st.markdown("### 📊 Ratios de Valoración")
    
    col1, col2, col3, col4 = st.columns(4)
    
    per = info.get('trailingPE')
    per_forward = info.get('forwardPE')
    with col1:
        st.markdown("**PER (Precio/Beneficio)**")
        st.markdown(f"### {per:.2f}" if per else "### N/A")
        if per_forward:
            st.caption(f"Forward PER: {per_forward:.2f}")
        st.caption("Años de beneficios que pagas")
    
    ev_ebitda = info.get('enterpriseToEbitda')
    with col2:
        st.markdown("**EV/EBITDA**")
        st.markdown(f"### {ev_ebitda:.2f}" if ev_ebitda else "### N/A")
        st.caption("Valor empresa vs EBITDA")
    
    p_fcf = data_accion['p_fcf']
    with col3:
        st.markdown("**P/FCF**")
        st.markdown(f"### {p_fcf:.2f}" if p_fcf else "### N/A")
        st.caption("Precio vs flujo de caja")
    
    p_bv = info.get('priceToBook')
    with col4:
        st.markdown("**P/BV**")
        st.markdown(f"### {p_bv:.2f}" if p_bv else "### N/A")
        st.caption("Precio vs valor contable")
    
    with st.expander("📖 ¿Cómo interpretar los ratios?"):
        st.markdown("""
        | Ratio | Bajo | Medio | Alto | Interpretación |
        |-------|------|-------|------|----------------|
        | **PER** | <10 | 10-20 | >25 | PER bajo puede indicar infravaloración; alto indica expectativas de crecimiento |
        | **EV/EBITDA** | <6 | 6-12 | >15 | Útil para comparar empresas del mismo sector |
        | **P/FCF** | <10 | 10-20 | >25 | Similar al PER pero basado en caja real |
        | **P/BV** | <1 | 1-3 | >3 | P/BV < 1 puede indicar infravaloración (común en bancos) |
        
        ⚠️ **Importante:** Compara siempre con empresas del mismo sector.
        """)
    
    st.markdown("---")
    
    # Métricas financieras
    st.markdown("### 💰 Métricas Financieras")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Ingresos (TTM)**")
        revenue = info.get('totalRevenue', 0)
        st.markdown(f"### {formatear_numero(revenue)}")
        
        st.markdown("**Beneficio Neto**")
        net_income = info.get('netIncomeToCommon', 0)
        st.markdown(f"### {formatear_numero(net_income)}")
    
    with col2:
        st.markdown("**EBITDA**")
        ebitda = info.get('ebitda', 0)
        st.markdown(f"### {formatear_numero(ebitda)}")
        
        st.markdown("**Flujo de Caja Libre**")
        fcf = info.get('freeCashflow', 0)
        st.markdown(f"### {formatear_numero(fcf)}")
    
    with col3:
        st.markdown("**Margen Operativo**")
        op_margin = info.get('operatingMargins', 0)
        st.markdown(f"### {op_margin*100:.1f}%" if op_margin else "### N/A")
        
        st.markdown("**ROE**")
        roe = info.get('returnOnEquity', 0)
        st.markdown(f"### {roe*100:.1f}%" if roe else "### N/A")
    
    st.markdown("---")
    
    # Dividendos
    st.markdown("### 💵 Dividendos")
    
    col1, col2, col3 = st.columns(3)
    
    div_yield = info.get('dividendYield', 0)
    div_rate = info.get('dividendRate', 0)
    payout = info.get('payoutRatio', 0)
    
    col1.metric("Rentabilidad por dividendo", f"{div_yield*100:.2f}%" if div_yield else "N/A")
    col2.metric("Dividendo anual", f"{div_rate:.2f} {info.get('currency', '')}" if div_rate else "N/A")
    col3.metric("Payout Ratio", f"{payout*100:.1f}%" if payout else "N/A")
    
    st.markdown("---")
    
    # Deuda
    st.markdown("### 🏦 Deuda y Solvencia")
    
    col1, col2, col3 = st.columns(3)
    
    total_debt = info.get('totalDebt', 0)
    total_cash = info.get('totalCash', 0)
    debt_equity = info.get('debtToEquity', 0)
    
    col1.metric("Deuda Total", formatear_numero(total_debt))
    col2.metric("Caja Total", formatear_numero(total_cash))
    col3.metric("Deuda/Patrimonio", f"{debt_equity:.1f}%" if debt_equity else "N/A")

# ==================================================
# MODO RECOMENDACIÓN COMPRA/VENTA
# ==================================================
elif modo == "🎯 Recomendación compra/venta":
    if not TICKERS or not TICKERS[0]:
        st.error("Introduce un ticker para analizar.")
        st.stop()
    
    ticker = TICKERS[0]
    
    with st.spinner(f"Analizando {ticker}..."):
        # Obtener datos fundamentales
        data_accion = obtener_info_accion(ticker, "1y")
        
        # Obtener datos históricos más largos para análisis técnico
        stock = yf.Ticker(ticker)
        hist_largo = stock.history(period="1y")
    
    if data_accion is None:
        st.error(f"No se pudieron obtener datos para {ticker}.")
        st.stop()
    
    info = data_accion['info']
    hist = data_accion['history']
    
    # Calcular scores
    s_fund, detalles_fund = score_fundamental(info)
    s_tech, detalles_tech = score_tecnico(hist_largo if not hist_largo.empty else hist)
    
    # Generar recomendación
    rec = generar_recomendacion(s_fund, s_tech, peso_fundamental)
    
    # --- HEADER ---
    st.markdown(f"## {info.get('longName', ticker)}")
    st.markdown(f"**{ticker}** | {info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}")
    
    # --- RECOMENDACIÓN PRINCIPAL ---
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; border: 2px solid {'#00ff88' if rec['color'] == '🟢' else '#ffaa00' if rec['color'] == '🟡' else '#ff4444'};">
            <h1 style="font-size: 3em; margin: 0;">{rec['color']}</h1>
            <h2 style="color: {'#00ff88' if rec['color'] == '🟢' else '#ffaa00' if rec['color'] == '🟡' else '#ff4444'}; margin: 10px 0;">{rec['recomendacion']}</h2>
            <p style="font-size: 2.5em; font-weight: bold; margin: 0;">{rec['score_total']:.0f}/100</p>
            <p style="color: #aaa; margin-top: 10px;">{rec['explicacion']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- SCORES DESGLOSADOS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📊 Score Fundamental: {s_fund}/100")
        st.progress(s_fund / 100)
        
        for indicador, datos in detalles_fund.items():
            if indicador != 'error':
                st.markdown(f"{datos['estado']} **{indicador}**: {datos['valor']} ({datos['puntos']}/{datos['max']} pts)")
    
    with col2:
        st.markdown(f"### 📈 Score Técnico: {s_tech}/100")
        st.progress(s_tech / 100)
        
        for indicador, datos in detalles_tech.items():
            if indicador != 'error':
                st.markdown(f"{datos['estado']} **{indicador}**: {datos['valor']} ({datos['puntos']}/{datos['max']} pts)")
    
    st.markdown("---")
    
    # --- GRÁFICO TÉCNICO ---
    st.markdown("### 📉 Análisis Técnico Visual")
    
    if not hist_largo.empty:
        close = hist_largo['Close']
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # --- GRÁFICO DE PRECIO ---
        ax1 = axes[0]
        ax1.plot(close.index, close, 'b-', linewidth=1.5, label='Precio')
        
        # Medias móviles
        if len(close) >= 50:
            ma50 = close.rolling(window=50).mean()
            ax1.plot(close.index, ma50, 'orange', linewidth=1, label='MA50')
        
        if len(close) >= 200:
            ma200 = close.rolling(window=200).mean()
            ax1.plot(close.index, ma200, 'red', linewidth=1, label='MA200')
        
        # Soportes y resistencias
        soportes, resistencias = calcular_soportes_resistencias(close)
        
        for soporte in soportes[:2]:
            ax1.axhline(y=soporte, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(close.index[-1], soporte, f' S: {soporte:.2f}', va='center', fontsize=8, color='green')
        
        for resistencia in resistencias[:2]:
            ax1.axhline(y=resistencia, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(close.index[-1], resistencia, f' R: {resistencia:.2f}', va='center', fontsize=8, color='red')
        
        ax1.fill_between(close.index, hist_largo['Low'], hist_largo['High'], alpha=0.1, color='blue')
        ax1.set_title(f'{ticker} - Precio y Medias Móviles')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Precio')
        
        # --- RSI ---
        ax2 = axes[1]
        rsi = calcular_rsi(close)
        ax2.plot(rsi.index, rsi, 'purple', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax2.fill_between(rsi.index, rsi, 70, where=(rsi >= 70), alpha=0.3, color='red')
        ax2.fill_between(rsi.index, rsi, 30, where=(rsi <= 30), alpha=0.3, color='green')
        ax2.set_ylim(0, 100)
        ax2.set_title('RSI (14)')
        ax2.set_ylabel('RSI')
        ax2.grid(True, alpha=0.3)
        
        # --- MACD ---
        ax3 = axes[2]
        macd, signal, histogram = calcular_macd(close)
        
        colors = ['green' if h >= 0 else 'red' for h in histogram]
        ax3.bar(histogram.index, histogram, color=colors, alpha=0.5, width=1)
        ax3.plot(macd.index, macd, 'blue', linewidth=1, label='MACD')
        ax3.plot(signal.index, signal, 'orange', linewidth=1, label='Señal')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.set_title('MACD')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # --- INFORMACIÓN ADICIONAL ---
    st.markdown("---")
    st.markdown("### ℹ️ Información Clave")
    
    col1, col2, col3, col4 = st.columns(4)
    
    precio_actual = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    col1.metric("Precio Actual", f"{precio_actual:.2f} {info.get('currency', '')}")
    col2.metric("Mín 52 sem", f"{info.get('fiftyTwoWeekLow', 'N/A')}")
    col3.metric("Máx 52 sem", f"{info.get('fiftyTwoWeekHigh', 'N/A')}")
    col4.metric("Capitalización", formatear_numero(info.get('marketCap', 0)))
    
    # --- LEYENDA ---
    with st.expander("📖 Cómo interpretar la recomendación"):
        st.markdown("""
        ### Sistema de Scoring
        
        **Score Fundamental (máx 100 pts)**
        - PER < 15: empresa "barata" respecto a beneficios
        - EV/EBITDA < 10: buena valoración considerando deuda
        - P/BV < 1.5: cotiza cerca de su valor contable
        - ROE > 15%: empresa eficiente generando beneficios
        - Deuda/Equity < 100%: endeudamiento controlado
        - Dividendo > 2%: retribución atractiva
        
        **Score Técnico (máx 100 pts)**
        - Precio > MA50 y MA200: tendencia alcista
        - Golden Cross: MA50 cruza por encima de MA200 (señal alcista)
        - RSI 30-50: recuperándose de zona de sobreventa
        - MACD positivo: impulso alcista
        - Volumen creciente: confirma movimientos
        
        **Recomendación Final**
        | Score | Recomendación |
        |-------|---------------|
        | 80-100 | 🟢 COMPRA FUERTE |
        | 65-79 | 🟢 COMPRA |
        | 50-64 | 🟡 MANTENER |
        | 35-49 | 🔴 VENTA |
        | 0-34 | 🔴 VENTA FUERTE |
        
        ⚠️ **Importante**: Esta herramienta es orientativa. No es asesoramiento financiero.
        """)

# ==================================================
# MODO CARTERA
# ==================================================
elif modo == "📊 Cartera (2+ activos)":
    if len(TICKERS) < 2:
        st.error("Necesitas al menos 2 tickers para optimizar una cartera. Usa el modo 'Acción individual' para analizar una sola acción.")
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

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Cartera Óptima", 
        "🎲 Simulación Monte Carlo", 
        "⚖️ Rebalanceo", 
        "📉 Frontera Eficiente"
    ])

    # TAB 1: CARTERA ÓPTIMA
    with tab1:
        st.subheader("Optimización de Cartera")
        
        # Mostrar modo de optimización
        if max_weight < 1.0:
            st.info(f"🔀 **Diversificación forzada**: máximo {max_weight:.0%} por activo")
        else:
            st.info("🎯 **Máximo Sharpe**: sin límites de concentración")
        
        best = optimal_portfolio(prices, rf, max_weight)
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

    # TAB 2: SIMULACIÓN MONTE CARLO
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

    # TAB 3: REBALANCEO
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

    # TAB 4: FRONTERA EFICIENTE
    with tab4:
        st.subheader("Frontera Eficiente")
        
        with st.spinner("Calculando frontera eficiente..."):
            frontier = efficient_frontier(prices, rf, n_points=100, max_weight=max_weight)
        
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
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<small>
<b>Disclaimer:</b> Esta herramienta es únicamente para fines educativos. 
Los resultados pasados no garantizan rendimientos futuros.
</small>
""", unsafe_allow_html=True)
