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
import warnings
warnings.filterwarnings('ignore')

# HMM y GARCH (opcionales)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False


# --------------------------------------------------
# DATOS DE BONOS Y ETFs
# --------------------------------------------------

# ETFs organizados por categoría
ETFs_POR_CATEGORIA = {
    "📈 Renta Variable USA": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "VTI": "Total Stock Market",
        "IWM": "Russell 2000 (Small Caps)",
        "DIA": "Dow Jones 30"
    },
    "🇪🇺 Renta Variable Europa": {
        "VGK": "Europa Total",
        "FEZ": "Euro Stoxx 50",
        "EWP": "España (IBEX)",
        "EWG": "Alemania (DAX)",
        "EWQ": "Francia (CAC)"
    },
    "🌏 Renta Variable Global": {
        "VT": "Total World",
        "EEM": "Emergentes",
        "EFA": "Desarrollados ex-USA",
        "VWO": "Emergentes (Vanguard)",
        "IEMG": "Emergentes Core"
    },
    "📊 Renta Variable Sectorial": {
        "XLK": "Tecnología",
        "XLF": "Financiero",
        "XLV": "Salud",
        "XLE": "Energía",
        "XLI": "Industrial"
    },
    "🏦 Renta Fija USA": {
        "BND": "Bonos Total USA",
        "AGG": "Aggregate Bond",
        "TLT": "Bonos Largo Plazo 20+",
        "IEF": "Bonos Medio Plazo 7-10",
        "SHY": "Bonos Corto Plazo 1-3"
    },
    "🏛️ Renta Fija Corporativa": {
        "LQD": "Investment Grade",
        "HYG": "High Yield",
        "VCIT": "Corporate Intermediate",
        "VCSH": "Corporate Short-Term",
        "JNK": "High Yield SPDR"
    },
    "🌍 Renta Fija Internacional": {
        "BNDX": "Bonos Internacionales",
        "EMB": "Bonos Emergentes",
        "BWX": "Tesoro Internacional",
        "IGOV": "Gobiernos Desarrollados",
        "VWOB": "Bonos Emergentes Gov"
    },
    "🥇 Materias Primas": {
        "GLD": "Oro",
        "SLV": "Plata",
        "USO": "Petróleo",
        "DBA": "Agricultura",
        "DBC": "Commodities Diversificado"
    },
    "🏠 Inmobiliario (REITs)": {
        "VNQ": "REITs USA",
        "VNQI": "REITs Internacional",
        "IYR": "Real Estate USA",
        "XLRE": "Real Estate Select",
        "REM": "Mortgage REITs"
    }
}

# Bonos de referencia (datos actualizados manualmente o via scraping)
# En producción se actualizarían con FRED API o scraping del Tesoro
BONOS_REFERENCIA = {
    "🇪🇸 España": {
        "Letra 3 meses": {"ticker": None, "fuente": "Tesoro"},
        "Letra 6 meses": {"ticker": None, "fuente": "Tesoro"},
        "Letra 12 meses": {"ticker": None, "fuente": "Tesoro"},
        "Bono 3 años": {"ticker": None, "fuente": "Tesoro"},
        "Bono 5 años": {"ticker": None, "fuente": "Tesoro"},
        "Bono 10 años": {"ticker": None, "fuente": "Tesoro"},
    },
    "🇺🇸 USA": {
        "Treasury 2 años": {"ticker": "^IRX", "fuente": "Yahoo"},
        "Treasury 5 años": {"ticker": "^FVX", "fuente": "Yahoo"},
        "Treasury 10 años": {"ticker": "^TNX", "fuente": "Yahoo"},
        "Treasury 30 años": {"ticker": "^TYX", "fuente": "Yahoo"},
    },
    "🇩🇪 Alemania": {
        "Bund 10 años": {"ticker": None, "fuente": "ECB"},
    }
}

# Perfiles de riesgo predefinidos
PERFILES_RIESGO = {
    "Conservador": {
        "descripcion": "Prioriza la preservación del capital. Baja tolerancia al riesgo.",
        "asignacion": {"Renta Variable": 20, "Renta Fija": 60, "Oro/Commodities": 10, "Liquidez": 10},
        "etfs_sugeridos": ["SPY", "BND", "AGG", "GLD", "SHY"]
    },
    "Moderado": {
        "descripcion": "Equilibrio entre crecimiento y seguridad. Tolerancia media al riesgo.",
        "asignacion": {"Renta Variable": 50, "Renta Fija": 35, "Oro/Commodities": 10, "Liquidez": 5},
        "etfs_sugeridos": ["VTI", "VGK", "BND", "GLD", "LQD"]
    },
    "Agresivo": {
        "descripcion": "Busca máximo crecimiento. Alta tolerancia al riesgo.",
        "asignacion": {"Renta Variable": 75, "Renta Fija": 15, "Oro/Commodities": 5, "Liquidez": 5},
        "etfs_sugeridos": ["QQQ", "VTI", "EEM", "TLT", "GLD"]
    },
    "Muy Agresivo": {
        "descripcion": "100% enfocado en crecimiento. Muy alta tolerancia al riesgo.",
        "asignacion": {"Renta Variable": 90, "Renta Fija": 5, "Oro/Commodities": 5, "Liquidez": 0},
        "etfs_sugeridos": ["QQQ", "IWM", "EEM", "XLK", "ARKK"]
    }
}


@st.cache_data(ttl=3600)
def obtener_rentabilidad_etf(ticker, periodo="1y"):
    """Obtiene la rentabilidad de un ETF."""
    try:
        etf = yf.Ticker(ticker)
        hist = etf.history(period=periodo)
        
        if hist.empty:
            return None
        
        precio_inicio = hist['Close'].iloc[0]
        precio_fin = hist['Close'].iloc[-1]
        rentabilidad = (precio_fin - precio_inicio) / precio_inicio * 100
        
        # Volatilidad anualizada
        returns = hist['Close'].pct_change().dropna()
        volatilidad = returns.std() * np.sqrt(252) * 100
        
        # Info adicional
        info = etf.info
        
        return {
            'ticker': ticker,
            'nombre': info.get('shortName', ticker),
            'precio': precio_fin,
            'rentabilidad': rentabilidad,
            'volatilidad': volatilidad,
            'volumen': info.get('averageVolume', 0),
            'aum': info.get('totalAssets', 0),
            'expense_ratio': info.get('annualReportExpenseRatio', 0),
            'categoria': info.get('category', 'N/A')
        }
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def obtener_tipos_bonos_usa():
    """Obtiene los tipos de interés de bonos USA desde Yahoo Finance."""
    bonos = {
        "Treasury 3 meses": "^IRX",
        "Treasury 2 años": "^IRX",  # Aproximación
        "Treasury 5 años": "^FVX",
        "Treasury 10 años": "^TNX",
        "Treasury 30 años": "^TYX"
    }
    
    resultados = {}
    for nombre, ticker in bonos.items():
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="5d")
            if not hist.empty:
                tipo_actual = hist['Close'].iloc[-1]
                tipo_anterior = hist['Close'].iloc[0] if len(hist) > 1 else tipo_actual
                cambio = tipo_actual - tipo_anterior
                resultados[nombre] = {
                    'tipo': tipo_actual,
                    'cambio': cambio,
                    'ticker': ticker
                }
        except:
            pass
    
    return resultados


@st.cache_data(ttl=86400)
def obtener_tipos_bonos_espana():
    """
    Obtiene tipos de interés de bonos españoles.
    Nota: En producción, hacer scraping del Tesoro Público.
    Por ahora usamos datos aproximados/estáticos.
    """
    # Datos aproximados (actualizar periódicamente o implementar scraping)
    # Fuente real: https://www.tesoro.es
    return {
        "Letra 3 meses": {"tipo": 3.15, "cambio": -0.05},
        "Letra 6 meses": {"tipo": 3.05, "cambio": -0.08},
        "Letra 12 meses": {"tipo": 2.85, "cambio": -0.10},
        "Bono 3 años": {"tipo": 2.65, "cambio": 0.02},
        "Bono 5 años": {"tipo": 2.80, "cambio": 0.05},
        "Bono 10 años": {"tipo": 3.15, "cambio": 0.03},
        "Obligación 15 años": {"tipo": 3.45, "cambio": 0.02},
        "Obligación 30 años": {"tipo": 3.75, "cambio": 0.01},
    }


def calcular_cartera_por_perfil(perfil, inversion):
    """Calcula la distribución de una cartera según el perfil de riesgo."""
    if perfil not in PERFILES_RIESGO:
        return None
    
    config = PERFILES_RIESGO[perfil]
    asignacion = config['asignacion']
    
    distribucion = {}
    for categoria, porcentaje in asignacion.items():
        distribucion[categoria] = {
            'porcentaje': porcentaje,
            'importe': inversion * porcentaje / 100
        }
    
    return {
        'distribucion': distribucion,
        'etfs_sugeridos': config['etfs_sugeridos'],
        'descripcion': config['descripcion']
    }

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
@st.cache_data(ttl=7200)  # Cache de 2 horas
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
        if "RateLimit" in str(type(e).__name__) or "rate" in str(e).lower():
            st.error("⚠️ Yahoo Finance ha bloqueado temporalmente las peticiones. Espera 1-2 minutos.")
        else:
            st.error(f"Error descargando datos: {e}")
        return None


@st.cache_data(ttl=7200)  # Cache de 2 horas
def obtener_info_accion(ticker, periodo="1y"):
    """Obtiene información fundamental de una acción."""
    try:
        stock = yf.Ticker(ticker)
        
        # Obtener histórico primero (menos propenso a rate limit)
        hist = stock.history(period=periodo)
        
        # Intentar obtener info fundamental
        try:
            info = stock.info
        except Exception:
            info = {}
        
        # Si faltan datos fundamentales, intentar calcularlos de otras fuentes
        if not info.get('trailingPE') or not info.get('returnOnEquity'):
            try:
                # Intentar obtener de financials
                financials = stock.quarterly_financials
                balance = stock.quarterly_balance_sheet
                
                if not financials.empty and not balance.empty:
                    # Calcular PER si no existe
                    if not info.get('trailingPE') and info.get('currentPrice'):
                        try:
                            net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else None
                            shares = info.get('sharesOutstanding', 0)
                            if net_income and shares and net_income > 0:
                                eps = (net_income * 4) / shares  # Anualizado
                                info['trailingPE'] = info['currentPrice'] / eps
                        except:
                            pass
                    
                    # Calcular ROE si no existe
                    if not info.get('returnOnEquity'):
                        try:
                            net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else None
                            equity = balance.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance.index else None
                            if net_income and equity and equity > 0:
                                info['returnOnEquity'] = (net_income * 4) / equity  # Anualizado
                        except:
                            pass
            except:
                pass
        
        # Si no hay precio actual, usar el último del histórico
        if not info.get('currentPrice') and not info.get('regularMarketPrice'):
            if not hist.empty:
                info['currentPrice'] = hist['Close'].iloc[-1]
        
        # Si no hay nombre, usar ticker
        if not info.get('longName') and not info.get('shortName'):
            info['longName'] = ticker
        
        market_cap = info.get('marketCap', 0)
        free_cash_flow = info.get('freeCashflow', 0)
        p_fcf = market_cap / free_cash_flow if free_cash_flow and free_cash_flow > 0 else None
        
        return {
            'info': info,
            'history': hist,
            'p_fcf': p_fcf
        }
    except Exception as e:
        st.warning(f"Error parcial obteniendo datos de {ticker}: {e}")
        return None


def obtener_accion_con_fallback(tickers_fallback, periodo="1y"):
    """Intenta obtener datos de una lista de tickers, usando el primero que funcione."""
    for ticker in tickers_fallback:
        try:
            data = obtener_info_accion(ticker, periodo)
            if data and not data['history'].empty:
                return data, ticker
        except:
            continue
    return None, None


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
    try:
        log_returns, mu, cov = compute_statistics(prices)
        n_assets = len(prices.columns)
        
        # Ajustar max_weight si es muy restrictivo
        min_weight_needed = 1.0 / n_assets
        if max_weight < min_weight_needed:
            max_weight = min_weight_needed + 0.1
        
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
        
        if not frontier:
            # Si no hay resultados, crear al menos un punto con pesos iguales
            w_equal = np.ones(n_assets) / n_assets
            ret_equal = np.dot(w_equal, mu)
            vol_equal = np.sqrt(np.dot(w_equal.T, np.dot(cov, w_equal)))
            frontier.append({'Return': ret_equal, 'Vol': vol_equal, 'Sharpe': (ret_equal - rf) / vol_equal, 'Weights': w_equal})
        
        return pd.DataFrame(frontier)
    except Exception as e:
        return pd.DataFrame(columns=['Return', 'Vol', 'Sharpe', 'Weights'])


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


def calcular_bollinger_bands(prices, period=30, std_dev=2):
    """Calcula las Bandas de Bollinger."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def detectar_vela_rechazo(hist_df):
    """
    Detecta velas de rechazo (martillo alcista o estrella fugaz bajista).
    Retorna: 'alcista', 'bajista', o None
    """
    if len(hist_df) < 2:
        return None, {}
    
    # Última vela
    ultimo = hist_df.iloc[-1]
    penultimo = hist_df.iloc[-2]
    
    open_price = ultimo['Open']
    close_price = ultimo['Close']
    high_price = ultimo['High']
    low_price = ultimo['Low']
    
    # Cálculo del cuerpo y mechas
    body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # Evitar división por cero
    if body < 0.0001:
        body = 0.0001
    
    detalles = {
        'body': body,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'ratio_lower': lower_wick / body if body > 0 else 0,
        'ratio_upper': upper_wick / body if body > 0 else 0,
        'cierre_vs_anterior': close_price - penultimo['Close']
    }
    
    # Rechazo alcista (martillo): mecha inferior larga + cierre > apertura
    if lower_wick >= body * 1.5 and close_price > open_price:
        return 'alcista', detalles
    
    # Rechazo bajista (estrella fugaz): mecha superior larga + cierre < apertura
    if upper_wick >= body * 1.5 and close_price < open_price:
        return 'bajista', detalles
    
    return None, detalles


def analizar_retorno_media(hist_df, rsi_period=13, bb_period=30, bb_std=2):
    """
    Implementa la estrategia de Retorno a la Media con RSI y Bandas de Bollinger.
    Retorna señal y detalles del análisis.
    """
    if len(hist_df) < max(rsi_period, bb_period, 200):
        return None, {}
    
    close = hist_df['Close']
    
    # Calcular indicadores
    rsi = calcular_rsi(close, period=rsi_period)
    upper_band, middle_band, lower_band = calcular_bollinger_bands(close, period=bb_period, std_dev=bb_std)
    sma200 = close.rolling(window=200).mean()
    
    # Valores actuales
    precio_actual = close.iloc[-1]
    rsi_actual = rsi.iloc[-1]
    bb_upper = upper_band.iloc[-1]
    bb_middle = middle_band.iloc[-1]
    bb_lower = lower_band.iloc[-1]
    sma200_actual = sma200.iloc[-1]
    
    # Detectar vela de rechazo
    tipo_rechazo, detalles_rechazo = detectar_vela_rechazo(hist_df)
    
    # Determinar tendencia principal (precio vs SMA200)
    if precio_actual > sma200_actual * 1.02:
        tendencia = 'alcista'
    elif precio_actual < sma200_actual * 0.98:
        tendencia = 'bajista'
    else:
        tendencia = 'neutral'
    
    # Posición respecto a Bollinger
    bb_position = (precio_actual - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    
    detalles = {
        'precio': precio_actual,
        'rsi': rsi_actual,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'bb_position': bb_position,  # 0 = banda inferior, 100 = banda superior
        'sma200': sma200_actual,
        'tendencia': tendencia,
        'tipo_rechazo': tipo_rechazo,
        'detalles_rechazo': detalles_rechazo
    }
    
    # Evaluar señales
    señal = None
    razon = []
    
    # SEÑAL LONG: Precio toca/cruza banda inferior + RSI ≤ 30 + rechazo alcista + tendencia no bajista
    if precio_actual <= bb_lower:
        razon.append("Precio en/bajo banda inferior")
        if rsi_actual <= 30:
            razon.append(f"RSI en sobreventa ({rsi_actual:.1f})")
            if tipo_rechazo == 'alcista':
                razon.append("Vela de rechazo alcista")
                if tendencia != 'bajista':
                    señal = 'LONG'
                    razon.append(f"Tendencia {tendencia}")
    
    # SEÑAL SHORT: Precio toca/cruza banda superior + RSI ≥ 70 + rechazo bajista + tendencia no alcista
    elif precio_actual >= bb_upper:
        razon.append("Precio en/sobre banda superior")
        if rsi_actual >= 70:
            razon.append(f"RSI en sobrecompra ({rsi_actual:.1f})")
            if tipo_rechazo == 'bajista':
                razon.append("Vela de rechazo bajista")
                if tendencia != 'alcista':
                    señal = 'SHORT'
                    razon.append(f"Tendencia {tendencia}")
    
    # Señales parciales (condiciones incompletas)
    if señal is None:
        if rsi_actual <= 30 and precio_actual <= bb_lower * 1.02:
            señal = 'VIGILAR_LONG'
            razon.append("Cerca de señal LONG - falta confirmación")
        elif rsi_actual >= 70 and precio_actual >= bb_upper * 0.98:
            señal = 'VIGILAR_SHORT'
            razon.append("Cerca de señal SHORT - falta confirmación")
        elif 30 < rsi_actual < 70 and bb_lower < precio_actual < bb_upper:
            señal = 'NEUTRAL'
            razon.append("Sin señal - precio dentro de bandas")
    
    detalles['señal'] = señal
    detalles['razon'] = razon
    
    # Calcular objetivos y stops
    if señal == 'LONG':
        detalles['objetivo'] = bb_middle
        detalles['stop'] = hist_df['Low'].iloc[-1] * 0.98  # 2% bajo mínimo
        detalles['riesgo_beneficio'] = (bb_middle - precio_actual) / (precio_actual - detalles['stop']) if (precio_actual - detalles['stop']) > 0 else 0
    elif señal == 'SHORT':
        detalles['objetivo'] = bb_middle
        detalles['stop'] = hist_df['High'].iloc[-1] * 1.02  # 2% sobre máximo
        detalles['riesgo_beneficio'] = (precio_actual - bb_middle) / (detalles['stop'] - precio_actual) if (detalles['stop'] - precio_actual) > 0 else 0
    
    return señal, detalles


def score_tecnico_mejorado(hist_df):
    """
    Score técnico mejorado con Bandas de Bollinger y estrategia de Retorno a la Media.
    Escala: 0-100 (0 = muy bajista, 50 = neutral, 100 = muy alcista)
    """
    score = 0
    detalles = {}
    
    if len(hist_df) < 200:
        return 50, {'error': 'Datos insuficientes'}
    
    close = hist_df['Close']
    volume = hist_df['Volume'] if 'Volume' in hist_df.columns else None
    
    # 1. Precio vs MA50 y MA200 (0-20 puntos)
    ma50 = close.rolling(window=50).mean().iloc[-1]
    ma200 = close.rolling(window=200).mean().iloc[-1]
    precio_actual = close.iloc[-1]
    
    pts_ma = 0
    if precio_actual > ma50:
        pts_ma += 7
    if precio_actual > ma200:
        pts_ma += 8
    if ma50 > ma200:  # Golden Cross
        pts_ma += 5
    
    score += pts_ma
    detalles['Tendencia MA'] = {
        'valor': f"{'↑' if precio_actual > ma200 else '↓'} P:{precio_actual:.2f} MA50:{ma50:.2f} MA200:{ma200:.2f}",
        'puntos': pts_ma,
        'max': 20,
        'estado': '🟢' if pts_ma >= 15 else '🟡' if pts_ma >= 10 else '🔴'
    }
    
    # 2. RSI(13) - Estrategia Retorno a Media (0-20 puntos)
    rsi = calcular_rsi(close, period=13).iloc[-1]
    
    if rsi <= 30:
        pts_rsi = 20  # Sobreventa = oportunidad de compra
        estado_rsi = '🟢'
        texto_rsi = f"{rsi:.1f} (Sobreventa - COMPRA)"
    elif rsi >= 70:
        pts_rsi = 0  # Sobrecompra = evitar compra
        estado_rsi = '🔴'
        texto_rsi = f"{rsi:.1f} (Sobrecompra - VENTA)"
    elif 30 < rsi <= 45:
        pts_rsi = 15
        estado_rsi = '🟢'
        texto_rsi = f"{rsi:.1f} (Zona baja)"
    elif 55 <= rsi < 70:
        pts_rsi = 5
        estado_rsi = '🟡'
        texto_rsi = f"{rsi:.1f} (Zona alta)"
    else:
        pts_rsi = 10
        estado_rsi = '🟡'
        texto_rsi = f"{rsi:.1f} (Neutral)"
    
    score += pts_rsi
    detalles['RSI(13)'] = {
        'valor': texto_rsi,
        'puntos': pts_rsi,
        'max': 20,
        'estado': estado_rsi
    }
    
    # 3. Bandas de Bollinger(30) (0-20 puntos)
    bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(close, period=30, std_dev=2)
    bb_pos = (precio_actual - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
    
    if bb_pos <= 10:
        pts_bb = 20  # Muy cerca de banda inferior = oportunidad
        estado_bb = '🟢'
        texto_bb = f"Pos: {bb_pos:.0f}% (Banda inferior)"
    elif bb_pos >= 90:
        pts_bb = 0  # Muy cerca de banda superior = peligro
        estado_bb = '🔴'
        texto_bb = f"Pos: {bb_pos:.0f}% (Banda superior)"
    elif bb_pos <= 30:
        pts_bb = 15
        estado_bb = '🟢'
        texto_bb = f"Pos: {bb_pos:.0f}% (Zona baja)"
    elif bb_pos >= 70:
        pts_bb = 5
        estado_bb = '🟡'
        texto_bb = f"Pos: {bb_pos:.0f}% (Zona alta)"
    else:
        pts_bb = 10
        estado_bb = '🟡'
        texto_bb = f"Pos: {bb_pos:.0f}% (Centro)"
    
    score += pts_bb
    detalles['Bollinger(30)'] = {
        'valor': texto_bb,
        'puntos': pts_bb,
        'max': 20,
        'estado': estado_bb
    }
    
    # 4. Vela de Rechazo (0-15 puntos)
    tipo_rechazo, det_rechazo = detectar_vela_rechazo(hist_df)
    
    if tipo_rechazo == 'alcista':
        pts_rechazo = 15
        estado_rechazo = '🟢'
        texto_rechazo = "Martillo alcista detectado"
    elif tipo_rechazo == 'bajista':
        pts_rechazo = 0
        estado_rechazo = '🔴'
        texto_rechazo = "Estrella fugaz bajista"
    else:
        pts_rechazo = 7
        estado_rechazo = '🟡'
        texto_rechazo = "Sin patrón de rechazo"
    
    score += pts_rechazo
    detalles['Vela Rechazo'] = {
        'valor': texto_rechazo,
        'puntos': pts_rechazo,
        'max': 15,
        'estado': estado_rechazo
    }
    
    # 5. MACD (0-15 puntos)
    macd, signal_line, histogram = calcular_macd(close)
    macd_actual = macd.iloc[-1]
    signal_actual = signal_line.iloc[-1]
    hist_actual = histogram.iloc[-1]
    hist_anterior = histogram.iloc[-2]
    
    if macd_actual > signal_actual and hist_actual > hist_anterior:
        pts_macd = 15
        estado_macd = '🟢'
        texto_macd = "MACD alcista + momentum"
    elif macd_actual > signal_actual:
        pts_macd = 12
        estado_macd = '🟢'
        texto_macd = "MACD alcista"
    elif macd_actual < signal_actual and hist_actual < hist_anterior:
        pts_macd = 0
        estado_macd = '🔴'
        texto_macd = "MACD bajista + momentum"
    elif macd_actual < signal_actual:
        pts_macd = 3
        estado_macd = '🔴'
        texto_macd = "MACD bajista"
    else:
        pts_macd = 7
        estado_macd = '🟡'
        texto_macd = "MACD neutral"
    
    score += pts_macd
    detalles['MACD'] = {
        'valor': texto_macd,
        'puntos': pts_macd,
        'max': 15,
        'estado': estado_macd
    }
    
    # 6. Volumen (0-10 puntos)
    if volume is not None and len(volume) >= 20:
        vol_ma = volume.rolling(window=20).mean().iloc[-1]
        vol_actual = volume.iloc[-5:].mean()
        vol_ratio = vol_actual / vol_ma if vol_ma > 0 else 1
        
        # Volumen creciente en tendencia alcista es positivo
        if vol_ratio > 1.2 and precio_actual > ma50:
            pts_vol = 10
            estado_vol = '🟢'
            texto_vol = f"Alto ({vol_ratio:.1f}x) + tendencia ↑"
        elif vol_ratio > 1.0:
            pts_vol = 7
            estado_vol = '🟡'
            texto_vol = f"Normal ({vol_ratio:.1f}x)"
        else:
            pts_vol = 5
            estado_vol = '🟡'
            texto_vol = f"Bajo ({vol_ratio:.1f}x)"
    else:
        pts_vol = 5
        estado_vol = '⚪'
        texto_vol = "N/A"
    
    score += pts_vol
    detalles['Volumen'] = {
        'valor': texto_vol,
        'puntos': pts_vol,
        'max': 10,
        'estado': estado_vol
    }
    
    return score, detalles


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
    indicadores_con_datos = 0
    total_indicadores = 6
    
    # PER (0-20 puntos)
    per = info.get('trailingPE')
    if per and per > 0:
        indicadores_con_datos += 1
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
        score += 10  # Puntos neutrales
        detalles['PER'] = {'valor': 'N/A', 'puntos': 10, 'max': 20, 'estado': '⚪'}
    
    # EV/EBITDA (0-20 puntos)
    ev_ebitda = info.get('enterpriseToEbitda')
    if ev_ebitda and ev_ebitda > 0:
        indicadores_con_datos += 1
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
        score += 10  # Puntos neutrales
        detalles['EV/EBITDA'] = {'valor': 'N/A', 'puntos': 10, 'max': 20, 'estado': '⚪'}
    
    # P/BV (0-15 puntos)
    p_bv = info.get('priceToBook')
    if p_bv and p_bv > 0:
        indicadores_con_datos += 1
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
        score += 7  # Puntos neutrales
        detalles['P/BV'] = {'valor': 'N/A', 'puntos': 7, 'max': 15, 'estado': '⚪'}
    
    # ROE (0-20 puntos)
    roe = info.get('returnOnEquity')
    if roe:
        indicadores_con_datos += 1
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
        score += 10  # Puntos neutrales
        detalles['ROE'] = {'valor': 'N/A', 'puntos': 10, 'max': 20, 'estado': '⚪'}
    
    # Deuda/Equity (0-15 puntos)
    debt_equity = info.get('debtToEquity')
    if debt_equity is not None and debt_equity >= 0:
        indicadores_con_datos += 1
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
        score += 7  # Puntos neutrales
        detalles['Deuda/Equity'] = {'valor': 'N/A', 'puntos': 7, 'max': 15, 'estado': '⚪'}
    
    # Dividend Yield (0-10 puntos)
    div_yield = info.get('dividendYield')
    if div_yield and div_yield > 0:
        indicadores_con_datos += 1
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
        score += 5  # Puntos neutrales
        detalles['Dividendo'] = {'valor': 'N/A', 'puntos': 5, 'max': 10, 'estado': '⚪'}
    
    # Añadir indicador de calidad de datos
    detalles['_datos_disponibles'] = f"{indicadores_con_datos}/{total_indicadores}"
    
    return score, detalles


def score_tecnico(hist):
    """
    Calcula el score técnico (0-100) con estrategia de Retorno a la Media.
    Incorpora: RSI(13), Bandas de Bollinger(30), Velas de Rechazo, MACD, MAs.
    """
    score = 0
    detalles = {}
    
    if hist.empty or len(hist) < 50:
        return 0, {'error': 'Datos insuficientes'}
    
    close = hist['Close']
    precio_actual = close.iloc[-1]
    
    # Verificar si tenemos suficientes datos para todos los indicadores
    tiene_200 = len(close) >= 200
    
    # 1. Precio vs MA50 y MA200 (0-15 puntos)
    ma50 = close.rolling(window=50).mean().iloc[-1]
    
    if tiene_200:
        ma200 = close.rolling(window=200).mean().iloc[-1]
        if precio_actual > ma200 and precio_actual > ma50:
            pts = 15
            estado = '🟢'
            texto = f"Precio > MA50 ({ma50:.2f}) y MA200 ({ma200:.2f})"
        elif precio_actual > ma50:
            pts = 10
            estado = '🟡'
            texto = f"Precio > MA50 ({ma50:.2f})"
        else:
            pts = 0
            estado = '🔴'
            texto = f"Precio < MA50 ({ma50:.2f})"
    else:
        if precio_actual > ma50:
            pts = 10
            estado = '🟢'
            texto = f"Precio > MA50 ({ma50:.2f})"
        else:
            pts = 0
            estado = '🔴'
            texto = f"Precio < MA50 ({ma50:.2f})"
    
    score += pts
    detalles['Tendencia MAs'] = {'valor': texto, 'puntos': pts, 'max': 15, 'estado': estado}
    
    # 2. Golden/Death Cross (0-15 puntos)
    if tiene_200:
        ma50_series = close.rolling(window=50).mean()
        ma200_series = close.rolling(window=200).mean()
        
        golden_cross = ma50_series.iloc[-1] > ma200_series.iloc[-1]
        
        if golden_cross:
            pts = 15
            estado = '🟢'
            texto = "Golden Cross (MA50 > MA200)"
        else:
            pts = 0
            estado = '🔴'
            texto = "Death Cross (MA50 < MA200)"
        score += pts
        detalles['Cruce de Medias'] = {'valor': texto, 'puntos': pts, 'max': 15, 'estado': estado}
    else:
        score += 7  # Puntos neutrales
        detalles['Cruce de Medias'] = {'valor': 'N/A (< 200 días)', 'puntos': 7, 'max': 15, 'estado': '⚪'}
    
    # 3. RSI(13) - Estrategia Retorno a Media (0-20 puntos)
    rsi = calcular_rsi(close, period=13).iloc[-1]
    
    if rsi <= 30:
        pts = 20  # Sobreventa = oportunidad de compra
        estado = '🟢'
        texto = f"{rsi:.1f} (Sobreventa - OPORTUNIDAD)"
    elif rsi >= 70:
        pts = 0  # Sobrecompra = peligro
        estado = '🔴'
        texto = f"{rsi:.1f} (Sobrecompra - PRECAUCIÓN)"
    elif 30 < rsi <= 45:
        pts = 15
        estado = '🟢'
        texto = f"{rsi:.1f} (Zona favorable)"
    elif 55 <= rsi < 70:
        pts = 5
        estado = '🟡'
        texto = f"{rsi:.1f} (Zona alta)"
    else:
        pts = 10
        estado = '🟡'
        texto = f"{rsi:.1f} (Neutral)"
    
    score += pts
    detalles['RSI(13)'] = {'valor': texto, 'puntos': pts, 'max': 20, 'estado': estado}
    
    # 4. Bandas de Bollinger(30) (0-20 puntos)
    if len(close) >= 30:
        bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(close, period=30, std_dev=2)
        bb_u = bb_upper.iloc[-1]
        bb_l = bb_lower.iloc[-1]
        bb_pos = (precio_actual - bb_l) / (bb_u - bb_l) * 100 if (bb_u - bb_l) > 0 else 50
        
        if bb_pos <= 10:
            pts = 20  # Banda inferior = oportunidad compra
            estado = '🟢'
            texto = f"Pos: {bb_pos:.0f}% (Banda inferior - COMPRA)"
        elif bb_pos >= 90:
            pts = 0  # Banda superior = peligro
            estado = '🔴'
            texto = f"Pos: {bb_pos:.0f}% (Banda superior - VENTA)"
        elif bb_pos <= 30:
            pts = 15
            estado = '🟢'
            texto = f"Pos: {bb_pos:.0f}% (Zona baja)"
        elif bb_pos >= 70:
            pts = 5
            estado = '🟡'
            texto = f"Pos: {bb_pos:.0f}% (Zona alta)"
        else:
            pts = 10
            estado = '🟡'
            texto = f"Pos: {bb_pos:.0f}% (Centro)"
        
        score += pts
        detalles['Bollinger(30)'] = {'valor': texto, 'puntos': pts, 'max': 20, 'estado': estado}
    else:
        score += 10
        detalles['Bollinger(30)'] = {'valor': 'N/A', 'puntos': 10, 'max': 20, 'estado': '⚪'}
    
    # 5. Vela de Rechazo (0-10 puntos)
    tipo_rechazo, _ = detectar_vela_rechazo(hist)
    
    if tipo_rechazo == 'alcista':
        pts = 10
        estado = '🟢'
        texto = "Martillo alcista ↑"
    elif tipo_rechazo == 'bajista':
        pts = 0
        estado = '🔴'
        texto = "Estrella fugaz ↓"
    else:
        pts = 5
        estado = '🟡'
        texto = "Sin patrón"
    
    score += pts
    detalles['Vela Rechazo'] = {'valor': texto, 'puntos': pts, 'max': 10, 'estado': estado}
    
    # 6. MACD (0-10 puntos)
    macd, signal_line, histogram = calcular_macd(close)
    macd_actual = macd.iloc[-1]
    signal_actual = signal_line.iloc[-1]
    hist_actual = histogram.iloc[-1]
    hist_anterior = histogram.iloc[-2] if len(histogram) > 1 else 0
    
    if macd_actual > signal_actual and hist_actual > hist_anterior:
        pts = 10
        estado = '🟢'
        texto = "MACD alcista + momentum ↑"
    elif macd_actual > signal_actual:
        pts = 7
        estado = '🟢'
        texto = "MACD alcista"
    elif macd_actual < signal_actual and hist_actual < hist_anterior:
        pts = 0
        estado = '🔴'
        texto = "MACD bajista + momentum ↓"
    elif macd_actual < signal_actual:
        pts = 3
        estado = '🔴'
        texto = "MACD bajista"
    else:
        pts = 5
        estado = '🟡'
        texto = "MACD neutral"
    
    score += pts
    detalles['MACD'] = {'valor': texto, 'puntos': pts, 'max': 10, 'estado': estado}
    
    # 7. Volumen (0-10 puntos)
    if 'Volume' in hist.columns:
        volume = hist['Volume']
        if len(volume) >= 20:
            vol_ma = volume.rolling(window=20).mean().iloc[-1]
            vol_actual = volume.iloc[-5:].mean()
            vol_ratio = vol_actual / vol_ma if vol_ma > 0 else 1
            
            if vol_ratio > 1.3 and precio_actual > ma50:
                pts = 10
                estado = '🟢'
                texto = f"Alto ({vol_ratio:.1f}x) + tendencia ↑"
            elif vol_ratio > 1.0:
                pts = 7
                estado = '🟡'
                texto = f"Normal ({vol_ratio:.1f}x)"
            else:
                pts = 4
                estado = '🟡'
                texto = f"Bajo ({vol_ratio:.1f}x)"
        else:
            pts = 5
            texto = "Datos insuf."
            estado = '⚪'
    else:
        pts = 5
        texto = "N/A"
        estado = '⚪'
    
    score += pts
    detalles['Volumen'] = {'valor': texto, 'puntos': pts, 'max': 10, 'estado': estado}
    
    return score, detalles


def generar_recomendacion(score_fund, score_tech, score_regimen=None, peso_fundamental=0.4, peso_tecnico=0.3, peso_regimen=0.3):
    """Genera recomendación final basada en scores."""
    if score_regimen is not None:
        score_total = score_fund * peso_fundamental + score_tech * peso_tecnico + score_regimen * peso_regimen
    else:
        # Sin HMM/GARCH, usar solo fundamental y técnico
        peso_fund_adj = peso_fundamental / (peso_fundamental + peso_tecnico)
        peso_tech_adj = peso_tecnico / (peso_fundamental + peso_tecnico)
        score_total = score_fund * peso_fund_adj + score_tech * peso_tech_adj
    
    if score_total >= 80:
        recomendacion = "COMPRA FUERTE"
        color = "🟢"
        explicacion = "Los indicadores fundamentales, técnicos y de régimen están alineados positivamente. Buen momento para entrar."
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
# FUNCIONES HMM (Hidden Markov Model)
# --------------------------------------------------
def detectar_regimenes_hmm(returns, n_states=3):
    """
    Detecta regímenes de mercado usando Hidden Markov Model.
    
    Estados:
    - 0: Bajista (media negativa, alta volatilidad)
    - 1: Lateral (media ~0, volatilidad media)  
    - 2: Alcista (media positiva, baja volatilidad)
    
    Returns: dict con régimen actual, probabilidades, historial
    """
    if not HMM_AVAILABLE:
        return None
    
    try:
        # Preparar datos
        returns_clean = returns.dropna().values.reshape(-1, 1)
        
        if len(returns_clean) < 50:
            return None
        
        # Entrenar HMM
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(returns_clean)
        
        # Obtener estados ocultos
        hidden_states = model.predict(returns_clean)
        state_probs = model.predict_proba(returns_clean)
        
        # Identificar qué estado es cuál basándose en media y varianza
        state_means = model.means_.flatten()
        state_vars = np.array([model.covars_[i][0][0] for i in range(n_states)])
        
        # Ordenar estados: bajista (peor media), lateral, alcista (mejor media)
        sorted_indices = np.argsort(state_means)
        state_mapping = {sorted_indices[0]: 'bajista', 
                        sorted_indices[1]: 'lateral', 
                        sorted_indices[2]: 'alcista'}
        
        # Estado actual y probabilidades
        current_state_idx = hidden_states[-1]
        current_state = state_mapping[current_state_idx]
        current_probs = state_probs[-1]
        
        # Probabilidades por régimen
        prob_bajista = current_probs[sorted_indices[0]]
        prob_lateral = current_probs[sorted_indices[1]]
        prob_alcista = current_probs[sorted_indices[2]]
        
        # Calcular duración media en cada régimen
        duraciones = {s: [] for s in range(n_states)}
        current_duration = 1
        for i in range(1, len(hidden_states)):
            if hidden_states[i] == hidden_states[i-1]:
                current_duration += 1
            else:
                duraciones[hidden_states[i-1]].append(current_duration)
                current_duration = 1
        duraciones[hidden_states[-1]].append(current_duration)
        
        duracion_media = {state_mapping[s]: np.mean(duraciones[s]) if duraciones[s] else 0 
                         for s in range(n_states)}
        
        # Matriz de transición
        transmat = model.transmat_
        
        return {
            'estado_actual': current_state,
            'prob_bajista': prob_bajista,
            'prob_lateral': prob_lateral,
            'prob_alcista': prob_alcista,
            'historial_estados': [state_mapping[s] for s in hidden_states],
            'duracion_media': duracion_media,
            'matriz_transicion': transmat,
            'state_mapping': state_mapping,
            'sorted_indices': sorted_indices,
            'means': state_means[sorted_indices],
            'vars': state_vars[sorted_indices]
        }
        
    except Exception as e:
        return None


def score_regimen_hmm(hmm_result):
    """
    Calcula score (0-100) basado en el régimen HMM detectado.
    """
    if hmm_result is None:
        return 50, {'error': 'HMM no disponible'}
    
    detalles = {}
    score = 0
    
    estado = hmm_result['estado_actual']
    prob_alcista = hmm_result['prob_alcista']
    prob_bajista = hmm_result['prob_bajista']
    prob_lateral = hmm_result['prob_lateral']
    
    # Score base por régimen (0-50 pts)
    if estado == 'alcista':
        pts_regimen = 50
        estado_emoji = '🟢'
    elif estado == 'lateral':
        pts_regimen = 25
        estado_emoji = '🟡'
    else:  # bajista
        pts_regimen = 0
        estado_emoji = '🔴'
    
    score += pts_regimen
    detalles['Régimen Actual'] = {
        'valor': f"{estado.capitalize()} {estado_emoji}",
        'puntos': pts_regimen,
        'max': 50,
        'estado': estado_emoji
    }
    
    # Bonus por probabilidad de régimen alcista (0-30 pts)
    pts_prob = int(prob_alcista * 30)
    score += pts_prob
    detalles['Prob. Alcista'] = {
        'valor': f"{prob_alcista:.1%}",
        'puntos': pts_prob,
        'max': 30,
        'estado': '🟢' if prob_alcista > 0.5 else '🟡' if prob_alcista > 0.3 else '🔴'
    }
    
    # Penalización por probabilidad bajista (0-20 pts, invertido)
    pts_bajista = int((1 - prob_bajista) * 20)
    score += pts_bajista
    detalles['Prob. Bajista'] = {
        'valor': f"{prob_bajista:.1%}",
        'puntos': pts_bajista,
        'max': 20,
        'estado': '🟢' if prob_bajista < 0.2 else '🟡' if prob_bajista < 0.4 else '🔴'
    }
    
    return min(score, 100), detalles


# --------------------------------------------------
# FUNCIONES GARCH
# --------------------------------------------------
def predecir_volatilidad_garch(returns, horizon=22):
    """
    Predice volatilidad futura usando GARCH(1,1).
    
    Args:
        returns: Serie de retornos
        horizon: Días a predecir (22 = 1 mes)
    
    Returns: dict con volatilidad predicha, intervalos de confianza
    """
    if not GARCH_AVAILABLE:
        return None
    
    try:
        # Preparar datos (GARCH necesita retornos en porcentaje)
        returns_pct = returns.dropna() * 100
        
        if len(returns_pct) < 100:
            return None
        
        # Ajustar modelo GARCH(1,1)
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
        fitted = model.fit(disp='off', show_warning=False)
        
        # Predicción de volatilidad
        forecast = fitted.forecast(horizon=horizon)
        
        # Volatilidad predicha (anualizada)
        vol_diaria_predicha = np.sqrt(forecast.variance.values[-1, :])
        vol_media_predicha = vol_diaria_predicha.mean()
        vol_anualizada = vol_media_predicha * np.sqrt(252) / 100  # Convertir a decimal
        
        # Volatilidad histórica para comparar
        vol_historica = returns.std() * np.sqrt(252)
        
        # Cambio en volatilidad
        cambio_vol = (vol_anualizada - vol_historica) / vol_historica * 100
        
        # Parámetros del modelo
        omega = fitted.params.get('omega', 0)
        alpha = fitted.params.get('alpha[1]', 0)
        beta = fitted.params.get('beta[1]', 0)
        persistencia = alpha + beta
        
        # VaR predicho (95%)
        var_95 = returns_pct.mean() - 1.645 * vol_media_predicha
        
        return {
            'vol_predicha_anual': vol_anualizada,
            'vol_historica_anual': vol_historica,
            'cambio_vol_pct': cambio_vol,
            'vol_diaria_predicha': vol_diaria_predicha / 100,  # Serie de predicciones
            'persistencia': persistencia,
            'var_95': var_95 / 100,
            'params': {'omega': omega, 'alpha': alpha, 'beta': beta}
        }
        
    except Exception as e:
        return None


def score_volatilidad_garch(garch_result, vol_threshold_low=0.15, vol_threshold_high=0.30):
    """
    Calcula score (0-100) basado en predicción GARCH de volatilidad.
    
    Menor volatilidad predicha = mejor score
    """
    if garch_result is None:
        return 50, {'error': 'GARCH no disponible'}
    
    detalles = {}
    score = 0
    
    vol_predicha = garch_result['vol_predicha_anual']
    vol_historica = garch_result['vol_historica_anual']
    cambio_vol = garch_result['cambio_vol_pct']
    persistencia = garch_result['persistencia']
    
    # Score por nivel de volatilidad predicha (0-40 pts)
    if vol_predicha < vol_threshold_low:
        pts_vol = 40
        estado_vol = '🟢'
    elif vol_predicha < 0.20:
        pts_vol = 30
        estado_vol = '🟢'
    elif vol_predicha < vol_threshold_high:
        pts_vol = 20
        estado_vol = '🟡'
    elif vol_predicha < 0.40:
        pts_vol = 10
        estado_vol = '🔴'
    else:
        pts_vol = 0
        estado_vol = '🔴'
    
    score += pts_vol
    detalles['Vol. Predicha'] = {
        'valor': f"{vol_predicha:.1%} anual",
        'puntos': pts_vol,
        'max': 40,
        'estado': estado_vol
    }
    
    # Score por cambio de volatilidad (0-30 pts)
    # Volatilidad decreciente es positivo
    if cambio_vol < -10:
        pts_cambio = 30
        estado_cambio = '🟢'
        texto_cambio = f"Bajando ({cambio_vol:+.0f}%)"
    elif cambio_vol < 0:
        pts_cambio = 20
        estado_cambio = '🟢'
        texto_cambio = f"Estable ({cambio_vol:+.0f}%)"
    elif cambio_vol < 10:
        pts_cambio = 15
        estado_cambio = '🟡'
        texto_cambio = f"Ligero aumento ({cambio_vol:+.0f}%)"
    elif cambio_vol < 25:
        pts_cambio = 5
        estado_cambio = '🔴'
        texto_cambio = f"Aumentando ({cambio_vol:+.0f}%)"
    else:
        pts_cambio = 0
        estado_cambio = '🔴'
        texto_cambio = f"Fuerte aumento ({cambio_vol:+.0f}%)"
    
    score += pts_cambio
    detalles['Tendencia Vol.'] = {
        'valor': texto_cambio,
        'puntos': pts_cambio,
        'max': 30,
        'estado': estado_cambio
    }
    
    # Score por persistencia (0-30 pts)
    # Alta persistencia = volatilidad tiende a mantenerse
    if persistencia < 0.8:
        pts_pers = 30
        estado_pers = '🟢'
        texto_pers = f"Baja ({persistencia:.2f})"
    elif persistencia < 0.9:
        pts_pers = 20
        estado_pers = '🟡'
        texto_pers = f"Media ({persistencia:.2f})"
    elif persistencia < 0.98:
        pts_pers = 10
        estado_pers = '🟡'
        texto_pers = f"Alta ({persistencia:.2f})"
    else:
        pts_pers = 0
        estado_pers = '🔴'
        texto_pers = f"Muy alta ({persistencia:.2f})"
    
    score += pts_pers
    detalles['Persistencia'] = {
        'valor': texto_pers,
        'puntos': pts_pers,
        'max': 30,
        'estado': estado_pers
    }
    
    return min(score, 100), detalles


def score_regimen_combinado(hmm_result, garch_result):
    """
    Combina scores de HMM y GARCH en un score de régimen único.
    HMM: 60% (detecta tendencia)
    GARCH: 40% (detecta riesgo)
    """
    score_hmm, detalles_hmm = score_regimen_hmm(hmm_result)
    score_garch, detalles_garch = score_volatilidad_garch(garch_result)
    
    # Si ambos están disponibles, combinar
    if 'error' not in detalles_hmm and 'error' not in detalles_garch:
        score_total = score_hmm * 0.6 + score_garch * 0.4
        detalles = {**detalles_hmm, **detalles_garch}
    elif 'error' not in detalles_hmm:
        score_total = score_hmm
        detalles = detalles_hmm
    elif 'error' not in detalles_garch:
        score_total = score_garch
        detalles = detalles_garch
    else:
        score_total = 50
        detalles = {'error': 'Modelos no disponibles'}
    
    return score_total, detalles


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
    
    st.caption("💡 España: añade .MC (ej: SAN.MC)")
    st.caption("💡 Alemania: añade .DE (ej: BMW.DE)")
    st.caption("💡 Francia: añade .PA (ej: BNP.PA)")

st.sidebar.markdown("---")

# Modo de análisis
st.sidebar.subheader("🎯 Modo de Análisis")
modo = st.sidebar.radio(
    "¿Qué quieres analizar?",
    ["🔍 Acción individual", "🎯 Recomendación compra/venta", "📊 Señales de Trading", "🌍 Análisis por Región", "📈 Comparador de Activos", "📊 Cartera (2+ activos)"],
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

# Regiones geográficas para el escáner
REGIONES = {
    "🇺🇸 USA Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
    "🇺🇸 USA Financiero": ["JPM", "BAC", "GS", "V", "MA", "WFC", "C", "AXP"],
    "🇺🇸 USA Consumo": ["KO", "PEP", "MCD", "NKE", "SBUX", "WMT", "COST", "HD"],
    "🇺🇸 USA Salud": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "🇪🇸 España (IBEX)": ["SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC", "CABK.MC", "FER.MC", "AENA.MC", "AMS.MC"],
    "🇩🇪 Alemania (DAX)": ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BMW.DE", "MBG.DE", "VOW3.DE"],
    "🇫🇷 Francia (CAC)": ["MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA", "ACA.PA", "SU.PA"],
    "🇮🇹 Italia (FTSE MIB)": ["ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "STLAM.MI", "RACE.MI", "G.MI", "TEN.MI"],
    "🇬🇧 Reino Unido (FTSE)": ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "BARC.L"],
    "🇨🇭 Suiza": ["NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ABBN.SW", "SREN.SW", "CSGN.SW", "ZURN.SW"],
    "🇳🇱 Holanda": ["ASML.AS", "INGA.AS", "PHIA.AS", "AD.AS", "HEIA.AS", "WKL.AS", "UNA.AS", "AKZA.AS"],
    "🇵🇹 Portugal": ["GALP.LS", "EDP.LS", "SON.LS", "BCP.LS", "JMT.LS", "NOS.LS"],
    "🇧🇪 Bélgica": ["ABI.BR", "KBC.BR", "UCB.BR", "SOLB.BR", "ACKB.BR", "GBLB.BR"],
    "🇪🇺 Europa Mix": ["ASML.AS", "NVO", "NESN.SW", "ROG.SW", "NOVN.SW", "SHEL", "UL", "GSK"],
    "🌍 ETFs Globales": ["SPY", "QQQ", "EEM", "VGK", "EFA", "VWO", "GLD", "SLV"],
    "🌏 Asia": ["BABA", "TSM", "SONY", "TM", "NIO", "JD", "BIDU", "PDD"],
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
            value="SAN.MC",
            help="Ejemplo: AAPL, MSFT, SAN.MC"
        ).strip().upper()
    TICKERS = [TICKER_INDIVIDUAL] if TICKER_INDIVIDUAL else []
elif modo == "🌍 Análisis por Región":
    # Selector de regiones
    regiones_seleccionadas = st.sidebar.multiselect(
        "Selecciona regiones a analizar",
        list(REGIONES.keys()),
        default=["🇪🇸 España (IBEX)", "🇺🇸 USA Tech"]
    )
    TICKERS = []  # No se usa en este modo
else:
    if usar_predefinidos:
        categoria = st.sidebar.selectbox("Categoría", list(tickers_populares.keys()))
        TICKERS = tickers_populares[categoria]
        st.sidebar.write(f"Tickers: {', '.join(TICKERS)}")
    else:
        tickers_input = st.sidebar.text_input(
            "Introduce tickers (separados por coma)",
            value="AAPL, MSFT, SAN.MC, NVO",
            help="Ejemplo: AAPL, MSFT, GOOGL. Para España añade .MC (ej: SAN.MC)"
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
    ticker_original = ticker
    
    # Lista de fallback si el ticker principal falla
    TICKERS_FALLBACK = [ticker, "SAN.MC", "BBVA.MC", "AAPL", "MSFT"]
    # Eliminar duplicados manteniendo orden
    TICKERS_FALLBACK = list(dict.fromkeys(TICKERS_FALLBACK))
    
    try:
        with st.spinner(f"Cargando datos de {ticker}..."):
            data_accion = obtener_info_accion(ticker, periodo)
        
        # Si no hay datos o el histórico está vacío, probar con fallback
        if data_accion is None or data_accion['history'].empty:
            st.warning(f"⚠️ No se pudieron cargar datos de {ticker}. Probando alternativas...")
            
            for fallback_ticker in TICKERS_FALLBACK[1:]:
                with st.spinner(f"Probando con {fallback_ticker}..."):
                    data_accion = obtener_info_accion(fallback_ticker, periodo)
                    if data_accion and not data_accion['history'].empty:
                        ticker = fallback_ticker
                        st.info(f"✅ Mostrando datos de {ticker} como alternativa.")
                        break
        
        if data_accion is None or data_accion['history'].empty:
            st.error(f"No se pudieron obtener datos. Verifica tu conexión a internet.")
            st.stop()
    
    except Exception as e:
        if "RateLimit" in str(type(e).__name__) or "rate" in str(e).lower():
            st.error("⚠️ Yahoo Finance ha bloqueado temporalmente las peticiones. Espera 1-2 minutos y recarga la página.")
        else:
            st.error(f"Error obteniendo datos: {e}")
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
        
        # Gráfico de Bandas de Bollinger y RSI
        if len(hist) >= 30:
            st.markdown("---")
            st.markdown("### 📊 Análisis Técnico: Bollinger(30) + RSI(13)")
            
            close = hist['Close']
            bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(close, period=30, std_dev=2)
            rsi_series = calcular_rsi(close, period=13)
            
            # Crear figura con 2 subplots
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Determinar cuántos días mostrar
            ultimos_dias = min(120, len(close))
            
            # Subplot 1: Precio con Bandas de Bollinger
            ax1.plot(close.index[-ultimos_dias:], close.iloc[-ultimos_dias:], 'b-', linewidth=1.5, label='Precio')
            ax1.plot(bb_upper.index[-ultimos_dias:], bb_upper.iloc[-ultimos_dias:], 'r--', linewidth=1, label='Banda Superior', alpha=0.7)
            ax1.plot(bb_middle.index[-ultimos_dias:], bb_middle.iloc[-ultimos_dias:], 'g-', linewidth=1, label='Media (30)', alpha=0.7)
            ax1.plot(bb_lower.index[-ultimos_dias:], bb_lower.iloc[-ultimos_dias:], 'r--', linewidth=1, label='Banda Inferior', alpha=0.7)
            
            ax1.fill_between(bb_upper.index[-ultimos_dias:], bb_lower.iloc[-ultimos_dias:], bb_upper.iloc[-ultimos_dias:], 
                             alpha=0.1, color='blue')
            
            # Marcar precio actual
            ax1.scatter(close.index[-1], close.iloc[-1], color='blue', s=100, zorder=5)
            
            ax1.set_title(f'{ticker} - Bandas de Bollinger(30, 2)', fontsize=12)
            ax1.set_ylabel(f'Precio ({info.get("currency", "USD")})')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: RSI
            ax2.plot(rsi_series.index[-ultimos_dias:], rsi_series.iloc[-ultimos_dias:], 'purple', linewidth=1.5)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            ax2.fill_between(rsi_series.index[-ultimos_dias:], 30, 70, alpha=0.1, color='gray')
            
            # Colorear zonas de sobrecompra/sobreventa
            rsi_vals = rsi_series.iloc[-ultimos_dias:]
            ax2.fill_between(rsi_vals.index, rsi_vals, 70, where=(rsi_vals >= 70), alpha=0.3, color='red')
            ax2.fill_between(rsi_vals.index, rsi_vals, 30, where=(rsi_vals <= 30), alpha=0.3, color='green')
            
            ax2.set_ylabel('RSI(13)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Indicadores actuales
            rsi_actual = rsi_series.iloc[-1]
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            # RSI con colores
            if rsi_actual <= 30:
                col1.metric("RSI(13)", f"{rsi_actual:.1f}", delta="Sobreventa 🟢")
            elif rsi_actual >= 70:
                col1.metric("RSI(13)", f"{rsi_actual:.1f}", delta="Sobrecompra 🔴")
            else:
                col1.metric("RSI(13)", f"{rsi_actual:.1f}", delta="Neutral")
            
            # Posición Bollinger
            if bb_pos <= 20:
                col2.metric("Pos. Bollinger", f"{bb_pos:.0f}%", delta="Zona baja 🟢")
            elif bb_pos >= 80:
                col2.metric("Pos. Bollinger", f"{bb_pos:.0f}%", delta="Zona alta 🔴")
            else:
                col2.metric("Pos. Bollinger", f"{bb_pos:.0f}%", delta="Centro")
            
            col3.metric("Banda Superior", f"{bb_upper.iloc[-1]:.2f}")
            col4.metric("Banda Inferior", f"{bb_lower.iloc[-1]:.2f}")
    
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
    ticker_original = ticker
    
    # Lista de fallback si el ticker principal falla
    TICKERS_FALLBACK = [ticker, "SAN.MC", "BBVA.MC", "AAPL", "MSFT"]
    TICKERS_FALLBACK = list(dict.fromkeys(TICKERS_FALLBACK))
    
    try:
        with st.spinner(f"Analizando {ticker}..."):
            data_accion = obtener_info_accion(ticker, "1y")
        
        # Si no hay datos o el histórico está vacío, probar con fallback
        if data_accion is None or data_accion['history'].empty:
            st.warning(f"⚠️ No se pudieron cargar datos de {ticker}. Probando alternativas...")
            
            for fallback_ticker in TICKERS_FALLBACK[1:]:
                with st.spinner(f"Probando con {fallback_ticker}..."):
                    data_accion = obtener_info_accion(fallback_ticker, "1y")
                    if data_accion and not data_accion['history'].empty:
                        ticker = fallback_ticker
                        st.info(f"✅ Mostrando análisis de {ticker} como alternativa.")
                        break
        
        if data_accion is None or data_accion['history'].empty:
            st.error(f"No se pudieron obtener datos. Verifica tu conexión a internet.")
            st.stop()
        
        info = data_accion['info']
        hist = data_accion['history']
        hist_largo = hist
        
    except Exception as e:
        if "RateLimit" in str(type(e).__name__) or "rate" in str(e).lower():
            st.error("⚠️ Yahoo Finance ha bloqueado temporalmente las peticiones. Espera 1-2 minutos y recarga la página.")
            st.info("💡 Esto ocurre cuando hay muchas consultas seguidas. Es una limitación de Yahoo Finance, no de la app.")
        else:
            st.error(f"Error obteniendo datos: {e}")
        st.stop()
    
    # Calcular scores
    s_fund, detalles_fund = score_fundamental(info)
    s_tech, detalles_tech = score_tecnico(hist_largo if not hist_largo.empty else hist)
    
    # Calcular HMM y GARCH
    returns = hist['Close'].pct_change().dropna()
    
    hmm_result = None
    garch_result = None
    s_regimen = None
    detalles_regimen = None
    
    if HMM_AVAILABLE or GARCH_AVAILABLE:
        with st.spinner("Analizando regímenes de mercado (HMM/GARCH)..."):
            if HMM_AVAILABLE:
                hmm_result = detectar_regimenes_hmm(returns)
            if GARCH_AVAILABLE:
                garch_result = predecir_volatilidad_garch(returns)
            
            if hmm_result is not None or garch_result is not None:
                s_regimen, detalles_regimen = score_regimen_combinado(hmm_result, garch_result)
    
    # Generar recomendación (con o sin HMM/GARCH)
    rec = generar_recomendacion(s_fund, s_tech, s_regimen, peso_fundamental)
    
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
    if s_regimen is not None:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
        col3 = None
    
    with col1:
        st.markdown(f"### 📊 Score Fundamental: {s_fund}/100")
        st.progress(s_fund / 100)
        
        for indicador, datos in detalles_fund.items():
            if indicador != 'error' and not indicador.startswith('_'):
                st.markdown(f"{datos['estado']} **{indicador}**: {datos['valor']} ({datos['puntos']}/{datos['max']} pts)")
        
        # Mostrar calidad de datos
        if '_datos_disponibles' in detalles_fund:
            st.caption(f"📊 Datos disponibles: {detalles_fund['_datos_disponibles']}")
    
    with col2:
        st.markdown(f"### 📈 Score Técnico: {s_tech}/100")
        st.progress(s_tech / 100)
        
        for indicador, datos in detalles_tech.items():
            if indicador != 'error':
                st.markdown(f"{datos['estado']} **{indicador}**: {datos['valor']} ({datos['puntos']}/{datos['max']} pts)")
    
    if col3 is not None and s_regimen is not None and detalles_regimen is not None:
        with col3:
            st.markdown(f"### 🔮 Score Régimen: {s_regimen:.0f}/100")
            st.progress(s_regimen / 100)
            
            if 'error' not in detalles_regimen:
                for indicador, datos in detalles_regimen.items():
                    st.markdown(f"{datos['estado']} **{indicador}**: {datos['valor']} ({datos['puntos']}/{datos['max']} pts)")
            else:
                st.info("Modelos HMM/GARCH no disponibles")
    
    # --- VISUALIZACIÓN HMM ---
    if hmm_result is not None:
        st.markdown("---")
        st.markdown("### 🔮 Análisis de Regímenes (HMM)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de regímenes sobre precio
            fig, ax = plt.subplots(figsize=(12, 5))
            
            close = hist['Close']
            estados = hmm_result['historial_estados']
            
            # Mapear colores a estados
            colores = {'alcista': '#00ff88', 'lateral': '#ffaa00', 'bajista': '#ff4444'}
            
            # Rellenar fondo según régimen
            for i in range(len(estados)):
                if i < len(close):
                    ax.axvspan(close.index[i], close.index[min(i+1, len(close)-1)], 
                              alpha=0.3, color=colores[estados[i]], linewidth=0)
            
            ax.plot(close.index[-len(estados):], close.iloc[-len(estados):], 'b-', linewidth=1.5)
            ax.set_title(f'{ticker} - Regímenes de Mercado (HMM)')
            ax.set_ylabel('Precio')
            ax.grid(True, alpha=0.3)
            
            # Leyenda manual
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#00ff88', alpha=0.3, label='Alcista'),
                Patch(facecolor='#ffaa00', alpha=0.3, label='Lateral'),
                Patch(facecolor='#ff4444', alpha=0.3, label='Bajista')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Estado Actual**")
            estado = hmm_result['estado_actual']
            if estado == 'alcista':
                st.success(f"🟢 ALCISTA")
            elif estado == 'lateral':
                st.warning(f"🟡 LATERAL")
            else:
                st.error(f"🔴 BAJISTA")
            
            st.markdown("**Probabilidades**")
            st.write(f"- Alcista: {hmm_result['prob_alcista']:.1%}")
            st.write(f"- Lateral: {hmm_result['prob_lateral']:.1%}")
            st.write(f"- Bajista: {hmm_result['prob_bajista']:.1%}")
            
            st.markdown("**Duración media (días)**")
            for estado, dur in hmm_result['duracion_media'].items():
                st.write(f"- {estado.capitalize()}: {dur:.0f}")
    
    # --- VISUALIZACIÓN GARCH ---
    if garch_result is not None:
        st.markdown("---")
        st.markdown("### 📊 Predicción de Volatilidad (GARCH)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Volatilidad realizada (rolling 22 días)
            vol_realizada = returns.rolling(window=22).std() * np.sqrt(252)
            ax.plot(vol_realizada.index, vol_realizada * 100, 'b-', linewidth=1, label='Vol. Realizada (22d)')
            
            # Línea de volatilidad predicha
            ax.axhline(y=garch_result['vol_predicha_anual'] * 100, color='red', 
                      linestyle='--', linewidth=2, label=f"Vol. Predicha: {garch_result['vol_predicha_anual']:.1%}")
            
            # Zonas de volatilidad
            ax.axhspan(0, 15, alpha=0.1, color='green', label='Baja (<15%)')
            ax.axhspan(15, 30, alpha=0.1, color='yellow')
            ax.axhspan(30, 100, alpha=0.1, color='red')
            
            ax.set_title(f'{ticker} - Volatilidad Histórica y Predicha (GARCH)')
            ax.set_ylabel('Volatilidad Anualizada (%)')
            ax.set_ylim(0, min(vol_realizada.max() * 100 * 1.5, 100))
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Volatilidad**")
            vol_pred = garch_result['vol_predicha_anual']
            if vol_pred < 0.15:
                st.success(f"🟢 Baja: {vol_pred:.1%}")
            elif vol_pred < 0.30:
                st.warning(f"🟡 Media: {vol_pred:.1%}")
            else:
                st.error(f"🔴 Alta: {vol_pred:.1%}")
            
            st.markdown("**Comparación**")
            st.write(f"- Histórica: {garch_result['vol_historica_anual']:.1%}")
            st.write(f"- Predicha: {garch_result['vol_predicha_anual']:.1%}")
            cambio = garch_result['cambio_vol_pct']
            st.write(f"- Cambio: {cambio:+.1f}%")
            
            st.markdown("**Parámetros GARCH**")
            st.write(f"- Persistencia: {garch_result['persistencia']:.3f}")
            st.write(f"- VaR 95%: {garch_result['var_95']:.2%}")
    
    if not HMM_AVAILABLE and not GARCH_AVAILABLE:
        st.info("💡 Instala `hmmlearn` y `arch` para habilitar análisis avanzado de regímenes.")
    
    st.markdown("---")
    
    # --- GRÁFICO TÉCNICO CON BOLLINGER ---
    st.markdown("### 📉 Análisis Técnico: Bollinger(30) + RSI(13) + MACD")
    
    if not hist_largo.empty:
        close = hist_largo['Close']
        
        # Calcular Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(close, period=30, std_dev=2)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # --- GRÁFICO DE PRECIO CON BOLLINGER ---
        ax1 = axes[0]
        ax1.plot(close.index, close, 'b-', linewidth=1.5, label='Precio')
        
        # Bandas de Bollinger
        ax1.plot(bb_upper.index, bb_upper, 'r--', linewidth=1, label='BB Superior', alpha=0.7)
        ax1.plot(bb_middle.index, bb_middle, 'g-', linewidth=1, label='BB Media (30)', alpha=0.7)
        ax1.plot(bb_lower.index, bb_lower, 'r--', linewidth=1, label='BB Inferior', alpha=0.7)
        ax1.fill_between(bb_upper.index, bb_lower, bb_upper, alpha=0.1, color='blue')
        
        # Medias móviles
        if len(close) >= 200:
            ma200 = close.rolling(window=200).mean()
            ax1.plot(close.index, ma200, 'purple', linewidth=1.5, label='SMA200', alpha=0.8)
        
        # Marcar precio actual
        ax1.scatter(close.index[-1], close.iloc[-1], color='blue', s=100, zorder=5)
        
        # Colorear zonas extremas
        for i in range(len(close)):
            if close.iloc[i] <= bb_lower.iloc[i]:
                ax1.scatter(close.index[i], close.iloc[i], color='green', s=20, alpha=0.5, zorder=4)
            elif close.iloc[i] >= bb_upper.iloc[i]:
                ax1.scatter(close.index[i], close.iloc[i], color='red', s=20, alpha=0.5, zorder=4)
        
        ax1.set_title(f'{ticker} - Bandas de Bollinger(30, 2)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Precio')
        
        # --- RSI(13) ---
        ax2 = axes[1]
        rsi = calcular_rsi(close, period=13)
        ax2.plot(rsi.index, rsi, 'purple', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa (30)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax2.fill_between(rsi.index, rsi, 70, where=(rsi >= 70), alpha=0.3, color='red')
        ax2.fill_between(rsi.index, rsi, 30, where=(rsi <= 30), alpha=0.3, color='green')
        ax2.set_ylim(0, 100)
        ax2.set_title('RSI(13)')
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper left', fontsize=8)
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
        
        **Score Fundamental (40% del total si HMM/GARCH activo, 57% si no)**
        - PER < 15: empresa "barata" respecto a beneficios
        - EV/EBITDA < 10: buena valoración considerando deuda
        - P/BV < 1.5: cotiza cerca de su valor contable
        - ROE > 15%: empresa eficiente generando beneficios
        - Deuda/Equity < 100%: endeudamiento controlado
        - Dividendo > 2%: retribución atractiva
        
        **Score Técnico (30% del total si HMM/GARCH activo, 43% si no)**
        - Precio > MA50 y MA200: tendencia alcista
        - Golden Cross: MA50 cruza por encima de MA200 (señal alcista)
        - RSI 30-50: recuperándose de zona de sobreventa
        - MACD positivo: impulso alcista
        - Volumen creciente: confirma movimientos
        
        **Score Régimen - HMM + GARCH (30% del total)**
        
        *Hidden Markov Model (HMM):*
        - Detecta 3 regímenes ocultos: Alcista, Lateral, Bajista
        - Analiza patrones en retornos para identificar cambios de tendencia
        - Calcula probabilidad de transición entre regímenes
        
        *GARCH (Volatilidad):*
        - Predice volatilidad futura basándose en clusters históricos
        - Volatilidad < 15% anual = Baja (positivo)
        - Volatilidad > 30% anual = Alta (negativo)
        - Persistencia alta = volatilidad tiende a mantenerse
        
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
# MODO SEÑALES DE TRADING
# ==================================================
elif modo == "📊 Señales de Trading":
    st.title("📊 Señales de Trading - Retorno a la Media")
    st.markdown("""
    **Estrategia**: Detecta movimientos extremos del precio y señales de entrada/salida 
    usando RSI(13), Bandas de Bollinger(30) y patrones de velas de rechazo.
    """)
    
    if not TICKERS or not TICKERS[0]:
        st.error("Introduce un ticker para analizar.")
        st.stop()
    
    ticker = TICKERS[0]
    
    try:
        with st.spinner(f"Analizando {ticker}..."):
            data_accion = obtener_info_accion(ticker, "2y")  # 2 años para tener SMA200
        
        if data_accion is None or data_accion['history'].empty:
            st.error(f"No se pudieron obtener datos para {ticker}.")
            st.stop()
        
        info = data_accion['info']
        hist = data_accion['history']
        
    except Exception as e:
        st.error(f"Error obteniendo datos: {e}")
        st.stop()
    
    # Información básica
    nombre = info.get('longName', info.get('shortName', ticker))
    precio_actual = info.get('currentPrice', info.get('regularMarketPrice', hist['Close'].iloc[-1]))
    moneda = info.get('currency', 'USD')
    
    st.markdown(f"## {nombre}")
    st.markdown(f"**Precio actual:** {precio_actual:.2f} {moneda}")
    
    # Analizar señales
    señal, detalles = analizar_retorno_media(hist)
    
    if not detalles:
        st.warning("No hay suficientes datos para el análisis completo (se requieren 200+ días).")
        st.stop()
    
    # Mostrar señal principal
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if señal == 'LONG':
            st.success("### 🟢 SEÑAL LONG")
            st.markdown("**Compra recomendada**")
        elif señal == 'SHORT':
            st.error("### 🔴 SEÑAL SHORT")
            st.markdown("**Venta recomendada**")
        elif señal == 'VIGILAR_LONG':
            st.info("### 🔵 VIGILAR LONG")
            st.markdown("**Preparar compra**")
        elif señal == 'VIGILAR_SHORT':
            st.warning("### 🟠 VIGILAR SHORT")
            st.markdown("**Preparar venta**")
        else:
            st.info("### ⚪ SIN SEÑAL")
            st.markdown("**Esperar**")
    
    with col2:
        st.metric("RSI(13)", f"{detalles['rsi']:.1f}", 
                 delta="Sobreventa" if detalles['rsi'] <= 30 else "Sobrecompra" if detalles['rsi'] >= 70 else "Neutral")
        st.metric("Posición Bollinger", f"{detalles['bb_position']:.0f}%",
                 delta="Banda inferior" if detalles['bb_position'] <= 20 else "Banda superior" if detalles['bb_position'] >= 80 else "Centro")
    
    with col3:
        st.metric("Tendencia (vs SMA200)", detalles['tendencia'].upper())
        rechazo_texto = detalles['tipo_rechazo'] if detalles['tipo_rechazo'] else "Ninguno"
        st.metric("Vela Rechazo", rechazo_texto.capitalize())
    
    # Razones de la señal
    if detalles.get('razon'):
        st.markdown("### 📋 Razones de la señal")
        for r in detalles['razon']:
            st.markdown(f"- {r}")
    
    # Objetivos y Stops (si hay señal)
    if señal in ['LONG', 'SHORT']:
        st.markdown("---")
        st.markdown("### 🎯 Niveles de Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Entrada", f"{detalles['precio']:.2f} {moneda}")
        col2.metric("Objetivo (Banda Media)", f"{detalles['objetivo']:.2f} {moneda}",
                   delta=f"{((detalles['objetivo']/detalles['precio'])-1)*100:+.1f}%")
        col3.metric("Stop Loss", f"{detalles['stop']:.2f} {moneda}",
                   delta=f"{((detalles['stop']/detalles['precio'])-1)*100:+.1f}%")
        col4.metric("Ratio R/B", f"{detalles.get('riesgo_beneficio', 0):.2f}")
    
    # Gráfico con Bandas de Bollinger
    st.markdown("---")
    st.markdown("### 📈 Gráfico con Bandas de Bollinger y RSI")
    
    close = hist['Close']
    bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(close, period=30, std_dev=2)
    rsi_series = calcular_rsi(close, period=13)
    sma200 = close.rolling(window=200).mean()
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Subplot 1: Precio con Bandas de Bollinger
    ultimos_dias = min(180, len(close))  # Últimos 180 días
    
    ax1.plot(close.index[-ultimos_dias:], close.iloc[-ultimos_dias:], 'b-', linewidth=1.5, label='Precio')
    ax1.plot(bb_upper.index[-ultimos_dias:], bb_upper.iloc[-ultimos_dias:], 'r--', linewidth=1, label='Banda Superior', alpha=0.7)
    ax1.plot(bb_middle.index[-ultimos_dias:], bb_middle.iloc[-ultimos_dias:], 'g-', linewidth=1, label='Media (30)', alpha=0.7)
    ax1.plot(bb_lower.index[-ultimos_dias:], bb_lower.iloc[-ultimos_dias:], 'r--', linewidth=1, label='Banda Inferior', alpha=0.7)
    ax1.plot(sma200.index[-ultimos_dias:], sma200.iloc[-ultimos_dias:], 'purple', linewidth=1, label='SMA200', alpha=0.5)
    
    ax1.fill_between(bb_upper.index[-ultimos_dias:], bb_lower.iloc[-ultimos_dias:], bb_upper.iloc[-ultimos_dias:], 
                     alpha=0.1, color='blue')
    
    # Marcar precio actual
    ax1.scatter(close.index[-1], close.iloc[-1], color='blue', s=100, zorder=5)
    ax1.axhline(y=close.iloc[-1], color='blue', linestyle=':', alpha=0.5)
    
    ax1.set_title(f'{ticker} - Bandas de Bollinger(30, 2)', fontsize=14)
    ax1.set_ylabel('Precio')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: RSI
    ax2.plot(rsi_series.index[-ultimos_dias:], rsi_series.iloc[-ultimos_dias:], 'purple', linewidth=1.5, label='RSI(13)')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(rsi_series.index[-ultimos_dias:], 30, 70, alpha=0.1, color='gray')
    
    # Marcar RSI actual
    ax2.scatter(rsi_series.index[-1], rsi_series.iloc[-1], color='purple', s=100, zorder=5)
    
    ax2.set_title('RSI(13)', fontsize=12)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Tabla de indicadores
    st.markdown("---")
    st.markdown("### 📊 Detalle de Indicadores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bandas de Bollinger (30, 2)**")
        st.write(f"- Banda Superior: {detalles['bb_upper']:.2f}")
        st.write(f"- Media Central: {detalles['bb_middle']:.2f}")
        st.write(f"- Banda Inferior: {detalles['bb_lower']:.2f}")
        st.write(f"- Posición actual: {detalles['bb_position']:.1f}%")
    
    with col2:
        st.markdown("**Tendencia y Momentum**")
        st.write(f"- RSI(13): {detalles['rsi']:.1f}")
        st.write(f"- SMA200: {detalles['sma200']:.2f}")
        st.write(f"- Tendencia: {detalles['tendencia'].capitalize()}")
        if detalles['tipo_rechazo']:
            st.write(f"- Vela de rechazo: {detalles['tipo_rechazo'].capitalize()}")
    
    # Condiciones de la estrategia
    st.markdown("---")
    with st.expander("📖 Condiciones de la Estrategia"):
        st.markdown("""
        **SEÑAL LONG (Compra):**
        1. ✅ Precio toca o cruza la banda inferior de Bollinger(30)
        2. ✅ RSI(13) ≤ 30 (sobreventa)
        3. ✅ Vela de rechazo alcista (martillo)
        4. ✅ Tendencia principal NO bajista
        
        **SEÑAL SHORT (Venta):**
        1. ✅ Precio toca o cruza la banda superior de Bollinger(30)
        2. ✅ RSI(13) ≥ 70 (sobrecompra)
        3. ✅ Vela de rechazo bajista (estrella fugaz)
        4. ✅ Tendencia principal NO alcista
        
        **Gestión de la operación:**
        - **Objetivo**: Banda media de Bollinger
        - **Stop Loss**: 2% por debajo/encima del mínimo/máximo de la vela de rechazo
        - **Riesgo máximo**: 1% del capital por operación
        """)

# ==================================================
# MODO ANÁLISIS POR REGIÓN
# ==================================================
elif modo == "🌍 Análisis por Región":
    st.title("🌍 Análisis por Región Geográfica")
    
    if not regiones_seleccionadas:
        st.warning("Selecciona al menos una región en el panel lateral.")
        st.stop()
    
    # Función para analizar una acción
    @st.cache_data(ttl=7200, show_spinner=False)
    def analizar_accion_rapido(ticker):
        """Analiza una acción y devuelve métricas resumidas."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                return None
            
            # Info básica
            try:
                info = stock.info
                nombre = info.get('shortName') or info.get('longName') or ticker
                precio = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
                currency = info.get('currency', 'USD')
            except:
                info = {}
                nombre = ticker
                precio = hist['Close'].iloc[-1]
                currency = 'USD'
            
            # Calcular scores
            s_fund, _ = score_fundamental(info)
            s_tech, _ = score_tecnico(hist)
            
            # HMM y GARCH
            returns = hist['Close'].pct_change().dropna()
            hmm_res = detectar_regimenes_hmm(returns) if HMM_AVAILABLE else None
            garch_res = predecir_volatilidad_garch(returns) if GARCH_AVAILABLE else None
            
            if hmm_res or garch_res:
                s_regimen, _ = score_regimen_combinado(hmm_res, garch_res)
                score_total = s_fund * 0.4 + s_tech * 0.3 + s_regimen * 0.3
            else:
                s_regimen = None
                score_total = s_fund * 0.57 + s_tech * 0.43
            
            # Régimen HMM
            if hmm_res:
                regimen = hmm_res['estado_actual']
                prob_alcista = hmm_res['prob_alcista']
            else:
                regimen = 'N/A'
                prob_alcista = 0
            
            # Volatilidad GARCH
            if garch_res:
                vol_garch = garch_res['vol_predicha_anual']
            else:
                vol_garch = returns.std() * np.sqrt(252)
            
            # Recomendación
            if score_total >= 80:
                recomendacion = "🟢 COMPRA FUERTE"
            elif score_total >= 65:
                recomendacion = "🟢 COMPRA"
            elif score_total >= 50:
                recomendacion = "🟡 MANTENER"
            elif score_total >= 35:
                recomendacion = "🔴 VENTA"
            else:
                recomendacion = "🔴 VENTA FUERTE"
            
            return {
                'ticker': ticker,
                'nombre': nombre[:25] + '...' if len(nombre) > 25 else nombre,
                'precio': precio,
                'currency': currency,
                'score_total': score_total,
                'score_fund': s_fund,
                'score_tech': s_tech,
                'score_regimen': s_regimen,
                'regimen': regimen,
                'prob_alcista': prob_alcista,
                'vol_garch': vol_garch,
                'recomendacion': recomendacion
            }
        except Exception as e:
            return None
    
    # Analizar todas las acciones seleccionadas
    resultados_por_region = {}
    todos_resultados = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_acciones = sum(len(REGIONES[r]) for r in regiones_seleccionadas)
    acciones_procesadas = 0
    
    for region in regiones_seleccionadas:
        tickers_region = REGIONES[region]
        resultados_region = []
        
        for ticker in tickers_region:
            status_text.text(f"Analizando {ticker}...")
            resultado = analizar_accion_rapido(ticker)
            
            if resultado:
                resultado['region'] = region
                resultados_region.append(resultado)
                todos_resultados.append(resultado)
            
            acciones_procesadas += 1
            progress_bar.progress(acciones_procesadas / total_acciones)
        
        resultados_por_region[region] = resultados_region
    
    progress_bar.empty()
    status_text.empty()
    
    if not todos_resultados:
        st.error("No se pudieron obtener datos. Puede ser un problema temporal con Yahoo Finance.")
        st.stop()
    
    # --- TOP 10 GLOBAL ---
    st.markdown("## 🏆 Top 10 Global - Mejores Oportunidades")
    
    top_global = sorted(todos_resultados, key=lambda x: x['score_total'], reverse=True)[:10]
    
    top_df = pd.DataFrame([{
        'Rank': i+1,
        'Ticker': r['ticker'],
        'Nombre': r['nombre'],
        'Región': r['region'].split()[0],  # Solo emoji
        'Score': f"{r['score_total']:.0f}",
        'Régimen': f"{'🟢' if r['regimen']=='alcista' else '🟡' if r['regimen']=='lateral' else '🔴' if r['regimen']=='bajista' else '⚪'} {r['regimen'].capitalize() if r['regimen'] != 'N/A' else 'N/A'}",
        'Vol.': f"{r['vol_garch']:.0%}",
        'Recomendación': r['recomendacion']
    } for i, r in enumerate(top_global)])
    
    st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    # --- FILTROS ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filtro_regimen = st.selectbox(
            "Filtrar por régimen",
            ["Todos", "🟢 Solo Alcistas", "🟡 Solo Laterales", "🔴 Solo Bajistas"]
        )
    
    with col2:
        filtro_recom = st.selectbox(
            "Filtrar por recomendación",
            ["Todas", "Solo COMPRA FUERTE", "Solo COMPRA", "Solo MANTENER", "Solo VENTA"]
        )
    
    with col3:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["Score Total", "Score Fundamental", "Score Técnico", "Menor Volatilidad", "Mayor Prob. Alcista"]
        )
    
    # Aplicar filtros
    resultados_filtrados = todos_resultados.copy()
    
    if filtro_regimen == "🟢 Solo Alcistas":
        resultados_filtrados = [r for r in resultados_filtrados if r['regimen'] == 'alcista']
    elif filtro_regimen == "🟡 Solo Laterales":
        resultados_filtrados = [r for r in resultados_filtrados if r['regimen'] == 'lateral']
    elif filtro_regimen == "🔴 Solo Bajistas":
        resultados_filtrados = [r for r in resultados_filtrados if r['regimen'] == 'bajista']
    
    if "COMPRA FUERTE" in filtro_recom:
        resultados_filtrados = [r for r in resultados_filtrados if "COMPRA FUERTE" in r['recomendacion']]
    elif "Solo COMPRA" in filtro_recom:
        resultados_filtrados = [r for r in resultados_filtrados if "COMPRA" in r['recomendacion']]
    elif "MANTENER" in filtro_recom:
        resultados_filtrados = [r for r in resultados_filtrados if "MANTENER" in r['recomendacion']]
    elif "VENTA" in filtro_recom:
        resultados_filtrados = [r for r in resultados_filtrados if "VENTA" in r['recomendacion']]
    
    # Ordenar
    if ordenar_por == "Score Total":
        resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score_total'], reverse=True)
    elif ordenar_por == "Score Fundamental":
        resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score_fund'], reverse=True)
    elif ordenar_por == "Score Técnico":
        resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['score_tech'], reverse=True)
    elif ordenar_por == "Menor Volatilidad":
        resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['vol_garch'])
    elif ordenar_por == "Mayor Prob. Alcista":
        resultados_filtrados = sorted(resultados_filtrados, key=lambda x: x['prob_alcista'], reverse=True)
    
    # --- RESULTADOS POR REGIÓN ---
    st.markdown("---")
    st.markdown("## 📊 Resultados por Región")
    
    for region in regiones_seleccionadas:
        resultados_region = [r for r in resultados_filtrados if r['region'] == region]
        
        if not resultados_region:
            continue
        
        with st.expander(f"{region} ({len(resultados_region)} acciones)", expanded=True):
            # Métricas resumen de la región
            avg_score = np.mean([r['score_total'] for r in resultados_region])
            alcistas = sum(1 for r in resultados_region if r['regimen'] == 'alcista')
            bajistas = sum(1 for r in resultados_region if r['regimen'] == 'bajista')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Score Medio", f"{avg_score:.0f}/100")
            col2.metric("Acciones Alcistas", f"{alcistas}/{len(resultados_region)}")
            col3.metric("Acciones Bajistas", f"{bajistas}/{len(resultados_region)}")
            col4.metric("Top Score", f"{max(r['score_total'] for r in resultados_region):.0f}")
            
            # Tabla de la región
            region_df = pd.DataFrame([{
                'Ticker': r['ticker'],
                'Nombre': r['nombre'],
                'Precio': f"{r['precio']:.2f} {r['currency']}",
                'Score Total': f"{r['score_total']:.0f}",
                'Fund.': f"{r['score_fund']}",
                'Téc.': f"{r['score_tech']}",
                'Rég.': f"{r['score_regimen']:.0f}" if r['score_regimen'] else 'N/A',
                'Régimen': f"{'🟢' if r['regimen']=='alcista' else '🟡' if r['regimen']=='lateral' else '🔴' if r['regimen']=='bajista' else '⚪'}",
                'Vol.': f"{r['vol_garch']:.0%}",
                'Recomendación': r['recomendacion']
            } for r in resultados_region])
            
            st.dataframe(region_df, use_container_width=True, hide_index=True)
    
    # --- COMPARATIVA ENTRE REGIONES ---
    st.markdown("---")
    st.markdown("## 📈 Comparativa entre Regiones")
    
    comparativa_data = []
    for region in regiones_seleccionadas:
        resultados_region = resultados_por_region.get(region, [])
        if resultados_region:
            comparativa_data.append({
                'Región': region,
                'Acciones': len(resultados_region),
                'Score Medio': np.mean([r['score_total'] for r in resultados_region]),
                'Alcistas': sum(1 for r in resultados_region if r['regimen'] == 'alcista'),
                'Vol. Media': np.mean([r['vol_garch'] for r in resultados_region]),
                'Mejor Acción': max(resultados_region, key=lambda x: x['score_total'])['ticker']
            })
    
    if comparativa_data:
        comparativa_df = pd.DataFrame(comparativa_data)
        comparativa_df = comparativa_df.sort_values('Score Medio', ascending=False)
        comparativa_df['Score Medio'] = comparativa_df['Score Medio'].apply(lambda x: f"{x:.1f}")
        comparativa_df['Vol. Media'] = comparativa_df['Vol. Media'].apply(lambda x: f"{x:.0%}")
        comparativa_df['% Alcistas'] = comparativa_df.apply(lambda x: f"{x['Alcistas']/x['Acciones']*100:.0f}%", axis=1)
        
        st.dataframe(comparativa_df[['Región', 'Acciones', 'Score Medio', '% Alcistas', 'Vol. Media', 'Mejor Acción']], 
                    use_container_width=True, hide_index=True)
        
        # Gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(10, 5))
        regiones_nombres = [r['Región'].split('(')[0].strip() for r in comparativa_data]
        scores = [float(r['Score Medio']) if isinstance(r['Score Medio'], str) else r['Score Medio'] for r in comparativa_data]
        
        # Ordenar por score
        sorted_indices = np.argsort(scores)[::-1]
        regiones_nombres = [regiones_nombres[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        colors = ['#00ff88' if s >= 65 else '#ffaa00' if s >= 50 else '#ff4444' for s in scores]
        bars = ax.barh(regiones_nombres, scores, color=colors)
        
        ax.set_xlabel('Score Medio')
        ax.set_title('Comparativa de Regiones por Score Medio')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, score in zip(bars, scores):
            ax.text(score + 2, bar.get_y() + bar.get_height()/2, f'{score:.0f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)

# ==================================================
# MODO COMPARADOR DE ACTIVOS
# ==================================================
elif modo == "📈 Comparador de Activos":
    st.title("📈 Comparador de Activos")
    st.markdown("Compara rentabilidades de **ETFs**, **Bonos** y calcula tu cartera ideal según tu perfil de riesgo.")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏦 ETFs por Categoría",
        "📊 Bonos y Renta Fija", 
        "⚖️ Calculadora de Cartera",
        "📉 Comparador Visual"
    ])
    
    # TAB 1: ETFs por Categoría
    with tab1:
        st.subheader("🏦 ETFs por Categoría")
        
        # Selector de categorías
        categorias_seleccionadas = st.multiselect(
            "Selecciona categorías",
            list(ETFs_POR_CATEGORIA.keys()),
            default=["📈 Renta Variable USA", "🏦 Renta Fija USA", "🥇 Materias Primas"]
        )
        
        if not categorias_seleccionadas:
            st.warning("Selecciona al menos una categoría.")
        else:
            # Selector de periodo
            periodo_etf = st.selectbox(
                "Periodo de rentabilidad",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3,
                format_func=lambda x: {"1mo": "1 mes", "3mo": "3 meses", "6mo": "6 meses", 
                                       "1y": "1 año", "2y": "2 años", "5y": "5 años"}[x]
            )
            
            # Obtener datos de ETFs
            todos_etfs = []
            
            progress = st.progress(0)
            total = sum(len(ETFs_POR_CATEGORIA[cat]) for cat in categorias_seleccionadas)
            procesados = 0
            
            for categoria in categorias_seleccionadas:
                for ticker, nombre in ETFs_POR_CATEGORIA[categoria].items():
                    datos = obtener_rentabilidad_etf(ticker, periodo_etf)
                    if datos:
                        datos['categoria_grupo'] = categoria
                        datos['nombre_corto'] = nombre
                        todos_etfs.append(datos)
                    procesados += 1
                    progress.progress(procesados / total)
            
            progress.empty()
            
            if todos_etfs:
                # Crear DataFrame
                df_etfs = pd.DataFrame(todos_etfs)
                
                # Ordenar por rentabilidad
                df_etfs = df_etfs.sort_values('rentabilidad', ascending=False)
                
                # Mostrar métricas resumen
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ETFs Analizados", len(df_etfs))
                col2.metric("Mejor Rentabilidad", f"{df_etfs['rentabilidad'].max():.1f}%")
                col3.metric("Peor Rentabilidad", f"{df_etfs['rentabilidad'].min():.1f}%")
                col4.metric("Rentabilidad Media", f"{df_etfs['rentabilidad'].mean():.1f}%")
                
                st.markdown("---")
                
                # Top 10 mejores
                st.markdown("### 🏆 Top 10 Mejores Rentabilidades")
                top_10 = df_etfs.head(10)
                
                top_df = pd.DataFrame({
                    'Rank': range(1, len(top_10) + 1),
                    'Ticker': top_10['ticker'],
                    'Nombre': top_10['nombre_corto'],
                    'Categoría': [c.split()[0] for c in top_10['categoria_grupo']],
                    'Rentabilidad': [f"{r:+.1f}%" for r in top_10['rentabilidad']],
                    'Volatilidad': [f"{v:.1f}%" for v in top_10['volatilidad']],
                    'Precio': [f"${p:.2f}" for p in top_10['precio']]
                })
                st.dataframe(top_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Por categoría
                st.markdown("### 📊 Detalle por Categoría")
                
                for categoria in categorias_seleccionadas:
                    etfs_cat = df_etfs[df_etfs['categoria_grupo'] == categoria].copy()
                    
                    if not etfs_cat.empty:
                        with st.expander(f"{categoria} ({len(etfs_cat)} ETFs)", expanded=True):
                            # Métricas de la categoría
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Rent. Media", f"{etfs_cat['rentabilidad'].mean():+.1f}%")
                            col2.metric("Vol. Media", f"{etfs_cat['volatilidad'].mean():.1f}%")
                            mejor = etfs_cat.iloc[0]
                            col3.metric("Mejor", f"{mejor['ticker']} ({mejor['rentabilidad']:+.1f}%)")
                            
                            # Tabla
                            cat_df = pd.DataFrame({
                                'Ticker': etfs_cat['ticker'],
                                'Nombre': etfs_cat['nombre_corto'],
                                'Rentabilidad': [f"{r:+.1f}%" for r in etfs_cat['rentabilidad']],
                                'Volatilidad': [f"{v:.1f}%" for v in etfs_cat['volatilidad']],
                                'Precio': [f"${p:.2f}" for p in etfs_cat['precio']]
                            })
                            st.dataframe(cat_df, use_container_width=True, hide_index=True)
                
                # Gráfico comparativo
                st.markdown("---")
                st.markdown("### 📈 Gráfico Comparativo")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                colores = {'📈': 'blue', '🇪🇺': 'green', '🌏': 'orange', '📊': 'purple',
                          '🏦': 'red', '🏛️': 'brown', '🌍': 'teal', '🥇': 'gold', '🏠': 'pink'}
                
                for categoria in categorias_seleccionadas:
                    etfs_cat = df_etfs[df_etfs['categoria_grupo'] == categoria]
                    emoji = categoria.split()[0]
                    color = colores.get(emoji, 'gray')
                    ax.scatter(etfs_cat['volatilidad'], etfs_cat['rentabilidad'], 
                              label=categoria, s=100, alpha=0.7, c=color)
                    
                    for _, row in etfs_cat.iterrows():
                        ax.annotate(row['ticker'], (row['volatilidad'], row['rentabilidad']),
                                   fontsize=8, alpha=0.7)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Volatilidad (%)')
                ax.set_ylabel('Rentabilidad (%)')
                ax.set_title('Rentabilidad vs Volatilidad por Categoría')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # TAB 2: Bonos y Renta Fija
    with tab2:
        st.subheader("📊 Bonos y Renta Fija")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🇪🇸 Bonos España (Tesoro Público)")
            
            bonos_esp = obtener_tipos_bonos_espana()
            
            bonos_esp_df = pd.DataFrame([
                {
                    'Instrumento': nombre,
                    'Tipo Interés': f"{datos['tipo']:.2f}%",
                    'Variación': f"{datos['cambio']:+.2f}%",
                    'Estado': '🟢' if datos['cambio'] < 0 else '🔴' if datos['cambio'] > 0 else '🟡'
                }
                for nombre, datos in bonos_esp.items()
            ])
            st.dataframe(bonos_esp_df, use_container_width=True, hide_index=True)
            
            st.caption("💡 Tipos bajando = Precios de bonos subiendo")
            st.caption("📅 Datos orientativos. Consultar Tesoro Público para valores oficiales.")
        
        with col2:
            st.markdown("### 🇺🇸 Bonos USA (Treasury)")
            
            with st.spinner("Obteniendo datos de bonos USA..."):
                bonos_usa = obtener_tipos_bonos_usa()
            
            if bonos_usa:
                bonos_usa_df = pd.DataFrame([
                    {
                        'Instrumento': nombre,
                        'Tipo Interés': f"{datos['tipo']:.2f}%",
                        'Variación': f"{datos['cambio']:+.2f}%",
                        'Estado': '🟢' if datos['cambio'] < 0 else '🔴' if datos['cambio'] > 0 else '🟡'
                    }
                    for nombre, datos in bonos_usa.items()
                ])
                st.dataframe(bonos_usa_df, use_container_width=True, hide_index=True)
            else:
                st.info("No se pudieron obtener datos de bonos USA.")
        
        st.markdown("---")
        
        # Curva de tipos
        st.markdown("### 📉 Curva de Tipos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Curva España
            bonos_esp = obtener_tipos_bonos_espana()
            plazos_esp = [0.25, 0.5, 1, 3, 5, 10, 15, 30]
            tipos_esp = [bonos_esp[k]['tipo'] for k in bonos_esp.keys()]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(plazos_esp, tipos_esp, 'b-o', linewidth=2, markersize=8)
            ax.fill_between(plazos_esp, tipos_esp, alpha=0.3)
            ax.set_xlabel('Plazo (años)')
            ax.set_ylabel('Tipo de Interés (%)')
            ax.set_title('🇪🇸 Curva de Tipos España')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 32)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Curva USA
            if bonos_usa:
                plazos_usa = [0.25, 2, 5, 10, 30]
                tipos_usa = [bonos_usa[k]['tipo'] for k in bonos_usa.keys()]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(plazos_usa[:len(tipos_usa)], tipos_usa, 'r-o', linewidth=2, markersize=8)
                ax.fill_between(plazos_usa[:len(tipos_usa)], tipos_usa, alpha=0.3, color='red')
                ax.set_xlabel('Plazo (años)')
                ax.set_ylabel('Tipo de Interés (%)')
                ax.set_title('🇺🇸 Curva de Tipos USA')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 32)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Comparativa ETFs de renta fija
        st.markdown("---")
        st.markdown("### 🏦 ETFs de Renta Fija - Rentabilidad")
        
        etfs_rf = ["BND", "AGG", "TLT", "LQD", "HYG", "SHY", "IEF"]
        
        with st.spinner("Obteniendo datos de ETFs de renta fija..."):
            datos_rf = []
            for ticker in etfs_rf:
                datos = obtener_rentabilidad_etf(ticker, "1y")
                if datos:
                    datos_rf.append(datos)
        
        if datos_rf:
            df_rf = pd.DataFrame(datos_rf).sort_values('rentabilidad', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['green' if r > 0 else 'red' for r in df_rf['rentabilidad']]
            bars = ax.barh(df_rf['ticker'], df_rf['rentabilidad'], color=colors)
            ax.set_xlabel('Rentabilidad 1 año (%)')
            ax.set_title('Rentabilidad ETFs Renta Fija (1 año)')
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            
            for bar, rent in zip(bars, df_rf['rentabilidad']):
                ax.text(rent + 0.2 if rent >= 0 else rent - 0.5, bar.get_y() + bar.get_height()/2,
                       f'{rent:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # TAB 3: Calculadora de Cartera
    with tab3:
        st.subheader("⚖️ Calculadora de Cartera por Perfil de Riesgo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Tu Perfil")
            
            perfil = st.radio(
                "Selecciona tu perfil de riesgo",
                list(PERFILES_RIESGO.keys()),
                index=1
            )
            
            inversion = st.number_input(
                "Inversión total (€)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            st.markdown("---")
            st.markdown(f"**{perfil}**")
            st.write(PERFILES_RIESGO[perfil]['descripcion'])
        
        with col2:
            st.markdown("### Distribución Recomendada")
            
            cartera = calcular_cartera_por_perfil(perfil, inversion)
            
            # Gráfico circular
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            labels = list(cartera['distribucion'].keys())
            sizes = [d['porcentaje'] for d in cartera['distribucion'].values()]
            colors_pie = ['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
            
            ax1.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors_pie[:len(labels)],
                   explode=[0.02] * len(labels))
            ax1.set_title(f'Distribución - Perfil {perfil}')
            
            # Barras con importes
            importes = [d['importe'] for d in cartera['distribucion'].values()]
            ax2.barh(labels, importes, color=colors_pie[:len(labels)])
            ax2.set_xlabel('Importe (€)')
            ax2.set_title('Distribución por Importe')
            
            for i, (label, importe) in enumerate(zip(labels, importes)):
                ax2.text(importe + 100, i, f'{importe:,.0f}€', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla de distribución
            st.markdown("### 💶 Detalle de Inversión")
            
            dist_df = pd.DataFrame([
                {
                    'Categoría': cat,
                    'Porcentaje': f"{datos['porcentaje']}%",
                    'Importe': f"{datos['importe']:,.2f}€"
                }
                for cat, datos in cartera['distribucion'].items()
            ])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            # ETFs sugeridos
            st.markdown("### 💡 ETFs Sugeridos")
            
            etfs_sugeridos = cartera['etfs_sugeridos']
            
            with st.spinner("Obteniendo datos de ETFs sugeridos..."):
                datos_sugeridos = []
                for ticker in etfs_sugeridos:
                    datos = obtener_rentabilidad_etf(ticker, "1y")
                    if datos:
                        datos_sugeridos.append(datos)
            
            if datos_sugeridos:
                sug_df = pd.DataFrame([
                    {
                        'Ticker': d['ticker'],
                        'Nombre': d['nombre'][:30],
                        'Rent. 1Y': f"{d['rentabilidad']:+.1f}%",
                        'Volatilidad': f"{d['volatilidad']:.1f}%",
                        'Precio': f"${d['precio']:.2f}"
                    }
                    for d in datos_sugeridos
                ])
                st.dataframe(sug_df, use_container_width=True, hide_index=True)
    
    # TAB 4: Comparador Visual
    with tab4:
        st.subheader("📉 Comparador Visual de Activos")
        
        st.markdown("Compara la evolución de hasta 5 activos.")
        
        # Selector de activos
        col1, col2 = st.columns([2, 1])
        
        with col1:
            activos_comparar = st.multiselect(
                "Selecciona activos para comparar",
                ["SPY", "QQQ", "VTI", "EEM", "BND", "TLT", "GLD", "VGK", "EWP", "AGG", 
                 "IWM", "LQD", "HYG", "SLV", "VNQ", "XLK", "XLF", "XLE"],
                default=["SPY", "BND", "GLD"]
            )
        
        with col2:
            periodo_comp = st.selectbox(
                "Periodo",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3,
                format_func=lambda x: {"1mo": "1 mes", "3mo": "3 meses", "6mo": "6 meses", 
                                       "1y": "1 año", "2y": "2 años", "5y": "5 años"}[x],
                key="periodo_comparador"
            )
        
        if len(activos_comparar) < 2:
            st.warning("Selecciona al menos 2 activos para comparar.")
        elif len(activos_comparar) > 5:
            st.warning("Máximo 5 activos para una comparación clara.")
        else:
            with st.spinner("Descargando datos..."):
                # Descargar datos
                data = yf.download(activos_comparar, period=periodo_comp, progress=False)['Close']
            
            if not data.empty:
                # Normalizar a 100
                data_norm = data / data.iloc[0] * 100
                
                # Gráfico de evolución normalizada
                st.markdown("### 📈 Evolución Normalizada (Base 100)")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for ticker in activos_comparar:
                    if ticker in data_norm.columns:
                        ax.plot(data_norm.index, data_norm[ticker], linewidth=2, label=ticker)
                
                ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Valor (Base 100)')
                ax.set_title('Comparación de Rentabilidad Normalizada')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Métricas comparativas
                st.markdown("### 📊 Métricas Comparativas")
                
                metricas = []
                for ticker in activos_comparar:
                    if ticker in data.columns:
                        precios = data[ticker].dropna()
                        rent = (precios.iloc[-1] / precios.iloc[0] - 1) * 100
                        vol = precios.pct_change().std() * np.sqrt(252) * 100
                        sharpe = rent / vol if vol > 0 else 0
                        max_dd = ((precios / precios.cummax()) - 1).min() * 100
                        
                        metricas.append({
                            'Ticker': ticker,
                            'Rentabilidad': f"{rent:+.1f}%",
                            'Volatilidad': f"{vol:.1f}%",
                            'Sharpe': f"{sharpe:.2f}",
                            'Max Drawdown': f"{max_dd:.1f}%"
                        })
                
                metricas_df = pd.DataFrame(metricas)
                st.dataframe(metricas_df, use_container_width=True, hide_index=True)
                
                # Matriz de correlación
                st.markdown("### 🔗 Matriz de Correlación")
                
                returns = data.pct_change().dropna()
                corr_matrix = returns.corr()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
                
                ax.set_xticks(range(len(activos_comparar)))
                ax.set_yticks(range(len(activos_comparar)))
                ax.set_xticklabels(activos_comparar)
                ax.set_yticklabels(activos_comparar)
                
                for i in range(len(activos_comparar)):
                    for j in range(len(activos_comparar)):
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=10)
                
                plt.colorbar(im, ax=ax, label='Correlación')
                ax.set_title('Correlación entre Activos')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("💡 Correlación baja o negativa = Mayor diversificación")

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

    # Calcular regímenes HMM/GARCH para cada activo (si disponible)
    regimenes_cartera = {}
    if HMM_AVAILABLE or GARCH_AVAILABLE:
        with st.spinner("Analizando regímenes de mercado para cada activo..."):
            for ticker in TICKERS:
                returns = prices[ticker].pct_change().dropna()
                hmm_res = detectar_regimenes_hmm(returns) if HMM_AVAILABLE else None
                garch_res = predecir_volatilidad_garch(returns) if GARCH_AVAILABLE else None
                regimenes_cartera[ticker] = {
                    'hmm': hmm_res,
                    'garch': garch_res
                }

    # TABS
    if HMM_AVAILABLE or GARCH_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Cartera Óptima", 
            "🎲 Simulación Monte Carlo", 
            "🔮 Regímenes HMM/GARCH",
            "⚖️ Rebalanceo", 
            "📉 Frontera Eficiente"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Cartera Óptima", 
            "🎲 Simulación Monte Carlo", 
            "⚖️ Rebalanceo", 
            "📉 Frontera Eficiente"
        ])
        tab5 = None

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

    # TAB 3: REGÍMENES HMM/GARCH
    if tab5 is not None:  # Solo si HMM/GARCH está disponible
        with tab3:
            st.subheader("Análisis de Regímenes por Activo")
            
            if not regimenes_cartera:
                st.info("Instala `hmmlearn` y `arch` para habilitar este análisis.")
            else:
                # Tabla resumen de regímenes
                st.markdown("#### 📊 Resumen de Regímenes")
                
                resumen_data = []
                for ticker in TICKERS:
                    hmm_res = regimenes_cartera[ticker]['hmm']
                    garch_res = regimenes_cartera[ticker]['garch']
                    
                    # Estado HMM
                    if hmm_res:
                        estado_hmm = hmm_res['estado_actual']
                        prob_alcista = hmm_res['prob_alcista']
                        emoji_hmm = '🟢' if estado_hmm == 'alcista' else '🟡' if estado_hmm == 'lateral' else '🔴'
                    else:
                        estado_hmm = 'N/A'
                        prob_alcista = 0
                        emoji_hmm = '⚪'
                    
                    # Volatilidad GARCH
                    if garch_res:
                        vol_pred = garch_res['vol_predicha_anual']
                        cambio_vol = garch_res['cambio_vol_pct']
                        emoji_vol = '🟢' if vol_pred < 0.20 else '🟡' if vol_pred < 0.35 else '🔴'
                    else:
                        vol_pred = 0
                        cambio_vol = 0
                        emoji_vol = '⚪'
                    
                    # Score combinado
                    if hmm_res or garch_res:
                        s_reg, _ = score_regimen_combinado(hmm_res, garch_res)
                    else:
                        s_reg = 50
                    
                    resumen_data.append({
                        'Activo': ticker,
                        'Régimen HMM': f"{emoji_hmm} {estado_hmm.capitalize() if estado_hmm != 'N/A' else 'N/A'}",
                        'Prob. Alcista': f"{prob_alcista:.0%}" if hmm_res else 'N/A',
                        'Vol. GARCH': f"{emoji_vol} {vol_pred:.1%}" if garch_res else 'N/A',
                        'Δ Volatilidad': f"{cambio_vol:+.0f}%" if garch_res else 'N/A',
                        'Score Régimen': f"{s_reg:.0f}/100"
                    })
                
                resumen_df = pd.DataFrame(resumen_data)
                st.dataframe(resumen_df, use_container_width=True, hide_index=True)
                
                # Recomendación basada en regímenes
                st.markdown("#### 💡 Recomendación de Cartera basada en Regímenes")
                
                scores_regimen = []
                for ticker in TICKERS:
                    hmm_res = regimenes_cartera[ticker]['hmm']
                    garch_res = regimenes_cartera[ticker]['garch']
                    s_reg, _ = score_regimen_combinado(hmm_res, garch_res)
                    scores_regimen.append(s_reg)
                
                # Calcular pesos ajustados por régimen
                scores_array = np.array(scores_regimen)
                if scores_array.sum() > 0:
                    pesos_regimen = scores_array / scores_array.sum()
                else:
                    pesos_regimen = np.ones(len(TICKERS)) / len(TICKERS)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pesos sugeridos (ajustados por régimen)**")
                    ajuste_df = pd.DataFrame({
                        'Activo': TICKERS,
                        'Peso Sharpe': [f"{w:.1%}" for w in weights],
                        'Peso por Régimen': [f"{w:.1%}" for w in pesos_regimen],
                        'Score Régimen': [f"{s:.0f}" for s in scores_regimen]
                    })
                    st.dataframe(ajuste_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Gráfico comparativo
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(TICKERS))
                    width = 0.35
                    ax.bar(x - width/2, weights * 100, width, label='Sharpe Óptimo', color='steelblue')
                    ax.bar(x + width/2, pesos_regimen * 100, width, label='Ajustado Régimen', color='coral')
                    ax.set_ylabel('Peso (%)')
                    ax.set_title('Pesos Óptimos vs Ajustados por Régimen')
                    ax.set_xticks(x)
                    ax.set_xticklabels(TICKERS, rotation=45)
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Detalle por activo
                st.markdown("#### 📈 Detalle por Activo")
                
                ticker_detalle = st.selectbox("Selecciona activo para ver detalle", TICKERS)
                
                hmm_res = regimenes_cartera[ticker_detalle]['hmm']
                garch_res = regimenes_cartera[ticker_detalle]['garch']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if hmm_res:
                        st.markdown(f"**HMM - {ticker_detalle}**")
                        
                        # Gráfico de regímenes
                        fig, ax = plt.subplots(figsize=(10, 4))
                        close = prices[ticker_detalle]
                        estados = hmm_res['historial_estados']
                        
                        colores = {'alcista': '#00ff88', 'lateral': '#ffaa00', 'bajista': '#ff4444'}
                        
                        for i in range(min(len(estados), len(close))):
                            if i < len(close) - 1:
                                ax.axvspan(close.index[i], close.index[i+1], 
                                          alpha=0.3, color=colores.get(estados[i], 'gray'), linewidth=0)
                        
                        ax.plot(close.index[-len(estados):], close.iloc[-len(estados):], 'b-', linewidth=1)
                        ax.set_title(f'{ticker_detalle} - Regímenes HMM')
                        ax.set_ylabel('Precio')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        st.write(f"Estado actual: **{hmm_res['estado_actual'].upper()}**")
                        st.write(f"Prob. Alcista: {hmm_res['prob_alcista']:.1%}")
                        st.write(f"Prob. Bajista: {hmm_res['prob_bajista']:.1%}")
                    else:
                        st.info("HMM no disponible para este activo")
                
                with col2:
                    if garch_res:
                        st.markdown(f"**GARCH - {ticker_detalle}**")
                        
                        # Gráfico de volatilidad
                        fig, ax = plt.subplots(figsize=(10, 4))
                        returns = prices[ticker_detalle].pct_change().dropna()
                        vol_realizada = returns.rolling(window=22).std() * np.sqrt(252)
                        
                        ax.plot(vol_realizada.index, vol_realizada * 100, 'b-', linewidth=1, label='Vol. Realizada')
                        ax.axhline(y=garch_res['vol_predicha_anual'] * 100, color='red', 
                                  linestyle='--', linewidth=2, label=f"Predicha: {garch_res['vol_predicha_anual']:.1%}")
                        ax.set_title(f'{ticker_detalle} - Volatilidad GARCH')
                        ax.set_ylabel('Volatilidad (%)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        st.write(f"Vol. Histórica: {garch_res['vol_historica_anual']:.1%}")
                        st.write(f"Vol. Predicha: {garch_res['vol_predicha_anual']:.1%}")
                        st.write(f"Persistencia: {garch_res['persistencia']:.3f}")
                    else:
                        st.info("GARCH no disponible para este activo")

    # TAB 4: REBALANCEO
    with tab4 if tab5 is not None else tab3:
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

    # TAB 5: FRONTERA EFICIENTE
    with tab5 if tab5 is not None else tab4:
        st.subheader("Frontera Eficiente")
        
        with st.spinner("Calculando frontera eficiente..."):
            frontier = efficient_frontier(prices, rf, n_points=100, max_weight=max_weight)
        
        if frontier.empty or 'Vol' not in frontier.columns or 'Return' not in frontier.columns:
            st.warning("No se pudo calcular la frontera eficiente con los parámetros actuales. Prueba a ajustar el peso máximo por activo.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(frontier['Vol'] * 100, frontier['Return'] * 100, 'b-', linewidth=2, label='Frontera Eficiente')
            ax.scatter(best['Vol'] * 100, best['Return'] * 100, marker='*', s=300, c='red', label=f'Cartera Óptima (Sharpe={best["Sharpe"]:.2f})')
            
            log_returns, mu, cov = compute_statistics(prices)
            for ticker in TICKERS:
                ax.scatter(np.sqrt(cov.loc[ticker, ticker]) * 100, mu[ticker] * 100, marker='o', s=100, label=ticker)
            
            sharpe_opt = best['Sharpe']
            if not frontier['Vol'].empty:
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
