import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------
# 1. Leer datos desde CSV
# --------------------------------------------------
def load_prices(path="data/prices.csv"):
    """Carga precios históricos desde CSV."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.dropna()


# --------------------------------------------------
# 2. Calcular retornos y estadísticas
# --------------------------------------------------
def compute_statistics(prices):
    """Calcula retornos logarítmicos, media y covarianza anualizados."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * 252
    cov = log_returns.cov() * 252
    return log_returns, mu, cov


# --------------------------------------------------
# 3. Cartera óptima (Sharpe) - Optimización analítica
# --------------------------------------------------
def optimal_portfolio(prices, rf=0.02, method="scipy"):
    """
    Encuentra la cartera con máximo Sharpe ratio.
    
    Parameters:
    -----------
    prices : DataFrame
        Precios históricos de los activos
    rf : float
        Tasa libre de riesgo anual
    method : str
        'scipy' para optimización analítica (recomendado)
        'montecarlo' para búsqueda aleatoria
    
    Returns:
    --------
    dict con keys: 'Return', 'Vol', 'Sharpe', 'Weights'
    """
    log_returns, mu, cov = compute_statistics(prices)
    n_assets = len(prices.columns)
    
    if method == "scipy":
        # Función objetivo: negativo del Sharpe (minimizamos)
        def neg_sharpe(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -(ret - rf) / vol
        
        # Restricciones: pesos suman 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Límites: pesos entre 0 y 1 (sin ventas en corto)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Punto inicial: equiponderado
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        
    else:  # Monte Carlo
        n_portfolios = 20000
        best_sharpe = -np.inf
        weights = None
        
        for _ in range(n_portfolios):
            w = np.random.random(n_assets)
            w /= np.sum(w)
            
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            sharpe = (ret - rf) / vol
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                weights = w
    
    # Calcular métricas finales
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf) / vol
    
    return {
        'Return': ret,
        'Vol': vol,
        'Sharpe': sharpe,
        'Weights': weights
    }


# --------------------------------------------------
# 4. Frontera eficiente
# --------------------------------------------------
def efficient_frontier(prices, rf=0.02, n_points=50):
    """Calcula la frontera eficiente."""
    log_returns, mu, cov = compute_statistics(prices)
    n_assets = len(prices.columns)
    
    # Rango de retornos objetivo
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier = []
    
    for target in target_returns:
        # Minimizar volatilidad para un retorno objetivo
        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_vol,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            vol = result.fun
            sharpe = (target - rf) / vol if vol > 0 else 0
            frontier.append({
                'Return': target,
                'Vol': vol,
                'Sharpe': sharpe,
                'Weights': result.x
            })
    
    return pd.DataFrame(frontier)


# --------------------------------------------------
# 5. Monte Carlo de precios futuros (CORREGIDO)
# --------------------------------------------------
def monte_carlo(prices, weights, investment, n_days=21, n_sim=5000):
    """
    Simula el valor futuro de la cartera usando GBM correlacionado.
    
    Parameters:
    -----------
    prices : DataFrame
        Precios históricos
    weights : array
        Pesos objetivo de la cartera
    investment : float
        Inversión total en euros
    n_days : int
        Días a simular
    n_sim : int
        Número de simulaciones
    
    Returns:
    --------
    dict con:
        - 'returns': array de retornos simulados
        - 'final_values': array de valores finales
        - 'paths': matriz de trayectorias (opcional, si n_sim <= 100)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Parámetros del modelo
    mu_d = log_returns.mean().values  # Media diaria
    cov_d = log_returns.cov().values  # Covarianza diaria
    
    # Descomposición de Cholesky para correlaciones
    L = np.linalg.cholesky(cov_d)
    
    # Precios iniciales y cantidades de acciones
    S0 = prices.iloc[-1].values
    
    # Calcular número de acciones que compramos con la inversión
    amounts_invested = investment * weights  # € invertidos en cada activo
    n_shares = amounts_invested / S0  # Número de acciones de cada activo
    
    V0 = investment  # Valor inicial de la cartera
    
    final_values = []
    paths = [] if n_sim <= 100 else None
    
    for _ in range(n_sim):
        prices_sim = S0.copy()
        path = [np.dot(prices_sim, n_shares)]
        
        for _ in range(n_days):
            # Generar shocks correlacionados
            Z = np.random.standard_normal(len(weights))
            correlated_Z = L @ Z
            
            # Actualizar precios con GBM
            prices_sim = prices_sim * np.exp(
                (mu_d - 0.5 * np.diag(cov_d)) + correlated_Z
            )
            
            if paths is not None:
                path.append(np.dot(prices_sim, n_shares))
        
        final_value = np.dot(prices_sim, n_shares)
        final_values.append(final_value)
        
        if paths is not None:
            paths.append(path)
    
    final_values = np.array(final_values)
    returns = (final_values - V0) / V0
    
    result = {
        'returns': returns,
        'final_values': final_values,
        'V0': V0
    }
    
    if paths is not None:
        result['paths'] = np.array(paths)
    
    return result


# --------------------------------------------------
# 6. Métricas de riesgo
# --------------------------------------------------
def risk_metrics(returns, confidence=0.95):
    """
    Calcula métricas de riesgo de la distribución de retornos.
    
    Returns:
    --------
    dict con VaR, CVaR, probabilidad de pérdida, percentiles
    """
    alpha = 1 - confidence
    
    # Value at Risk (VaR)
    var = np.percentile(returns, alpha * 100)
    
    # Conditional VaR (Expected Shortfall)
    cvar = returns[returns <= var].mean()
    
    return {
        'VaR': var,
        'CVaR': cvar,
        'prob_loss': (returns < 0).mean(),
        'percentile_5': np.percentile(returns, 5),
        'percentile_25': np.percentile(returns, 25),
        'percentile_50': np.percentile(returns, 50),
        'percentile_75': np.percentile(returns, 75),
        'percentile_95': np.percentile(returns, 95),
        'mean': returns.mean(),
        'std': returns.std()
    }


# --------------------------------------------------
# 7. Rebalanceo por desviación (CORREGIDO)
# --------------------------------------------------
def rebalance(prices, target_weights, n_shares, threshold=0.05):
    """
    Determina si es necesario rebalancear la cartera.
    
    Parameters:
    -----------
    prices : DataFrame
        Precios históricos (se usa el último precio)
    target_weights : array
        Pesos objetivo
    n_shares : array
        Número de acciones que tenemos de cada activo
    threshold : float
        Umbral de desviación para rebalancear
    
    Returns:
    --------
    tuple: (rebalance_needed, current_weights, deviations)
    """
    current_prices = prices.iloc[-1].values
    
    # Valor actual de cada posición
    position_values = current_prices * n_shares
    
    # Valor total de la cartera
    total_value = position_values.sum()
    
    # Pesos actuales
    current_weights = position_values / total_value
    
    # Desviaciones
    deviations = np.abs(current_weights - target_weights)
    
    # ¿Necesitamos rebalancear?
    rebalance_needed = (deviations > threshold).any()
    
    return rebalance_needed, current_weights, deviations


def calculate_rebalance_trades(current_prices, n_shares, target_weights, total_value=None):
    """
    Calcula las operaciones necesarias para rebalancear.
    
    Returns:
    --------
    DataFrame con las operaciones (compra/venta) necesarias
    """
    if total_value is None:
        total_value = (current_prices * n_shares).sum()
    
    target_values = total_value * target_weights
    current_values = current_prices * n_shares
    
    diff_values = target_values - current_values
    diff_shares = diff_values / current_prices
    
    return pd.DataFrame({
        'current_shares': n_shares,
        'target_shares': n_shares + diff_shares,
        'trade_shares': diff_shares,
        'trade_value': diff_values
    })
