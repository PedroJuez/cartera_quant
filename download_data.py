"""
Script para descargar datos históricos de precios de activos.

Uso:
    python download_data.py
    python download_data.py --tickers AAPL GOOGL MSFT --start 2019-01-01
"""

import yfinance as yf
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta

# Configuración por defecto
DEFAULT_TICKERS = ["AAPL", "MSFT", "BNP.PA","NVO"]
DEFAULT_START = "2020-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "data"


def download_prices(tickers, start, end, output_path):
    """
    Descarga precios ajustados de los activos especificados.
    
    Parameters:
    -----------
    tickers : list
        Lista de símbolos de activos
    start : str
        Fecha de inicio (YYYY-MM-DD)
    end : str
        Fecha de fin (YYYY-MM-DD)
    output_path : str
        Ruta donde guardar el CSV
    
    Returns:
    --------
    DataFrame con los precios descargados
    """
    print(f"📥 Descargando datos para: {', '.join(tickers)}")
    print(f"   Período: {start} → {end}")
    
    try:
        # Descargar datos
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=True,
            auto_adjust=True  # Usa precios ajustados por dividendos y splits
        )
        
        # Manejar caso de un solo ticker
        if len(tickers) == 1:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            prices = data['Close']
        
        # Verificar datos descargados
        if prices.empty:
            raise ValueError("No se descargaron datos. Verifica los tickers.")
        
        # Estadísticas de los datos
        print(f"\n📊 Resumen de datos descargados:")
        print(f"   Observaciones: {len(prices)}")
        print(f"   Primer registro: {prices.index[0].strftime('%Y-%m-%d')}")
        print(f"   Último registro: {prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Valores faltantes por activo:")
        
        for ticker in tickers:
            missing = prices[ticker].isna().sum()
            pct = missing / len(prices) * 100
            status = "✅" if missing == 0 else "⚠️"
            print(f"      {status} {ticker}: {missing} ({pct:.1f}%)")
        
        # Guardar
        prices.to_csv(output_path)
        print(f"\n✅ Datos guardados en: {output_path}")
        
        return prices
        
    except Exception as e:
        print(f"\n❌ Error descargando datos: {e}")
        raise


def validate_tickers(tickers):
    """Valida que los tickers existan en Yahoo Finance."""
    print("\n🔍 Validando tickers...")
    
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info.get('regularMarketPrice') is not None:
                valid_tickers.append(ticker)
                print(f"   ✅ {ticker}: {info.get('shortName', 'N/A')}")
            else:
                invalid_tickers.append(ticker)
                print(f"   ❌ {ticker}: No encontrado")
        except Exception:
            invalid_tickers.append(ticker)
            print(f"   ❌ {ticker}: Error de validación")
    
    if invalid_tickers:
        print(f"\n⚠️ Tickers inválidos ignorados: {', '.join(invalid_tickers)}")
    
    return valid_tickers


def download_additional_data(tickers, output_dir):
    """Descarga información adicional de los activos (fundamentales, etc.)."""
    
    info_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            info_data.append({
                'ticker': ticker,
                'name': info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency'),
                'exchange': info.get('exchange'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta')
            })
        except Exception as e:
            print(f"   ⚠️ No se pudo obtener info de {ticker}: {e}")
    
    if info_data:
        info_df = pd.DataFrame(info_data)
        info_path = os.path.join(output_dir, "tickers_info.csv")
        info_df.to_csv(info_path, index=False)
        print(f"✅ Información de activos guardada en: {info_path}")
        return info_df
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Descarga datos históricos de precios de Yahoo Finance"
    )
    
    parser.add_argument(
        '--tickers', '-t',
        nargs='+',
        default=DEFAULT_TICKERS,
        help=f'Lista de tickers (default: {DEFAULT_TICKERS})'
    )
    
    parser.add_argument(
        '--start', '-s',
        default=DEFAULT_START,
        help=f'Fecha de inicio YYYY-MM-DD (default: {DEFAULT_START})'
    )
    
    parser.add_argument(
        '--end', '-e',
        default=DEFAULT_END,
        help=f'Fecha de fin YYYY-MM-DD (default: {DEFAULT_END})'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=os.path.join(DATA_DIR, "prices.csv"),
        help='Ruta del archivo de salida'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validar tickers antes de descargar'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Descargar información adicional de los activos'
    )
    
    args = parser.parse_args()
    
    # Crear directorio si no existe
    os.makedirs(DATA_DIR, exist_ok=True)
    
    tickers = args.tickers
    
    # Validar tickers si se solicita
    if args.validate:
        tickers = validate_tickers(tickers)
        if not tickers:
            print("❌ No hay tickers válidos. Abortando.")
            return
    
    # Descargar precios
    prices = download_prices(tickers, args.start, args.end, args.output)
    
    # Descargar info adicional si se solicita
    if args.info:
        print("\n📋 Descargando información adicional...")
        download_additional_data(tickers, DATA_DIR)
    
    # Mostrar estadísticas básicas
    print("\n📈 Rendimientos en el período:")
    for ticker in tickers:
        if ticker in prices.columns:
            ret = (prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1) * 100
            print(f"   {ticker}: {ret:+.1f}%")


if __name__ == "__main__":
    main()
