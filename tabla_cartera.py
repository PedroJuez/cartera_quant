import yfinance as yf

tickers = ["AAPL", "MSFT", "BNP.PA"]

data = yf.download(
    tickers,
    start="2020-01-01",
    end="2026-01-01",
    interval="1d",
    progress=False
)

# Miramos la columna "Adj Close" (precio ajustado)
prices = data["Adj Close"].dropna()
print(prices.head())
