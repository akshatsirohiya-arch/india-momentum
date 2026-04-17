import yfinance as yf
import pandas as pd
import numpy as np
import requests

def get_nifty_500_tickers():
    """Fetches the latest Nifty 500 list from NSE."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    # Convert 'SYMBOL' to 'SYMBOL.NS' for yfinance
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def calculate_slope(prices):
    """Calculates momentum slope using log-linear regression."""
    if len(prices) < 30: return 0
    y = np.log(prices)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def generate():
    tickers = get_nifty_500_tickers()
    print(f"Scanning {len(tickers)} stocks...")
    
    # Batch download last 90 days for all stocks (High Speed)
    data = yf.download(tickers, period="90d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            # Extract 'Close' and 'Volume' for this specific ticker
            s_data = data[t].dropna()
            if len(s_data) < 30: continue
            
            price = s_data['Close'].iloc[-1]
            slope = calculate_slope(s_data['Close'])
            # Relative Volume (last day vs 20-day avg)
            rvol = s_data['Volume'].iloc[-1] / s_data['Volume'].tail(20).mean()
            
            results.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(price), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(rvol), 2)
            })
        except: continue
    
    df = pd.DataFrame(results)
    df.to_csv("daily_watchlist.csv", index=False)
    print(f"Watchlist generated: {len(df)} stocks processed.")

if __name__ == "__main__":
    generate()
