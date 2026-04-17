import yfinance as yf
import pandas as pd
import numpy as np
import io
import requests

# --- CONFIGURATION ---
MARKET_CAP_FILTER = 500 * 1e7  # Example: 500 Crore minimum (INR)

def get_full_nse_universe():
    """Fetches every listed stock on the NSE using the official Bhavcopy data."""
    url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarketlist.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    # Returns all tickers with the .NS suffix
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def calculate_slope(prices):
    if len(prices) < 30: return 0
    y = np.log(prices)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def generate():
    tickers = get_full_nse_universe()
    print(f"Scanning entire NSE market: {len(tickers)} symbols...")
    
    # BATCH DOWNLOAD: Fetching last 90 days for ALL stocks at once
    data = yf.download(tickers, period="90d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            s_data = data[t].dropna()
            if len(s_data) < 40: continue # Ensure enough history
            
            # 1. Momentum Logic
            price = s_data['Close'].iloc[-1]
            slope = calculate_slope(s_data['Close'])
            
            # 2. RVOL (Relative Volume)
            rvol = s_data['Volume'].iloc[-1] / s_data['Volume'].tail(20).mean()
            
            # 3. Market Cap Filtering (Optional)
            # info = yf.Ticker(t).info # Slow! Better to filter by Nifty Total Market indices
            
            results.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(price), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(rvol), 2)
            })
        except: continue
    
    df = pd.DataFrame(results)
    df.to_csv("daily_watchlist.csv", index=False)
    print(f"Success! {len(df)} stocks written to daily_watchlist.csv")

if __name__ == "__main__":
    generate()
