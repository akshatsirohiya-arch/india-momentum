import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io

def get_tickers():
    """Fetches the Nifty 500 or Total Market list."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(response.text))
        tickers = [f"{s}.NS" for s in df['Symbol'].tolist()]
        print(f"✅ Fetched {len(tickers)} tickers from NSE.")
        return tickers
    except Exception as e:
        print(f"⚠️ Fetch failed ({e}). using hardcoded Nifty 50 list.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
                "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS",
                "LT.NS", "BAJFINANCE.NS", "ADANIENT.NS", "SUNPHARMA.NS", "TITAN.NS"]

def check_momentum(df):
    """
    Refined Institutional Logic:
    1. Slope of log prices > 0 (Upward trend)
    2. Higher Highs: Current 5-day high > Previous 5-day high
    3. Higher Lows: Current 5-day low > Previous 5-day low
    """
    if len(df) < 30: return False
    
    # Structure Check
    curr_high = df['High'].tail(5).max()
    prev_high = df['High'].iloc[-10:-5].max()
    curr_low = df['Low'].tail(5).min()
    prev_low = df['Low'].iloc[-10:-5].min()
    
    # Slope Check
    y = np.log(df['Close'].values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    
    # Return True only if it passes both
    return (curr_high > prev_high) and (curr_low > prev_low) and (slope > 0.0005)

def generate():
    tickers = get_tickers()
    print("Downloading data...")
    # Grouping into batches of 100 to avoid SSL/Timeout issues
    batch_size = 100
    all_results = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data = yf.download(batch, period="60d", interval="1d", group_by='ticker', progress=False)
        
        for t in batch:
            try:
                s = data[t].dropna()
                if check_momentum(s):
                    slope = np.polyfit(np.arange(len(s)), np.log(s['Close'].values), 1)[0]
                    all_results.append({
                        "Ticker": t.replace(".NS", ""),
                        "Price": round(float(s['Close'].iloc[-1]), 2),
                        "Slope": round(float(slope), 6),
                        "RVOL": round(float(s['Volume'].iloc[-1] / s['Volume'].tail(20).mean()), 2)
                    })
            except: continue

    df = pd.DataFrame(all_results)
    df.to_csv("daily_watchlist.csv", index=False)
    print(f"🚀 SUCCESS: Found {len(df)} momentum stocks out of {len(tickers)} scanned.")

if __name__ == "__main__":
    generate()
