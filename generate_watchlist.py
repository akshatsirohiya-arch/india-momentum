import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time

def get_tickers():
    """Fetches Nifty 500 to ensure we have a liquid, scan-able universe."""
    url = "https://raw.githubusercontent.com/stock-data/india-tickers/master/nifty500.json"
    try:
        return [f"{t}.NS" for t in requests.get(url).json()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

def check_structure(df):
    """
    Strict Higher High / Higher Low Logic:
    1. Current Price > 20-day SMA
    2. Higher High: Max of last 5 days > Max of previous 5-10 days
    3. Higher Low: Min of last 5 days > Min of previous 5-10 days
    """
    if len(df) < 40: return False
    
    recent_high = df['High'].tail(5).max()
    prev_high = df['High'].iloc[-10:-5].max()
    
    recent_low = df['Low'].tail(5).min()
    prev_low = df['Low'].iloc[-10:-5].min()
    
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    return (recent_high > prev_high) and (recent_low > prev_low) and (current_price > sma20)

def generate():
    tickers = get_tickers()
    print(f"Scanning {len(tickers)} stocks for Clean Breakouts...")
    
    # Process in smaller batches to avoid SSL 'Bad Record Mac' errors
    batch_size = 50
    all_results = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="100d", interval="1d", group_by='ticker', progress=False)
            
            for t in batch:
                try:
                    s = data[t].dropna()
                    if not check_structure(s): continue
                    
                    # Log-linear slope for smooth institutional trend
                    y = np.log(s['Close'].values)
                    x = np.arange(len(y))
                    slope, _ = np.polyfit(x, y, 1)
                    
                    if slope > 0.001: # Only positive momentum
                        all_results.append({
                            "Ticker": t.replace(".NS", ""),
                            "Price": round(float(s['Close'].iloc[-1]), 2),
                            "Slope": round(float(slope), 6),
                            "RVOL": round(float(s['Volume'].iloc[-1] / s['Volume'].tail(20).mean()), 2)
                        })
                except: continue
            time.sleep(1) # Small pause to prevent rate-limiting
        except: continue

    df = pd.DataFrame(all_results)
    df.to_csv("daily_watchlist.csv", index=False)
    print(f"✅ Success: {len(df)} stocks found with clean momentum structures.")

if __name__ == "__main__":
    generate()
