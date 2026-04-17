import yfinance as yf
import pandas as pd
import numpy as np
import requests

def get_all_tickers():
    """Fetches 500+ liquid tickers from a reliable GitHub mirror to avoid NSE blocks."""
    url = "https://raw.githubusercontent.com/stock-data/india-tickers/master/nifty500.json"
    try:
        data = requests.get(url).json()
        return [f"{t}.NS" for t in data]
    except:
        # Emergency backup list if the mirror is down
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]

def is_clean_trend(s):
    """Filters for stocks making Higher Highs and Higher Lows over last 20 days."""
    recent = s.tail(20)
    hh = recent['High'].iloc[-1] > recent['High'].iloc[-10]
    hl = recent['Low'].iloc[-1] > recent['Low'].iloc[-10]
    return hh and hl

def generate():
    tickers = get_all_tickers()
    print(f"Scanning {len(tickers)} stocks for Clean Momentum...")
    
    # Vectorized Download (Fastest way to hit the whole market)
    data = yf.download(tickers, period="100d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            s = data[t].dropna()
            if len(s) < 50: continue
            
            # 1. Slope Calculation (Institutional Momentum)
            y = np.log(s['Close'].values)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            
            # 2. Clean Trend Filter (HH/HL)
            if not is_clean_trend(s): continue
            
            price = s['Close'].iloc[-1]
            if price < 20: continue # Ignore junk penny stocks

            results.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(price), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(s['Volume'].iloc[-1] / s['Volume'].tail(20).mean()), 2)
            })
        except: continue
    
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("daily_watchlist.csv", index=False)
        print(f"✅ Success! {len(df)} High-Momentum stocks found.")
    else:
        print("❌ No stocks met the HH/HL criteria today.")

if __name__ == "__main__":
    generate()
