import yfinance as yf
import pandas as pd
import numpy as np
import io
import requests

def get_nse_tickers():
    # Use a high-quality list from a reliable source
    url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarketlist.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        # Create a session to bypass basic bot detection
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(response.text))
        return [f"{s}.NS" for s in df['Symbol'].dropna().unique().tolist() if s != 'Symbol']
    except Exception as e:
        print(f"NSE Fetch failed: {e}. Using fallback.")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]

def generate():
    tickers = get_nse_tickers()
    print(f"Scanning {len(tickers)} stocks...")
    
    # Batch fetch data
    data = yf.download(tickers, period="60d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            s = data[t].dropna()
            if len(s) < 25: continue
            
            # Momentum logic
            y = np.log(s['Close'].values)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            
            results.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(s['Close'].iloc[-1]), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(s['Volume'].iloc[-1] / s['Volume'].tail(20).mean()), 2)
            })
        except: continue
    
    pd.DataFrame(results).to_csv("daily_watchlist.csv", index=False)
    print("✅ daily_watchlist.csv generated successfully.")

if __name__ == "__main__":
    generate()
