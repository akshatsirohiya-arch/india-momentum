import yfinance as yf
import pandas as pd
import numpy as np
import io
import requests
import time

def get_full_nse_universe():
    """Fetches the Nifty Total Market list (covers 750+ stocks) with browser headers."""
    # This URL covers Large, Mid, Small, and Microcap segments
    url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarketlist.csv"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/csv'
    }

    try:
        # Create a session to handle cookies/headers like a real user
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10) # Hit home page first to get cookies
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            tickers = [f"{s}.NS" for s in df['Symbol'].dropna().unique().tolist()]
            print(f"✅ Successfully loaded {len(tickers)} stocks from NSE.")
            return tickers
        else:
            print(f"⚠️ NSE returned status {response.status_code}. Using fallback.")
    except Exception as e:
        print(f"❌ Error fetching universe: {e}")

    # ULTIMATE FALLBACK: Core Momentum Stocks (If NSE Website is down)
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ADANIENT.NS", "TITAN.NS"]

def calculate_slope(prices):
    if len(prices) < 30: return 0
    y = np.log(prices)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def generate():
    tickers = get_full_nse_universe()
    print("Starting data fetch from Yahoo Finance...")
    
    # Batch download (Vectorized) is much faster for 'entire market' scans
    data = yf.download(tickers, period="90d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for t in tickers:
        try:
            # Handle yfinance multi-index format
            s_data = data[t].dropna()
            if len(s_data) < 35: continue
            
            price = s_data['Close'].iloc[-1]
            slope = calculate_slope(s_data['Close'])
            rvol = s_data['Volume'].iloc[-1] / s_data['Volume'].tail(20).mean()
            
            # Simple Filter: Ignore penny stocks under ₹10 to keep the data high-quality
            if price < 10: continue

            results.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(price), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(rvol), 2)
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    df.to_csv("daily_watchlist.csv", index=False)
    print(f"🔥 Successfully scanned {len(df)} stocks. Watchlist updated.")

if __name__ == "__main__":
    generate()
