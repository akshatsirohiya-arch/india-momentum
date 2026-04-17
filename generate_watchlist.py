import yfinance as yf
import pandas as pd
import numpy as np
import io
import requests

def get_nse_tickers():
    """Fetches tickers from NSE with a fallback to a GitHub backup if blocked."""
    urls = [
        "https://archives.nseindia.com/content/indices/ind_niftytotalmarketlist.csv",
        "https://raw.githubusercontent.com/datasets/nse-stocks/master/data/nifty500.csv" # Backup
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for url in urls:
        try:
            print(f"Attempting to fetch tickers from: {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                # Handle different column names between NSE and Backup
                col = 'Symbol' if 'Symbol' in df.columns else 'symbol'
                tickers = [f"{s}.NS" for s in df[col].dropna().unique().tolist()]
                if len(tickers) > 10:
                    print(f"✅ Success: Found {len(tickers)} tickers.")
                    return tickers
        except Exception as e:
            print(f"Skipping source due to error: {e}")
            continue
    
    # Emergency hardcoded list if all else fails
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS"]

def generate():
    tickers = get_nse_tickers()
    # Batch download last 60 days of data
    print("Downloading market data from Yahoo Finance...")
    data = yf.download(tickers, period="60d", interval="1d", group_by='ticker', progress=False)
    
    final_data = []
    for t in tickers:
        try:
            s = data[t].dropna()
            if len(s) < 20: continue
            
            # Momentum Calculation (Slope of Log Prices)
            y = np.log(s['Close'].values)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            
            # RVOL
            rvol = s['Volume'].iloc[-1] / s['Volume'].tail(20).mean()
            
            final_data.append({
                "Ticker": t.replace(".NS", ""),
                "Price": round(float(s['Close'].iloc[-1]), 2),
                "Slope": round(float(slope), 6),
                "RVOL": round(float(rvol), 2)
            })
        except: continue
    
    output_df = pd.DataFrame(final_data)
    if not output_df.empty:
        output_df.to_csv("daily_watchlist.csv", index=False)
        print(f"🚀 DONE: daily_watchlist.csv generated with {len(output_df)} stocks.")
    else:
        print("❌ FAILED: No data collected.")

if __name__ == "__main__":
    generate()
