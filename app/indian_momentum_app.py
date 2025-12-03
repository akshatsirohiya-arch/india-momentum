# indian_momentum_app_dynamic_universe.py
# TA-only (Slope + 1M + 3M returns), NO FUNDAMENTALS
# Universe: All NSE EQ stocks with market cap >= X Cr (default 1000 Cr)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import time

st.set_page_config(layout="wide", page_title="Momentum Scanner India (Auto Universe)")

# ----------------------------------
# CONSTANTS
# ----------------------------------

NSE_SEC_LIST_URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

# ----------------------------------
# UTILITIES
# ----------------------------------

def compute_slope(close_series: pd.Series) -> float:
    y = close_series.values
    x = np.arange(len(y))
    if len(y) < 5:
        return np.nan
    try:
        return np.polyfit(x, y, 1)[0]
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def fetch_prices(tickers, period="6mo", interval="1d"):
    try:
        return yf.download(tickers, period=period, interval=interval, progress=False)
    except Exception:
        return None

@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_nse_equity_symbols() -> pd.DataFrame:
    """
    Download NSE securities list and return EQ series symbols.
    """
    resp = requests.get(NSE_SEC_LIST_URL, headers=NSE_HEADERS, timeout=30)
    resp.raise_for_status()
    buf = io.StringIO(resp.text)
    df = pd.read_csv(buf)

    # Keep only EQ series (normal equities)
    df = df[df["Series"] == "EQ"].copy()
    df["YahooTicker"] = df["Symbol"].str.strip() + ".NS"
    return df[["Symbol", "Security Name", "YahooTicker"]]

def get_market_cap_yf(yahoo_ticker: str):
    """
    Get market cap from yfinance (in local currency, for .NS this is INR).
    Tries fast_info first, then falls back to info.
    """
    try:
        t = yf.Ticker(yahoo_ticker)
        mc = None
        # fast_info is quicker / lighter
        fast = getattr(t, "fast_info", None)
        if fast is not None:
            mc = fast.get("market_cap", None)

        if mc is None:
            info = t.info or {}
            mc = info.get("marketCap", None)

        return mc
    except Exception:
        return None

@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def build_universe(min_mcap_cr: float = 1000.0) -> pd.DataFrame:
    """
    Build universe of all NSE EQ stocks with market cap >= min_mcap_cr.
    Market cap is taken from yfinance (in INR).
    1 Cr = 1e7, so threshold (in INR) = min_mcap_cr * 1e7.
    """
    base = load_nse_equity_symbols()
    tickers = base["YahooTicker"].tolist()

    rows = []
    threshold_inr = min_mcap_cr * 1e7

    # Simple loop with basic throttling; you can optimize further if needed
    for i, yt in enumerate(tickers, start=1):
        mc = get_market_cap_yf(yt)
        if mc is None:
            continue

        if mc >= threshold_inr:
            rows.append({
                "YahooTicker": yt,
                "MarketCapINR": mc,
            })

        # very gentle sleep to avoid hammering
        time.sleep(0.05)

    uni = pd.DataFrame(rows)
    if uni.empty:
        return uni

    uni = uni.merge(base, on="YahooTicker", how="left")
    return uni.sort_values("MarketCapINR", ascending=False).reset_index(drop=True)

# ----------------------------------
# PROCESS BATCH
# ----------------------------------

def process_batch(tickers):
    data = fetch_prices(tickers)
    if data is None or "Close" not in data:
        return pd.DataFrame()

    close_df = data["Close"]
    rows = []

    for tk in close_df.columns:
        s = close_df[tk].dropna()
        if len(s) < 10:
            continue

        slope = compute_slope(s[-60:])
        ret_1m = (s.iloc[-1] / s.iloc[-21] - 1) * 100 if len(s) >= 22 else np.nan
        ret_3m = (s.iloc[-1] / s.iloc[-63] - 1) * 100 if len(s) >= 64 else np.nan

        rows.append({
            "Ticker": tk,
            "Close": s.iloc[-1],
            "Slope": slope,
            "Ret1M": ret_1m,
            "Ret3M": ret_3m,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if df["Slope"].notna().any():
        mn, mx = df["Slope"].min(), df["Slope"].max()
        df["SlopeNorm"] = (df["Slope"] - mn) / (mx - mn + 1e-9)
    else:
        df["SlopeNorm"] = 0.0

    df["FinalScore"] = (
        0.7 * df["SlopeNorm"].fillna(0) +
        0.15 * df["Ret1M"].fillna(0) +
        0.15 * df["Ret3M"].fillna(0)
    )

    return df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

# ----------------------------------
# STREAMLIT UI
# ----------------------------------

st.title("Momentum Scanner India — Auto Universe (No Fundamentals)")
st.write("Universe: NSE EQ stocks with Market Cap ≥ chosen threshold (Cr)")

# Sidebar controls
min_mcap_cr = st.sidebar.number_input(
    "Min Market Cap (₹ Cr)",
    min_value=100.0,
    max_value=100000.0,
    value=1000.0,
    step=100.0,
)

batch_size = st.sidebar.slider("Batch Size", 20, 200, 80, step=20)

if st.sidebar.button("Refresh Ticker Universe"):
    build_universe.clear()   # clear cached universe
    fetch_prices.clear()     # optionally clear price cache too
    st.experimental_rerun()

if st.sidebar.button("Clear All Cache (Prices + Universe)"):
    st.cache_data.clear()
    st.experimental_rerun()

# Load universe
with st.spinner("Building universe from NSE + yfinance..."):
    uni_df = build_universe(min_mcap_cr=min_mcap_cr)

if uni_df.empty:
    st.error("Universe is empty. NSE or yfinance may be blocking/limiting requests.")
    st.stop()

st.success(f"Universe size: {len(uni_df)} stocks with Market Cap ≥ {min_mcap_cr:.0f} Cr")

# You can inspect universe if needed
with st.expander("Show universe (symbols & market cap)"):
    st.dataframe(uni_df[["YahooTicker", "Symbol", "Security Name", "MarketCapINR"]])

# Now scan these tickers
tickers_list = uni_df["YahooTicker"].tolist()

final_df_list = []
for i in range(0, len(tickers_list), batch_size):
    batch = tickers_list[i: i + batch_size]
    st.write(f"Processing {len(batch)} tickers ({i+1}–{i+len(batch)} of {len(tickers_list)})...")
    out = process_batch(batch)
    if not out.empty:
        final_df_list.append(out)

if final_df_list:
    final = (
        pd.concat(final_df_list)
        .sort_values("FinalScore", ascending=False)
        .reset_index(drop=True)
    )

    # Keep your <20% 1M rule, or change as you like
    final = final[final["Ret1M"] < 20]

    st.subheader("Top NSE Momentum Picks (<20% 1M return)")
    st.dataframe(final.head(250))
else:
    st.error("No results. Check universe or yfinance status.")
