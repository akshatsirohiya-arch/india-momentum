# indian_momentum_app.py
# EXACT CLONE OF US APP — ONLY TICKERS CHANGED
# TA-only (Slope + 1M + 3M returns), NO FUNDAMENTALS

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide", page_title="Momentum Scanner India (No Fundamentals)")

# ----------------------------------
# LOAD TICKERS (INDIA / NSE)
# ----------------------------------
TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
    "LTIM.NS","LT.NS","PERSISTENT.NS","COFORGE.NS","KPITTECH.NS","LTI.NS",
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS",
    "IDFCFIRSTB.NS","BANDHANBNK.NS","INDUSINDBK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","HDFCLIFE.NS","ICICIPRULI.NS",
    "TITAN.NS","ASIANPAINT.NS","HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS",
    "DMART.NS","TRENT.NS",
    "MARUTI.NS","M&M.NS","TATAMOTORS.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "TVSMOTOR.NS","ASHOKLEY.NS",
    "ULTRACEMCO.NS","AMBUJACEM.NS","SHREECEM.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","VEDL.NS",
    "TATAPOWER.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIENSOL.NS",
    "NTPC.NS","POWERGRID.NS","NHPC.NS","SJVN.NS","IRFC.NS","IREDA.NS",
    "SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","DIVISLAB.NS","LUPIN.NS",
    "APOLLOHOSP.NS","FORTIS.NS",
    "ZOMATO.NS","PAYTM.NS","MAPMYINDIA.NS","INDIAMART.NS","NAUKRI.NS",
    "DIXON.NS","HAVELLS.NS","VOLTAS.NS","POLYCAB.NS","KEI.NS",
    "IRCTC.NS","BHEL.NS","BEL.NS","COCHINSHIP.NS","HAL.NS",
    "RVNL.NS","GRSE.NS","GAIL.NS",
    "DEEPAKNTR.NS","AARTIIND.NS","BALAMINES.NS","GNFC.NS",
]

# ----------------------------------
# UTILITIES
# ----------------------------------

def compute_slope(close_series):
    y = close_series.values
    x = np.arange(len(y))
    if len(y) < 5:
        return np.nan
    try:
        return np.polyfit(x, y, 1)[0]
    except:
        return np.nan

@st.cache_data(show_spinner=False)
def fetch_prices(tickers, period="6mo", interval="1d"):
    try:
        return yf.download(tickers, period=period, interval=interval, progress=False)
    except Exception:
        return None

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
        df["SlopeNorm"] = 0

    df["FinalScore"] = (
        0.7 * df["SlopeNorm"].fillna(0) +
        0.15 * df["Ret1M"].fillna(0) +
        0.15 * df["Ret3M"].fillna(0)
    )

    return df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

# ----------------------------------
# STREAMLIT UI
# ----------------------------------

st.title("Momentum Scanner India — Clean Version (No Fundamentals)")
st.write("Fast TA-only ranking for NSE large + mid + small caps")

batch_size = st.sidebar.slider("Batch Size", 20, 200, 80, step=20)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.experimental_rerun()

final_df = []
for i in range(0, len(TICKERS), batch_size):
    batch = TICKERS[i: i + batch_size]
    st.write(f"Processing {len(batch)} tickers...")
    out = process_batch(batch)
    if not out.empty:
        final_df.append(out)

if final_df:
    final = (
        pd.concat(final_df)
        .sort_values("FinalScore", ascending=False)
        .reset_index(drop=True)
    )

    final = final[final["Ret1M"] < 20]

    st.subheader("Top NSE Momentum Picks (<20% 1M return)")
    st.dataframe(final.head(250))

else:
    st.error("No results. Check tickers or yfinance status.")
