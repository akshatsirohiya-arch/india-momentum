import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="India Market Live", layout="wide")

st.sidebar.title("NSE Explorer")
ticker = st.sidebar.text_input("Enter NSE Ticker", "RELIANCE").upper()
symbol = f"{ticker}.NS"

# Sidebar period selection
period = st.sidebar.selectbox("Period", ["1mo", "6mo", "1y", "5y"])

try:
    df = yf.download(symbol, period=period)
    
    if not df.empty:
        st.header(f"Live Insights: {ticker}")
        
        # Metrics
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        st.metric("Price (INR)", f"₹{curr:,.2f}", f"{curr-prev:,.2f}")

        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], 
                        high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data found. Ensure the ticker is a valid NSE symbol.")
except Exception as e:
    st.error(f"Error: {e}")
