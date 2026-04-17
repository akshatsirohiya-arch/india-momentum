import streamlit as st
import pandas as pd
import yfinance as yf
import os
import time
from google import genai

# --- 1. SETTINGS & CLIENT INIT ---
st.set_page_config(layout="wide", page_title="India Institutional AI Hunter")

# Use Streamlit Secrets for the API Key
if "GEMINI_API_KEY" in st.secrets:
    try:
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"],
            http_options={'api_version': 'v1beta'} 
        )
    except Exception as e:
        st.sidebar.error(f"Client Init Error: {e}")
        client = None
else:
    st.sidebar.warning("🔑 Missing GEMINI_API_KEY in Streamlit Secrets.")
    client = None

# --- 2. ROBUST AI WRAPPER ---
def call_gemini_with_retry(prompt, model_id="gemini-2.0-flash"):
    if not client: return "AI Client not initialized."
    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                time.sleep(3)
                continue
            return f"AI Error: {str(e)}"
    return "Rate limit exceeded."

# --- 3. RESEARCH FUNCTIONS (Optimized for India) ---
def get_high_conviction_summary(df):
    csv_context = df[['Ticker', 'Price', 'Velocity %', 'RVOL']].head(15).to_csv(index=False)
    # Prompt updated for Indian Market context
    prompt = f"Act as an Indian Hedge Fund Manager. Analyze these NSE breakouts: {csv_context}. Identify Top 5 multi-bagger candidates considering India's 2026 GDP growth, budget cycles, and sector tailwinds."
    return call_gemini_with_retry(prompt)

def run_deep_audit(ticker):
    try:
        # AUTOMATIC SUFFIX: Ensures RELIANCE becomes RELIANCE.NS
        india_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
        stock = yf.Ticker(india_ticker)
        info = stock.info
        
        audit_data = {
            "Rev Growth": info.get("revenueGrowth"),
            "Debt/Equity": info.get("debtToEquity"),
            "Free Cash Flow": info.get("freeCashflow"),
            "Promoter Holding": info.get("heldPercentInstitutions") # Key for India
        }
        
        prompt = f"Perform a Deep Fundamental Audit on {india_ticker} (NSE India). DATA: {audit_data}. Analyze the Promoter quality, Moat in the Indian ecosystem, and 2026 Entry/Exit zones."
        return call_gemini_with_retry(prompt), audit_data
    except Exception as e:
        return f"Audit Error: {str(e)}", {}

# --- 4. MAIN APP UI ---
if os.path.exists("daily_watchlist.csv"):
    df = pd.read_csv("daily_watchlist.csv")
    
    # Logic for Velocity calculation
    if 'Slope' in df.columns:
        df['Velocity %'] = (df['Slope'] / df['Price']) * 252 * 100
    
    df = df.sort_values(by="Velocity %", ascending=False)
    
    st.title("🏹 Multi-Bagger Intelligence (India Edition)")
    
    # --- TOP: STRATEGIC SUMMARY ---
    with st.expander("🇮🇳 INDIAN SECTOR RESEARCH", expanded=True):
        if st.button("🚀 Run Institutional Research"):
            with st.spinner("Analyzing NSE Data..."):
                st.markdown(get_high_conviction_summary(df))
    
    st.markdown("---")
    
    # --- MIDDLE: THE DEEP DIVE ---
    st.header("🔬 Deep-Dive Individual Audit")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        target = st.selectbox("Select NSE Ticker", df['Ticker'].unique())
        if st.button(f"🔍 Audit {target}"):
            report, metrics = run_deep_audit(target)
            st.session_state['audit_report'] = report
            st.session_state['audit_metrics'] = metrics
    
    with col2:
        if 'audit_report' in st.session_state:
            m = st.session_state['audit_metrics']
            c = st.columns(3)
            c[0].metric("Rev Growth", f"{m.get('Rev Growth', 0)*100:.1f}%" if m.get('Rev Growth') else "N/A")
            c[1].metric("Debt/Equity", m.get('Debt/Equity', 'N/A'))
            c[2].metric("Inst. Holding", f"{m.get('Promoter Holding', 0)*100:.1f}%" if m.get('Promoter Holding') else "N/A")
            st.markdown(st.session_state['audit_report'])

    st.markdown("---")
    st.header("📊 NSE Watchlist Data")
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.error("daily_watchlist.csv not found. Please upload your NSE scanner output.")
