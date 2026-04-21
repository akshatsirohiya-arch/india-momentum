import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types

# --- Page Config & UI ---
st.set_page_config(layout="wide", page_title="India Alpha Hunter 2026")

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; border-radius: 8px; }
    .report-box { padding: 25px; border-radius: 12px; border: 1px solid #30363d; background-color: #161b22; color: #e6edf3; }
    .stDownloadButton>button { background-color: #238636 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Client Initialization ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_ID = "gemini-3-flash-preview" 
except Exception as e:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

def get_ai_alpha_research(df):
    """Deep Research Engine: Technicals + Industry + Return Potential."""
    
    candidates = df.sort_values(by="Slope", ascending=False).head(40)
    stock_summary = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = f"""
    Role: Institutional Equity Research Lead (NSE/BSE specialist).
    Task: Rank the Top 20 High-Conviction stocks for a 1-year horizon.
    
    Technical Watchlist:
    {stock_summary}
    
    INSTRUCTIONS:
    1. Use GOOGLE SEARCH to identify the Industry/Sector for each stock.
    2. Search for the latest Analyst Price Targets or Consensus Forecasts for 2027.
    3. Calculate the '1-Year Return Potential %' based on current vs target price.
    4. Provide a 'Research Alpha' snippet (Current catalysts like earnings, order wins).

    OUTPUT FORMAT (Markdown Table):
    | Rank | Ticker | Industry | Price | 1Y Potential (%) | Verdict | Key Catalyst |
    |------|--------|----------|-------|------------------|---------|--------------|
    
    Include a 'Sector Analysis' at the end to show where the most momentum is concentrated.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text
    except Exception as e:
        return f"Research Engine Error: {str(e)}"

# --- Main App Logic ---
def main():
    st.title("🏹 India Institutional Alpha Hunter")
    st.caption("Technical Momentum + AI Fundamental Grounding + Sector Analysis")
    
    if os.path.exists("daily_watchlist.csv"):
        df = pd.read_csv("daily_watchlist.csv")
        
        col_stats, col_action = st.columns([1, 2])
        with col_stats:
            st.metric("Technical Candidates", len(df))
        with col_action:
            if st.button("🔍 Run Full-Market AI Research"):
                with st.spinner("AI is researching sectors and price targets..."):
                    result = get_ai_alpha_research(df)
                    st.session_state['research_result'] = result

        if 'research_result' in st.session_state:
            st.markdown("### 🏆 AI Institutional Research: Top 20 Picks")
            
            # Action Buttons Row
            btn_col1, btn_col2 = st.columns([4, 1])
            with btn_col2:
                st.download_button(
                    label="📥 Download .TXT",
                    data=st.session_state['research_result'],
                    file_name=f"India_Top20_Report_{pd.Timestamp.now().strftime('%Y-%m-%d')}.txt",
                    mime="text/plain"
                )
            
            st.markdown(f'<div class="report-box">{st.session_state["research_result"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Technical Leaderboard")
        df['Velocity %'] = (df['Slope'] * 252 * 100).round(2)
        st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)
    else:
        st.warning("No technical data found. Run your GitHub Action scan first.")

if __name__ == "__main__":
    main()
