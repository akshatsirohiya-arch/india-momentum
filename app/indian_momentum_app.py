import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types

# --- Page Config & UI ---
st.set_page_config(layout="wide", page_title="India Institutional Alpha 2026")

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    .report-box { padding: 20px; border-radius: 10px; border: 1px solid #30363d; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Client initialization ---
try:
    # Uses 'gemini-3-flash-preview' - the 2026 production standard
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_ID = "gemini-3-flash-preview" 
except Exception as e:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

def get_ai_alpha_research(df):
    """The 'US-Logic' Engine: Technicals + Live Google Search Grounding."""
    
    # Take top 40 technical candidates to audit
    candidates = df.sort_values(by="Slope", ascending=False).head(40)
    stock_summary = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = f"""
    Role: Institutional Equity Research Lead (NSE/BSE specialist).
    Context: Analysis for April 2026 market conditions.
    
    Technical Watchlist:
    {stock_summary}
    
    Task:
    1. Use GOOGLE SEARCH to identify the latest catalysts (Q1 2026 results, order wins, or management changes).
    2. Cross-reference technical momentum (Slope) with fundamental health (Debt, Profit Growth).
    3. Return the TOP 20 HIGH-CONVICTION 'POSITION TRADES'.
    
    Output Format:
    - A numbered list (1-20) with: Ticker, Rating (Buy/Strong Buy), and a '30-word Research Alpha' snippet.
    - Specifically flag any stock with 'Negative Grounding' (bad news found during search).
    """

    try:
        # Enable Google Search Tooling (Grounding)
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
    
    if os.path.exists("daily_watchlist.csv"):
        df = pd.read_csv("daily_watchlist.csv")
        
        col_stats, col_action = st.columns([1, 2])
        with col_stats:
            st.metric("Technical Candidates", len(df))
        with col_action:
            if st.button("🔍 Run Full-Market AI Research"):
                with st.spinner("AI is browsing live NSE data for 2026 catalysts..."):
                    result = get_ai_alpha_research(df)
                    st.session_state['research_result'] = result

        if 'research_result' in st.session_state:
            st.markdown("### 🏆 AI Institutional Research: Top 20 Picks")
            st.markdown(f'<div class="report-box">{st.session_state["research_result"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Current Technical Leaderboard")
        st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)
    else:
        st.warning("No technical data found. Run your GitHub Action scan first.")

if __name__ == "__main__":
    main()
