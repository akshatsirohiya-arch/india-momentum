import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types

# --- Page Config ---
st.set_page_config(layout="wide", page_title="India Alpha Hunter 2026")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stDataFrame { border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Client Initialization ---
try:
    # Ensure GEMINI_API_KEY is in your Streamlit Secrets
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Key Missing. Add GEMINI_API_KEY to Streamlit Settings > Secrets.")
    st.stop()

def get_top_20_with_research(df):
    """Sends technical data to AI and triggers Google Search for deep audit."""
    
    # Sort and take top 40 technically strongest to give AI a good pool
    candidates = df.sort_values(by="Slope", ascending=False).head(40)
    tech_data = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = f"""
    You are a Lead Equity Analyst at a Tier-1 Indian Investment Bank. 
    I am providing you with technical scan data for NSE stocks making Higher Highs/Lows.

    TECHNICAL DATASET:
    {tech_data}

    YOUR TASK:
    1. Use GOOGLE SEARCH to research the latest (Q1/Q2 2026) fundamentals for these stocks.
    2. Check for: Net Profit margins, Debt-to-Equity, and recent NSE announcements/news.
    3. Filter and return the TOP 20 HIGH-CONVICTION BUYS.
    4. Rank them 1 to 20 based on both Technical Momentum and Fundamental Strength.
    
    OUTPUT FORMAT:
    - Provide a Markdown Table with columns: Rank, Ticker, Verdict, Fundamental Catalyst, and Risk Level.
    - Below the table, provide a 'Top 3 Deep Dive' explaining why those three are the best for 2026.
    - Mention any stocks to AVOID due to bad news or high debt.
    """

    try:
        # We use Gemini 2.5 Flash with the Search Tool enabled
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text
    except Exception as e:
        return f"AI Deep-Dive Failed: {str(e)}"

# --- UI Logic ---
def main():
    st.title("🏹 India Institutional Alpha Hunter")
    st.subheader("Technical Momentum + AI Fundamental Grounding")

    if os.path.exists("daily_watchlist.csv"):
        df = pd.read_csv("daily_watchlist.csv")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric("Stocks Scanned", len(df))
            st.write("Click below to run the AI Research engine on the technical leaders.")
            if st.button("🚀 Run Deep-Research Audit"):
                with st.spinner("AI is browsing NSE news and financials..."):
                    report = get_top_20_with_research(df)
                    st.session_state['ai_report'] = report

        with col2:
            if 'ai_report' in st.session_state:
                st.markdown("### 🏆 AI Institutional Recommendations")
                st.markdown(st.session_state['ai_report'])
            else:
                st.info("The AI report will appear here once you click the button.")

        st.markdown("---")
        st.subheader("Raw Technical Leaderboard (Last Scan)")
        # Show annualized velocity for easier reading
        df['Velocity %'] = (df['Slope'] * 252 * 100).round(2)
        st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)

    else:
        st.warning("⚠️ daily_watchlist.csv not found. Please trigger the GitHub Action scan first.")

if __name__ == "__main__":
    main()
