import streamlit as st
import pandas as pd
import os
import time
from google import genai
from google.genai import types

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="India Deep-Research Alpha")
st.markdown("""<style>.stButton>button { background-color: #007bff; color: white; height: 3em; }</style>""", unsafe_allow_html=True)

# --- AI Client Init ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_ID = "gemini-3-flash-preview" 
except:
    st.error("API Key Missing.")
    st.stop()

def get_deep_research_top_20(df):
    """Performs deep research using Search Grounding for Top 20 picks."""
    # Take the top 30 technically strongest to audit
    shortlist = df.sort_values(by="Slope", ascending=False).head(30)
    stock_summary = shortlist[['Ticker', 'Price', 'Velocity %', 'RVOL']].to_string(index=False)
    
    prompt = f"""
    You are a Senior Equity Research Analyst specializing in the Nifty 500. 
    Review these technical leaders:
    {stock_summary}
    
    TASK:
    1. Perform 'Deep Research' on these stocks using Google Search.
    2. Analyze Business Fundamentals: Look for Q4 FY26 results, profit growth, and order book strength.
    3. Analyze News: Check for recent major contract wins, PLI scheme approvals, or expansion plans.
    4. Provide the TOP 20 HIGH-CONVICTION suggestons.
    
    FOR EACH TOP 20 PICK:
    - Ticker & Rank.
    - Business Context: What do they actually do?
    - Fundamental Strength: Why is the balance sheet healthy? (e.g., Debt, ROCE).
    - News Catalyst: What is the specific 'spark' for the current breakout?
    - 1-Year Horizon Outlook: High/Medium/Low confidence.
    """

    try:
        # Enable Search Grounding (Crucial for live news & financials)
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

# --- Main App ---
def main():
    st.title("🏹 India Deep-Research Institutional Audit")
    
    if os.path.exists("daily_watchlist.csv"):
        df = pd.read_csv("daily_watchlist.csv")
        # Ensure Velocity column exists for the leaderboard
        df['Velocity %'] = (df['Slope'] * 252 * 100).round(2)
        
        if st.button("🔍 Perform Deep Fundamental & News Audit"):
            with st.spinner("AI is currently browsing NSE filings, news portals, and analyst reports..."):
                report = get_deep_research_top_20(df)
                st.session_state['deep_report'] = report
        
        if 'deep_report' in st.session_state:
            st.markdown("### 🏆 Institutional Research Report")
            st.markdown(st.session_state['deep_report'])
            st.download_button("Download Full Research (.txt)", st.session_state['deep_report'], "Deep_Research_Report.txt")

        st.markdown("---")
        st.subheader("Technical Shortlist (The Input Data)")
        st.dataframe(df.sort_values(by="Velocity %", ascending=False), use_container_width=True)
    else:
        st.warning("No technical data found. Run the GitHub Action scan first.")

if __name__ == "__main__":
    main()
