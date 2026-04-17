import streamlit as st
import pandas as pd
import os
from google import genai

st.set_page_config(layout="wide", page_title="India Institutional Hunter")

# Initialization
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API Key Missing in Secrets.")
    st.stop()

def get_ai_recommendations(df):
    """Sends the entire filtered batch to AI for a 'Top Picks' analysis."""
    
    # Sort by Slope and take top 25 to keep within token limits
    top_candidates = df.sort_values(by="Slope", ascending=False).head(25)
    
    # Convert to a compact string for the AI
    stock_list_str = top_candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = (
        "You are an institutional fund manager. Here is a list of NSE stocks with "
        "confirmed Higher-High/Higher-Low structures and high momentum slopes.\n\n"
        f"{stock_list_str}\n\n"
        "Task: Identify the TOP 3 stocks for a 'Position Trade'. "
        "Base your decision on the balance between high Slope and RVOL (Relative Volume). "
        "Provide a concise reasoning for each pick in 2 sentences."
    )
    
    try:
        # Use 1.5-Flash for higher batch-processing quotas
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"

# UI Logic
if os.path.exists("daily_watchlist.csv"):
    df = pd.read_csv("daily_watchlist.csv")
    
    st.title("🏹 India Full-Market Momentum")
    
    if st.button("🤖 Run AI Batch Analysis (Pick Top 3)"):
        with st.spinner("Analyzing the entire watchlist..."):
            recommendations = get_ai_recommendations(df)
            st.success("### Institutional Top Picks")
            st.markdown(recommendations)

    st.markdown("---")
    st.subheader("Full Momentum Watchlist")
    st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)
else:
    st.warning("Please run the GitHub Action to generate the watchlist first.")
