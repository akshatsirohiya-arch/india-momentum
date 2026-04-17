import streamlit as st
import pandas as pd
import os
from google import genai

# Initialization
try:
    # Using the standard 2026 client initialization
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API Key Missing in Streamlit Secrets.")
    st.stop()

def get_top_20_buys(df):
    """Sends the entire filtered batch to AI for a Top 20 ranking."""
    
    # Take the top 50 momentum stocks to let the AI filter for the best 20
    candidates = df.sort_values(by="Slope", ascending=False).head(50)
    
    # Convert to a clean table for the prompt
    stock_data = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = (
        "You are an Institutional Lead Trader for an Indian Hedge Fund. "
        "Review this watchlist of NSE stocks that have cleared our HH/HL filters:\n\n"
        f"{stock_data}\n\n"
        "TASK:\n"
        "1. Select the TOP 20 'High-Conviction' buys from this list.\n"
        "2. Rank them 1 to 20. Priority should be given to stocks where RVOL is between 1.2 and 3.0 "
        "(Institutional accumulation) and the Slope is steadily high.\n"
        "3. Provide a 'Verdict' for the Top 5 specifically.\n"
        "4. Output a clean markdown table of the Top 20."
    )
    
    try:
        # gemini-2.5-flash is the stable production model for 2026
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Fallback to 3.0-flash if your region is on the experimental tier
        if "404" in str(e):
            try:
                response = client.models.generate_content(model="gemini-3-flash", contents=prompt)
                return response.text
            except: pass
        return f"AI Analysis Error: {str(e)}"

# UI Logic
if os.path.exists("daily_watchlist.csv"):
    df = pd.read_csv("daily_watchlist.csv")
    
    st.title("🏹 India Institutional Momentum Hunter")
    
    if st.button("🚀 Analyze Full Watchlist for Top 20 Buys"):
        with st.spinner("Processing full market batch..."):
            recommendations = get_top_20_buys(df)
            st.markdown(recommendations)

    st.markdown("---")
    st.subheader("Raw Technical Scan")
    st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)
else:
    st.warning("Run the GitHub Action to generate 'daily_watchlist.csv' first.")
