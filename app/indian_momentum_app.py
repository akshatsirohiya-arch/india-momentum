import streamlit as st
import pandas as pd
import os
import time
from google import genai

# ... (Configuration and CSS) ...

# Safe Client Initialization
try:
    # We use the key from st.secrets
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

def get_ai_recommendations(df):
    """Sends the filtered batch to AI for a Top 20 analysis."""
    
    # Sort by Slope and take enough to let the AI choose the best 20
    # We take top 40 so the AI has a pool to filter from
    candidates = df.sort_values(by="Slope", ascending=False).head(40)
    
    # Convert to string
    stock_list_str = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = (
        "You are an expert Institutional Trader specializing in the Indian Market (NSE). "
        "I am providing you with a list of stocks that have passed a technical filter for "
        "confirmed Higher-High/Higher-Low structures and high momentum slopes.\n\n"
        f"{stock_list_str}\n\n"
        "TASK:\n"
        "1. Identify the TOP 20 'High-Conviction' buys from this list.\n"
        "2. Rank them 1 to 20 based on the 'Quality of Trend' (High Slope + Healthy RVOL between 1.0 and 3.0).\n"
        "3. For each of the top 5, provide a 1-sentence technical justification.\n"
        "4. Format the output as a clean numbered list."
    )
    
    try:
        # UPDATED MODEL STRING: Using the full identifier 'gemini-1.5-flash'
        # Some SDK versions prefer just the name, others the full path. 
        # 'gemini-1.5-flash' is the standard for the genai SDK.
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        # If it still fails, try the 1.5-flash-8b (lighter version)
        return f"AI Analysis Error: {str(e)}"

# UI Logic
if os.path.exists("daily_watchlist.csv"):
    df = pd.read_csv("daily_watchlist.csv")
    
    st.title("🏹 India Institutional Momentum")
    
    if st.button("🤖 Run AI Deep-Dive (Top 20 Buys)"):
        if len(df) < 5:
            st.warning("Watchlist too small for a Top 20 analysis. Run the scan first.")
        else:
            with st.spinner("Analyzing market structure for the Top 20..."):
                recommendations = get_ai_recommendations(df)
                st.markdown("### 🏆 Institutional Top 20 Recommendations")
                st.write(recommendations)

    st.markdown("---")
    st.subheader("Raw Scan Results")
    st.dataframe(df, use_container_width=True)
