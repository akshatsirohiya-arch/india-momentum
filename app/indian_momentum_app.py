import streamlit as st
import pandas as pd
import os
from google import genai

st.set_page_config(layout="wide", page_title="India Institutional Hunter")

# AI Setup
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if os.path.exists("daily_watchlist.csv"):
    df = pd.read_csv("daily_watchlist.csv")
    
    # Exact Momentum Metric: Annualized Log Velocity
    df['Velocity %'] = (df['Slope'] * 252) * 100
    df = df.sort_values(by="Velocity %", ascending=False)

    st.title("🏹 Full India Market AI Hunter")
    
    # Summary of Top Breakouts
    if st.button("🚀 Institutional AI Summary"):
        top_context = df.head(20).to_csv(index=False)
        prompt = f"Analyze these top Indian market velocity leaders: {top_context}. Identify 3 candidates with high probability of institutional accumulation."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        st.markdown(response.text)

    st.markdown("---")
    st.header("📊 Full NSE Watchlist")
    st.dataframe(df, use_container_width=True)
else:
    st.error("Market scan file not found. Trigger the GitHub Action to generate it.")
