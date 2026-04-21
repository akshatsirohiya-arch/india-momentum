import streamlit as st
import pandas as pd
import os
from google import genai

# --- Page Config ---
st.set_page_config(layout="wide", page_title="India Momentum Dashboard")

# --- AI Client Initialization ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_ID = "gemini-3-flash-preview" 
except:
    st.error("API Key Missing in Secrets.")
    st.stop()

def get_top_20_suggestions(df):
    """Simple prompt for top 20 investment logic based on technicals."""
    # Send the top 30 to give AI some choice
    candidates = df.sort_values(by="Slope", ascending=False).head(30)
    data_str = candidates[['Ticker', 'Price', 'Slope', 'RVOL']].to_string(index=False)
    
    prompt = f"""
    Acting as an institutional fund manager, analyze this technical shortlist of NSE stocks:
    {data_str}
    
    1. Identify the TOP 20 investment suggestions from this list.
    2. Rank them based on trend sustainability (Slope) and volume health (RVOL).
    3. Provide a brief (1-sentence) technical reason for each pick.
    4. Format as a clean numbered list.
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text
    except Exception as e:
        return f"AI Logic Error: {str(e)}"

# --- Main UI ---
def main():
    st.title("🏹 India Institutional Hunter")
    
    if os.path.exists("daily_watchlist.csv"):
        df = pd.read_csv("daily_watchlist.csv")
        
        # --- Sidebar Actions ---
        st.sidebar.header("Data Actions")
        
        # 1. DOWNLOAD FULL SHORTLIST (For use in Gemini App)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Full Shortlist (CSV)",
            data=csv_data,
            file_name=f"NSE_Shortlist_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            help="Download this file to upload it into the Gemini App for custom analysis."
        )

        # --- Main Panel ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Stocks in Watchlist", len(df))
            if st.button("🤖 Generate Top 20 Suggestions"):
                with st.spinner("AI analyzing momentum..."):
                    result = get_top_20_suggestions(df)
                    st.session_state['suggestions'] = result

        with col2:
            if 'suggestions' in st.session_state:
                st.markdown("### 🏆 Top 20 Investment Suggestions")
                st.markdown(st.session_state['suggestions'])
                # Option to download the AI text specifically
                st.download_button("Download AI Suggestions (.txt)", st.session_state['suggestions'], "Top20_Picks.txt")

        st.markdown("---")
        st.subheader("Technical Leaderboard")
        st.dataframe(df.sort_values(by="Slope", ascending=False), use_container_width=True)
        
    else:
        st.warning("No data found. Run your GitHub Action scan first.")

if __name__ == "__main__":
    main()
