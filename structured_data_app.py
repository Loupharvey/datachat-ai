# structured_data_app.py

import pandas as pd
import streamlit as st


# LangChain & OpenAI imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenA
from langchain_experimental.agents import create_pandas_dataframe_agent


# Read API key directly from Streamlit secrets
API_KEY = st.secrets["OPENAI_API_KEY"]
if not API_KEY:
    st.error("🔑 No OPENAI_API_KEY found in Streamlit secrets!")
    st.stop()

# Install note:
# pip install streamlit pandas openai langchain-community langchain-experimental tabulate openpyxl

st.set_page_config(page_title="📊 DataChat AI", layout="centered")
st.title("💬 DataChat AI — Ask Your Spreadsheets")

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file", 
    type=["xlsx", "xls", "csv"]
)

# Persist agent in session
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.df = None

if uploaded_file:
    # Read into DataFrame
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    st.success(f"📥 Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head())

    # Create / cache the agent
    if st.session_state.agent is None or st.session_state.df is not df:
        llm = ChatOpenAI(api_key=API_KEY, temperature=0)
        st.session_state.agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, allow_dangerous_code=True
        )
        st.session_state.df = df

    # --- Query input ---
    query = st.text_input("Ask a question about your data:")
    if st.button("🤖 Ask DataChat"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = st.session_state.agent.run(query)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
