# structured_data_app.py

import pandas as pd
import streamlit as st
import pkg_resources

# 1️⃣ PAGE CONFIG must come first
st.set_page_config(page_title="📊 DataChat AI", layout="centered")

# Debug: list any installed packages that start with "sentry"
installed = {pkg.key for pkg in pkg_resources.working_set}
sentry_installed = any(name.startswith("sentry") for name in installed)
st.write("🔍 sentry-sdk installed?", sentry_installed)
st.write("🔍 Found packages:", sorted([name for name in installed if name.startswith("sentry")]))

import sentry_sdk

sentry_sdk.init(
    dsn="https://706656d5eb7a8fe73aecc1ecfad78a61@o4509464691015680.ingest.us.sentry.io/4509464705499136",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)

# … your Sentry init and logger setup above …

# ——————————————————————————————————————————————
# 🔥 Test Sentry integration
# ——————————————————————————————————————————————
if st.sidebar.button("💥 Test Sentry"):
    # This will raise an unhandled exception and be captured by Sentry
    1 / 0

# ——————————————————————————————————————————————
# … rest of your app …


# — Basic Password Auth —————————————————————————————
PASSWORD = st.secrets["PASSWORD"]
pw = st.sidebar.text_input("🔒 Enter app password", type="password")
if pw != PASSWORD:
    st.error("❌ Incorrect password")
    st.stop()
# — End Basic Auth ————————————————————————————————————

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
