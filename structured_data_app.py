import os
import io
import re
import logging
import streamlit as st
import pandas as pd
import importlib.metadata

# LangChain imports
from langchain.schema import SystemMessage
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 📋 Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="📊 DataChat AI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 🔒 Simple password auth (use st.secrets for secure storage)
PASSWORD = st.secrets.get("PASSWORD", "")
if not PASSWORD:
    st.error("App password is not configured. Please set 'PASSWORD' in Streamlit secrets.")
    st.stop()
password = st.sidebar.text_input("🔒 App Password", type="password")
if password != PASSWORD:
    st.sidebar.error("❌ Incorrect password")
    st.stop()

# 🐞 Verify sentry-sdk presence
installed = [dist.metadata.get("Name") for dist in importlib.metadata.distributions()]
st.sidebar.write("🔍 Installed packages starting with 'sentry':", [n for n in installed if n and n.lower().startswith("sentry")])

# 🌐 Initialize Sentry SDK if configured and send a test message once
dsn = st.secrets.get("SENTRY_DSN")
if dsn:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    logging_integration = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )
    sentry_sdk.init(
        dsn=dsn,
        integrations=[logging_integration],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Sentry initialized for DataChat AI app.")

    # 🔔 One-time Sentry verification message
    if not st.session_state.get("sentry_tested"):
        st.session_state["sentry_tested"] = True
        sentry_sdk.capture_message("Initial Sentry test event from DataChat AI app")

# 🔑 Load OpenAI API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    st.stop()

# 🗂️ Caching utilities using bytes for hashability
@st.cache_data
def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    # System prompt: expert financial analyst
    system_msg = SystemMessage(content=(
        "You are an expert financial data analyst and data engineer. "
        "The DataFrame 'df' contains numeric values where positive values represent revenues and negative values represent costs. "
        "Profitability is revenue minus cost (i.e., sum of all values). "
        "When given a question, generate a Python code snippet in ```python that uses pandas to compute exactly what is asked (e.g., total revenues for 2023, total costs for 2023, profitability for March 2023). "
        "After the code, output the numeric results and a concise summary in plain English."
    ))
    # Use GPT-4 with deterministic output
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.0)
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix_messages=[system_msg],
    )

# 🖥️ App UI
st.title("💬 DataChat AI — Ask Your Spreadsheets")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Excel or CSV",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    df = get_dataframe(file_bytes, uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head())

    agent = get_agent(df)
    query = st.text_input("Ask a question about your data:")
    if st.button("🤖 Ask DataChat"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = agent.run(query)
                    # Fix runs-together sentences by ensuring a space after periods
                    answer = re.sub(r"\.([A-Za-z])", r". \1", answer)
                except Exception as e:
                    logging.error("Agent run failed", exc_info=True)
                    st.error(f"❌ Error: {str(e)}")
                else:
                    st.markdown("**Answer:**")
                    st.write(answer)
else:
    st.info("👉 Upload a spreadsheet in the sidebar to get started!")
