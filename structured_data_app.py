import os
import io
import logging
import streamlit as st
import pandas as pd
import importlib.metadata

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

# 🐞 Optional: Verify installed packages
installed = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
st.sidebar.write("🔍 Packages starting with 'sentry':", [n for n in installed if n.lower().startswith("sentry")])

# 🌐 Initialize Sentry if configured
dsn = st.secrets.get("SENTRY_DSN")
if dsn:
    from sentry_sdk import init as sentry_init
    from sentry_sdk.integrations.logging import LoggingIntegration

    logging_integration = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR
    )
    sentry_init(
        dsn="https://706656d5eb7a8fe73aecc1ecfad78a61@o4509464691015680.ingest.us.sentry.io/4509464705499136",
        integrations=[logging_integration],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Sentry initialized for DataChat AI app.")

    # 🔔 One-time Sentry verification: raise error on first run
    if "sentry_tested" not in st.session_state:
        st.session_state.sentry_tested = True
        1 / 0
        
division = 1 / 0

# 🔑 Load OpenAI API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    st.stop()

# 🤖 Import ChatOpenAI with fallback
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent

# 🗂️ Caching utilities using bytes for hashability
@st.cache_data

def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

@st.cache_resource
# Cache the agent (keyed on the DataFrame object)
def get_agent(df: pd.DataFrame):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )

# 🖥️ App UI
st.title("💬 DataChat AI — Ask Your Spreadsheets")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Excel or CSV",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file:
    # Read and cache DataFrame
    file_bytes = uploaded_file.read()
    df = get_dataframe(file_bytes, uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head())

    # Create or retrieve the agent
    agent = get_agent(df)
    query = st.text_input("Ask a question about your data:")
    if st.button("🤖 Ask DataChat"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = agent.run(query)
                except Exception as e:
                    logging.error("Agent run failed", exc_info=True)
                    st.error(f"❌ Error: {str(e)}")
                else:
                    st.markdown("**Answer:**")
                    st.write(answer)
else:
    st.info("👉 Upload a spreadsheet in the sidebar to get started!")
