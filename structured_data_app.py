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

# 📋 Page configuration
st.set_page_config(
    page_title="📊 DataChat AI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 🔒 Simple password auth
PASSWORD = st.secrets.get("PASSWORD", "")
if not PASSWORD:
    st.error("App password is not configured. Please set 'PASSWORD' in Streamlit secrets.")
    st.stop()
password = st.sidebar.text_input("🔒 App Password", type="password")
if password != PASSWORD:
    st.sidebar.error("❌ Incorrect password")
    st.stop()

# 🐞 Check sentry-sdk presence
installed = [dist.metadata.get("Name") for dist in importlib.metadata.distributions()]
st.sidebar.write("🔍 Packages starting with 'sentry':", [n for n in installed if n and n.lower().startswith("sentry")])

# 🌐 Initialize Sentry SDK
dsn = st.secrets.get("SENTRY_DSN")
if dsn:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    logging_integration = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
    sentry_sdk.init(dsn="https://706656d5eb7a8fe73aecc1ecfad78a61@o4509464691015680.ingest.us.sentry.io/4509464705499136"
                    , integrations=[logging_integration], traces_sample_rate=0.1, send_default_pii=True)
    logger = logging.getLogger(__name__)
    logger.info("Sentry initialized")
    if not st.session_state.get("sentry_tested"):
        st.session_state["sentry_tested"] = True
        sentry_sdk.capture_message("Initial Sentry test event from DataChat AI app")

# 🔑 Load API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    st.stop()

# 🗂️ Caching utilities
@st.cache_data
def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    # System prompt for financial analysis
    system_msg = SystemMessage(content=(
        "You are an expert financial data analyst. df has positive values = revenues, negative = costs. "
        "When asked for specific aggregates, produce pandas code in ```python and then results and summary."
    ))
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.0)
    return create_pandas_dataframe_agent(
        llm, df,
        verbose=False, allow_dangerous_code=True,
        handle_parsing_errors=True, prefix_messages=[system_msg]
    )

# 🖥️ App UI
st.title("💬 DataChat AI — Ask Your Spreadsheets")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel or CSV", type=["xlsx", "xls", "csv"] )

if uploaded_file:
    file_bytes = uploaded_file.read()
    df = get_dataframe(file_bytes, uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]}×{df.shape[1]}")
    st.dataframe(df.head())

    agent = get_agent(df)
    query = st.text_input("Ask a question about your data:")
    if st.button("🤖 Ask DataChat"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            q = query.lower()
            # identify numeric columns and category column
            num_cols = df.select_dtypes(include="number").columns
            cat_col = next((c for c in df.columns if "type" in c.lower()), None)
            # direct computation for common requests
            if "each revenue" in q or "for each revenue" in q:
                if cat_col:
                    rev = df[df[num_cols] >= 0]
                    grouped = rev.groupby(cat_col)[num_cols].sum().sum(axis=1)
                    st.table(grouped.rename_axis(cat_col).reset_index(name="Total Revenue"))
                else:
                    st.error("No category column found for grouping revenues.")
            elif "each cost" in q or "for each cost" in q:
                if cat_col:
                    cost = df[df[num_cols] < 0]
                    grouped = cost.groupby(cat_col)[num_cols].sum().abs().sum(axis=1)
                    st.table(grouped.rename_axis(cat_col).reset_index(name="Total Cost"))
                else:
                    st.error("No category column found for grouping costs.")
            elif "total revenue" in q:
                total_rev = df[num_cols].clip(lower=0).sum().sum()
                st.markdown(f"**Total revenue:** {total_rev:,.2f}")
            elif "total cost" in q:
                total_cost = df[num_cols].clip(upper=0).abs().sum().sum()
                st.markdown(f"**Total cost:** {total_cost:,.2f}")
            elif "profitability" in q:
                rev = df[num_cols].clip(lower=0).sum().sum()
                cost = df[num_cols].clip(upper=0).abs().sum().sum()
                profit = rev - cost
                st.markdown(f"**Profitability:** {profit:,.2f}  \n*(Revenue {rev:,.2f} – Cost {cost:,.2f})*")
            else:
                with st.spinner("Thinking…"):
                    try:
                        answer = agent.run(query)
                        answer = re.sub(r"\.([A-Za-z])", r". \1", answer)
                    except Exception as e:
                        logging.error("Agent run failed", exc_info=True)
                        st.error(f"❌ Error: {str(e)}")
                    else:
                        st.markdown("**Answer:**")
                        st.write(answer)
else:
    st.info("👉 Upload a spreadsheet in the sidebar to get started!")
