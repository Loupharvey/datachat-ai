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

# 🔒 Password auth
PASSWORD = st.secrets.get("PASSWORD", "")
if not PASSWORD:
    st.error("App password not configured. Set 'PASSWORD' in Streamlit secrets.")
    st.stop()
password = st.sidebar.text_input("🔒 App Password", type="password")
if password != PASSWORD:
    st.sidebar.error("❌ Incorrect password")
    st.stop()

# 🐞 Debug: Check sentry-sdk
installed = [dist.metadata.get("Name") for dist in importlib.metadata.distributions()]
st.sidebar.write("🔍 sentry-sdk installed?", [n for n in installed if n and n.startswith("sentry")])

# 🌐 Initialize Sentry SDK
dsn = st.secrets.get("SENTRY_DSN")
if dsn:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    sentry_sdk.init(
        dsn="https://706656d5eb7a8fe73aecc1ecfad78a61@o4509464691015680.ingest.us.sentry.io/4509464705499136",
        integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )
    logging.getLogger(__name__).info("Sentry initialized")
    if not st.session_state.get("sentry_tested"):
        st.session_state["sentry_tested"] = True
        sentry_sdk.capture_message("Initial Sentry test event from DataChat AI app")

# 🔑 Load API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in secrets or env vars.")
    st.stop()

# 🗂️ Caching utilities
@st.cache_data
def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    system_msg = SystemMessage(content=(
        "You are an expert financial data analyst. df has positive values = revenues, negative = costs. "
        "When asked for aggregates, produce pandas code in ```python``` and then results & summary."
    ))
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.0)
    return create_pandas_dataframe_agent(
        llm, df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix_messages=[system_msg]
    )

# 🖥️ App UI
st.title("💬 DataChat AI — Ask Your Spreadsheets")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel or CSV", type=["csv","xls","xlsx"])

if uploaded_file:
    df = get_dataframe(uploaded_file.read(), uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]}×{df.shape[1]}")
    st.dataframe(df.head())

    agent = get_agent(df)
    query = st.text_input("Ask a question about your data:")

    if st.button("🤖 Ask DataChat"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            q = query.lower()
            num_cols = df.select_dtypes(include="number").columns
            cat_col = next((c for c in df.columns if "type" in c.lower()), None)

            # revenues per category
            if ("each revenue" in q or "for each revenue" in q) and cat_col:
                df_pos = df.copy()
                df_pos[num_cols] = df_pos[num_cols].clip(lower=0)
                grp = df_pos.groupby(cat_col)[num_cols].sum().sum(axis=1)
                grp = grp[grp > 0].sort_values(ascending=False)
                st.table(grp.rename_axis(cat_col).reset_index(name="Total Revenue"))

            # costs per category
            elif ("each cost" in q or "for each cost" in q) and cat_col:
                df_neg = df.copy()
                df_neg[num_cols] = df_neg[num_cols].clip(upper=0).abs()
                grp = df_neg.groupby(cat_col)[num_cols].sum().sum(axis=1)
                grp = grp[grp > 0].sort_values(ascending=False)
                st.table(grp.rename_axis(cat_col).reset_index(name="Total Cost"))

            # total revenue year
            elif "total revenue" in q:
                total_rev = df[num_cols].clip(lower=0).sum().sum()
                st.markdown(f"**Total revenue (year):** {total_rev:,.2f}")

            # total cost year
            elif "total cost" in q:
                total_cost = df[num_cols].clip(upper=0).abs().sum().sum()
                st.markdown(f"**Total cost (year):** {total_cost:,.2f}")

            # profitability queries
            elif "profitability" in q:
                # compile pattern for date columns
                date_pattern = re.compile(r"^[A-Za-z]+-\d{4}$")
                # extract tokens in query
                tokens = re.findall(r"([A-Za-z]+-\d{4})", q)
                # find matching columns
                date_cols = [c for c in df.columns if date_pattern.match(c)]
                # sort chronologically
                month_map = {m: i+1 for i, m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"])}
                date_cols_sorted = sorted(
                    date_cols,
                    key=lambda c: (int(c.split('-')[1]), month_map.get(c.split('-')[0].lower(), 0))
                )
                # multi-month range
                if len(tokens) == 2:
                    start, end = tokens
                    if start in date_cols_sorted and end in date_cols_sorted:
                        i1, i2 = date_cols_sorted.index(start), date_cols_sorted.index(end)
                        sel = date_cols_sorted[min(i1, i2): max(i1, i2) + 1]
                    else:
                        sel = []
                    if not sel:
                        st.error("No data columns found for that date range.")
                    else:
                        rev = df[sel].clip(lower=0).sum().sum()
                        cost = df[sel].clip(upper=0).abs().sum().sum()
                        profit = rev - cost
                        st.markdown(f"**Profitability ({start} to {end}):** {profit:,.2f}  \n*(Revenue {rev:,.2f} – Cost {cost:,.2f})*")
                # single month
                elif len(tokens) == 1:
                    key = tokens[0]
                    if key in date_cols:
                        rev = df[key].clip(lower=0).sum()
                        cost = df[key].clip(upper=0).abs().sum()
                        profit = rev - cost
                        st.markdown(f"**Profitability for {key}:** {profit:,.2f}  \n*(Revenue {rev:,.2f} – Cost {cost:,.2f})*")
                    else:
                        st.error(f"Column '{tokens[0]}' not found.")
                # annual
                else:
                    rev = df[num_cols].clip(lower=0).sum().sum()
                    cost = df[num_cols].clip(upper=0).abs().sum().sum()
                    profit = rev - cost
                    st.markdown(f"**Profitability (year):** {profit:,.2f}  \n*(Revenue {rev:,.2f} – Cost {cost:,.2f})*")
            else:
                # fallback to LLM
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
    st.info("👉 Upload a spreadsheet to get started!")
