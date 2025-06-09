import os
import io
import logging
import re

import streamlit as st
import pandas as pd
import importlib.metadata

from fpdf import FPDF
from langchain_experimental.agents import create_pandas_dataframe_agent

# 📋 Page configuration (must be first)
st.set_page_config(page_title="📊 DataChat AI", layout="centered")

# 🔒 Password auth
PASSWORD = st.secrets.get("PASSWORD", "")
if not PASSWORD:
    st.error("App password not configured; set 'PASSWORD' in Streamlit secrets.")
    st.stop()
pw = st.sidebar.text_input("🔒 App Password", type="password")
if pw != PASSWORD:
    st.sidebar.error("❌ Incorrect password")
    st.stop()

# 🐞 Optional: Show installed Sentry packages
installed = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
st.sidebar.write("🔍 Packages starting with 'sentry':", [n for n in installed if n.lower().startswith("sentry")])

# 🌐 Initialize Sentry if configured
dsn = st.secrets.get("SENTRY_DSN")
if dsn:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    logging_integration = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
    sentry_sdk.init(
        dsn=dsn,
        integrations=[logging_integration],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Sentry initialized for DataChat AI app.")

# 🔑 Load OpenAI key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in secrets or env vars.")
    st.stop()

# 🤖 Import ChatOpenAI with fallback
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# PDF helper
def df_to_pdf(df: pd.DataFrame, title: str = "Report") -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    col_width = pdf.epw / len(df.columns)  # equal column widths

    # Header row
    for col in df.columns:
        pdf.cell(col_width, 8, col, border=1, align="C")
    pdf.ln()

    # Data rows
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, 8, str(item), border=1)
        pdf.ln()

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# 🗂 Caching helpers
@st.cache_data
def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, model="gpt-4")
    system_msg = {"role": "system", "content": (
        "You are an expert data analyst. "
        "First summarize the DataFrame, then answer the user's question clearly. "
        "When needed, generate Python/pandas code snippets."
    )}
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix_messages=[system_msg]
    )

# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 DataChat AI — Ask Your Spreadsheets")

uploaded_file = st.sidebar.file_uploader("📂 Upload Excel or CSV", type=["xlsx","xls","csv"])
if not uploaded_file:
    st.info("👉 Upload a spreadsheet in the sidebar to get started!")
    st.stop()

# Load DataFrame
file_bytes = uploaded_file.read()
df = get_dataframe(file_bytes, uploaded_file.name)
st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df.head())

# Prepare agent
agent = get_agent(df)

# Prepare month‐matching tools
month_pattern = re.compile(r"^[A-Za-z]+-\d{4}$")
month_map = {
    "January":1, "February":2, "March":3, "April":4,
    "May":5, "June":6, "July":7, "August":8,
    "September":9, "October":10, "November":11, "December":12
}

# Ask loop
query = st.text_input("Ask a question about your data:")
if st.button("🤖 Ask DataChat"):
    q = query.lower()

    # ───── Profitability by month ─────────────────────────────────────────
    if "profitability" in q and "month" in q:
        # find all month columns in df
        month_cols = [c for c in df.columns if isinstance(c,str) and month_pattern.match(c)]
        # sort chronologically
        month_cols = sorted(
            month_cols,
            key=lambda c: (int(c.split("-")[1]), month_map[c.split("-")[0]])
        )
        # compute profitability per month
        data = []
        for m in month_cols:
            rev = df[m].clip(lower=0).sum()
            cost = df[m].clip(upper=0).abs().sum()
            data.append({"Month": m, "Profitability": rev - cost})
        df_profit = pd.DataFrame(data)
        df_profit = df_profit[df_profit["Profitability"] != 0]

        st.subheader("📈 Profitability by Month")
        st.dataframe(df_profit)

        st.bar_chart(df_profit.set_index("Month"))

        # CSV download
        csv_bytes = df_profit.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes,
                           file_name="profit_by_month.csv", mime="text/csv")

        # PDF download
        pdf_buffer = df_to_pdf(df_profit, title="Profitability by Month")
        st.download_button("📥 Download PDF", data=pdf_buffer,
                           file_name="profit_by_month.pdf", mime="application/pdf")

    else:
        # fallback to LLM agent
        with st.spinner("Thinking…"):
            try:
                answer = agent.run(query)
            except Exception as e:
                logging.error("Agent run failed", exc_info=True)
                st.error(f"❌ Error: {e}")
            else:
                st.markdown("**LLM Answer**")
                st.write(answer)
