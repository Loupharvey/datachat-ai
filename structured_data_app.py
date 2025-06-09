import os
import io
import logging
import re
import streamlit as st
import pandas as pd
import importlib.metadata
from fpdf import FPDF

# 1) Page config
st.set_page_config(page_title="📊 DataChat AI", layout="centered")

# 2) Simple password auth
PASSWORD = st.secrets.get("PASSWORD", "")
if not PASSWORD:
    st.error("App password not set in secrets.")
    st.stop()
pw = st.sidebar.text_input("🔒 App Password", type="password")
if pw != PASSWORD:
    st.sidebar.error("❌ Incorrect password")
    st.stop()

# 3) (Optional) Sentry init
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
    logger = logging.getLogger(__name__)
    logger.info("Sentry initialized")

# 4) Load OpenAI key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

# 5) ChatOpenAI fallback import
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ──────────────────────────────────────────────────────────────────
# PDF helper for export
def df_to_pdf(df: pd.DataFrame, title: str = "Report") -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    col_width = pdf.epw / len(df.columns)
    # Header
    for col in df.columns:
        pdf.cell(col_width, 8, col, border=1, align="C")
    pdf.ln()
    # Rows
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, 8, str(item), border=1)
        pdf.ln()
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# 6) Caching helpers
@st.cache_data
def get_dataframe(bytes_data: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(bytes_data)
    if name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(bio)
    return pd.read_csv(bio)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, model="gpt-4")
    system_msg = {
        "role":"system",
        "content":(
            "You are an expert data analyst. Summarize the DataFrame, then answer the user's question. "
            "When needed, generate Python/pandas code."
        )
    }
    return create_pandas_dataframe_agent(
        llm, df, verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix_messages=[system_msg]
    )

# ──────────────────────────────────────────────────────────────────
st.title("💬 DataChat AI — Ask Your Spreadsheets")

# File uploader
uploaded = st.sidebar.file_uploader("📂 Upload Excel or CSV", type=["csv","xls","xlsx"])
if not uploaded:
    st.info("Upload a spreadsheet to get started.")
    st.stop()

# Load and show DataFrame
data_bytes = uploaded.read()
df = get_dataframe(data_bytes, uploaded.name)
st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df.head())

# Prepare agent
agent = get_agent(df)

# Prepare month regex & map
month_pattern = re.compile(r"^[A-Za-z]+-\d{4}$")
month_map = {
  "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
  "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}

# Ask loop
query = st.text_input("Ask a question about your data:")
if st.button("🤖 Ask DataChat"):
    q = query.lower()

    # ─── Profitability by Month branch ─────────────────────────────────
    if "profitability" in q and "month" in q:
        # 1) Identify month columns
        month_cols = [c for c in df.columns
                      if isinstance(c,str) and month_pattern.match(c)]
        # 2) Sort chronologically
        month_cols = sorted(
            month_cols,
            key=lambda c: (int(c.split("-")[1]), month_map[c.split("-")[0]])
        )
        # 3) Build results
        rows = []
        for m in month_cols:
            rev  = df[m].clip(lower=0).sum()
            cost = df[m].clip(upper=0).abs().sum()
            prof = rev - cost
            if prof != 0:
                rows.append({"Month": m, "Profitability": prof})
        df_out = pd.DataFrame(rows)

        # 4) Display table & chart
        st.subheader("📈 Profitability by Month")
        st.dataframe(df_out)
        st.bar_chart(df_out.set_index("Month"))

        # 5) CSV export
        csv_data = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="profitability_by_month.csv",
            mime="text/csv"
        )

        # 6) PDF export
        pdf_buf = df_to_pdf(df_out, title="Profitability by Month")
        st.download_button(
            "📥 Download PDF",
            data=pdf_buf,
            file_name="profitability_by_month.pdf",
            mime="application/pdf"
        )

    else:
        # Fallback to LLM
        with st.spinner("Thinking…"):
            try:
                answer = agent.run(query)
            except Exception as e:
                logging.error("Agent failed", exc_info=True)
                st.error(f"❌ {e}")
            else:
                st.markdown("**LLM Answer**")
                st.write(answer)