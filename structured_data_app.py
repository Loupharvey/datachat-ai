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
    layout="wide",
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

# 🌐 Sentry SDK init (optional)
if st.secrets.get("SENTRY_DSN"):
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    sentry_sdk.init(
        dsn="https://706656d5eb7a8fe73aecc1ecfad78a61@o4509464691015680.ingest.us.sentry.io/4509464705499136",
        integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )

# 🔑 Load API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in secrets or env vars.")
    st.stop()

# 🗂 Caching utilities
@st.cache_data
def get_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

@st.cache_resource
def get_agent(df: pd.DataFrame):
    msg = SystemMessage(content=(
        "You are an expert financial analyst. df has positive values = revenues, negative = costs. "
        "When asked for aggregates, produce pandas code in ```python``` and then results & summary."
    ))
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.0)
    return create_pandas_dataframe_agent(
        llm, df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix_messages=[msg]
    )

# 🖥️ App UI
st.title("💬 DataChat AI — Ask Your Spreadsheets")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel or CSV", type=["csv","xls","xlsx"])

# Chart & export controls
chart_type = st.sidebar.selectbox("📈 Chart type (after query)", ["Line chart", "Bar chart"])
export_csv = st.sidebar.checkbox("📄 Enable CSV export of results", value=True)

if uploaded_file:
    df = get_dataframe(uploaded_file.read(), uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
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

            # --- Direct Pandas computations ---
            # revenue by month
            if ("each month" in q or "by each month" in q) and "revenue" in q:
                rev_by_month = df[num_cols].clip(lower=0).sum()
                rev_by_month = rev_by_month[rev_by_month > 0]
                st.subheader("Total Revenue by Month")
                table = rev_by_month.rename_axis("Month").reset_index(name="Total")
                st.table(table)
                if export_csv:
                    csv = table.to_csv(index=False)
                    st.download_button("Download CSV", csv, file_name="revenue_by_month.csv")
                chart_df = rev_by_month.to_frame(name="Total")
                if chart_type == "Bar chart":
                    st.bar_chart(chart_df)
                else:
                    st.line_chart(chart_df)

            # revenues per category
            elif ("each revenue" in q or "for each revenue" in q) and cat_col:
                df_pos = df.copy()
                df_pos[num_cols] = df_pos[num_cols].clip(lower=0)
                grp = df_pos.groupby(cat_col)[num_cols].sum().sum(axis=1)
                grp = grp[grp>0].sort_values(ascending=False)
                st.subheader("Total Revenue by Category")
                st.table(grp.rename_axis(cat_col).reset_index(name="Total"))
                if export_csv:
                    csv = grp.to_csv(header=["Total"])
                    st.download_button("Download CSV", csv, file_name="revenue_by_category.csv")
                chart_df = grp.to_frame(name="Total").sort_values("Total", ascending=False)
                if chart_type == "Bar chart": st.bar_chart(chart_df)
                else: st.line_chart(chart_df)

            # costs per category
            elif ("each cost" in q or "for each cost" in q) and cat_col:
                df_neg = df.copy()
                df_neg[num_cols] = df_neg[num_cols].clip(upper=0).abs()
                grp = df_neg.groupby(cat_col)[num_cols].sum().sum(axis=1)
                grp = grp[grp>0].sort_values(ascending=False)
                st.subheader("Total Cost by Category")
                st.table(grp.rename_axis(cat_col).reset_index(name="Total"))
                if export_csv:
                    csv = grp.to_csv(header=["Total"])
                    st.download_button("Download CSV", csv, file_name="cost_by_category.csv")
                chart_df = grp.to_frame(name="Total")
                if chart_type == "Bar chart": st.bar_chart(chart_df)
                else: st.line_chart(chart_df)

            # total revenue year
            elif "total revenue" in q:
                total_rev = df[num_cols].clip(lower=0).sum().sum()
                st.metric("Total Revenue (Year)", f"{total_rev:,.2f}")

            # total cost year
            elif "total cost" in q:
                total_cost = df[num_cols].clip(upper=0).abs().sum().sum()
                st.metric("Total Cost (Year)", f"{total_cost:,.2f}")

            # profitability queries
            elif "profitability" in q:
                # parse quarter
                q_match = re.search(r"q([1-4])-(\d{4})", q)
                if q_match:
                    qn, qy = int(q_match.group(1)), int(q_match.group(2))
                    months = {1:(1,2,3),2:(4,5,6),3:(7,8,9),4:(10,11,12)}[qn]
                    date_cols = [c for c in df.columns if re.match(rf"[A-Za-z]+-{qy}", c)]
                    sel = [c for c in date_cols if int(c.split('-')[1])==qy and int(c.split('-')[0].lower().translate(str.maketrans({m:str(i+1) for i,m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"])}))) in months]
                else:
                    # fallback to regex month-year
                    tokens = re.findall(r"([A-Za-z]+-\d{4})", q)
                    sel = []
                    if len(tokens)==2:
                        # reuse multi-month slicing logic
                        # (similar to revenue_by_month selection)
                        pattern = re.compile(r"^[A-Za-z]+-\d{4}$")
                        date_cols = [c for c in df.columns if pattern.match(str(c))]
                        month_map = {m:i+1 for i,m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"])}
                        sorted_cols = sorted(date_cols, key=lambda c:(int(c.split('-')[1]), month_map.get(c.split('-')[0].lower(),0)))
                        start, end = tokens[0].lower(), tokens[1].lower()
                        lc_map = {c.lower():c for c in sorted_cols}
                        if start in lc_map and end in lc_map:
                            i1, i2 = sorted_cols.index(lc_map[start]), sorted_cols.index(lc_map[end])
                            sel = sorted_cols[min(i1,i2):max(i1,i2)+1]
                if sel:
                    rev = df[sel].clip(lower=0).sum().sum()
                    cost = df[sel].clip(upper=0).abs().sum().sum()
                    profit = rev - cost
                    st.metric(f"Profitability ({', '.join(sel)})", f"{profit:,.2f}")
                else:
                    rev = df[num_cols].clip(lower=0).sum().sum()
                    cost = df[num_cols].clip(upper=0).abs().sum().sum()
                    profit = rev - cost
                    st.metric("Profitability (Year)", f"{profit:,.2f}")

            # fallback to LLM
            else:
                with st.spinner("Thinking…"):
                    try:
                        answer = agent.run(query)
                        answer = re.sub(r"\.([A-Za-z])", r". \1", answer)
                    except Exception as e:
                        logging.error("Agent run failed", exc_info=True)
                        st.error(f"❌ Error: {str(e)}")
                    else:
                        st.subheader("LLM Answer")
                        st.write(answer)
else:
    st.info("👉 Upload a spreadsheet to get started!")
