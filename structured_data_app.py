import os
import io
import re
import logging
import streamlit as st
import pandas as pd
import importlib.metadata

# Attempt to import matplotlib for pie charts
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

# 🌐 Optional Sentry init
if st.secrets.get("SENTRY_DSN"):
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    sentry_sdk.init(
        dsn=st.secrets["SENTRY_DSN"],
        integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )

# 🔑 Load OpenAI key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not found in secrets or environment variables.")
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

# Controls
chart_type = st.sidebar.selectbox("📈 Chart type", ["Bar chart", "Line chart"])
export_csv = st.sidebar.checkbox("📄 Enable CSV export", value=True)

if uploaded_file:
    df = get_dataframe(uploaded_file.read(), uploaded_file.name)
    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]}×{df.shape[1]}")
    st.dataframe(df.head())

    agent = get_agent(df)
    query = st.text_input("Ask a question about your data:")

    if st.button("🤖 Ask DataChat"):
        q = query.lower().strip()
        if not q:
            st.warning("Please enter a question.")
            st.stop()
        num_cols = df.select_dtypes(include="number").columns
        cat_col = next((c for c in df.columns if "type" in c.lower()), None)

        # --- Revenue by Month ---
        if ("month" in q and "revenue" in q):
            rev_month = df[num_cols].clip(lower=0).sum()
            rev_month = rev_month[rev_month > 0]
            st.subheader("Total Revenue by Month")
            table = rev_month.rename_axis("Month").reset_index(name="Total")
            st.table(table)
            if export_csv:
                st.download_button("Download CSV", table.to_csv(index=False), "revenue_by_month.csv")
            if chart_type == "Bar chart":
                st.bar_chart(rev_month)
            else:
                st.line_chart(rev_month)

        # --- Category breakdown ---
        elif ("each revenue" in q and cat_col):
            df_pos = df.copy()
            df_pos[num_cols] = df_pos[num_cols].clip(lower=0)
            grp = df_pos.groupby(cat_col)[num_cols].sum().sum(axis=1)
            grp = grp[grp > 0].sort_values(ascending=False)
            st.subheader("Total Revenue by Category")
            st.table(grp.rename_axis(cat_col).reset_index(name="Total"))
            if export_csv:
                st.download_button("Download CSV", grp.to_csv(header=["Total"]), "revenue_by_category.csv")
            # bar/line
            if chart_type == "Bar chart": st.bar_chart(grp)
            else: st.line_chart(grp)
            # pie
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots()
                ax.pie(grp, labels=grp.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("Install matplotlib to see pie chart breakdown.")

        elif ("each cost" in q and cat_col):
            df_neg = df.copy()
            df_neg[num_cols] = df_neg[num_cols].clip(upper=0).abs()
            grp = df_neg.groupby(cat_col)[num_cols].sum().sum(axis=1)
            grp = grp[grp > 0].sort_values(ascending=False)
            st.subheader("Total Cost by Category")
            st.table(grp.rename_axis(cat_col).reset_index(name="Total"))
            if export_csv:
                st.download_button("Download CSV", grp.to_csv(header=["Total"]), "cost_by_category.csv")
            if chart_type == "Bar chart": st.bar_chart(grp)
            else: st.line_chart(grp)
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots()
                ax.pie(grp, labels=grp.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("Install matplotlib to see pie chart breakdown.")

        # --- Downloadable report ---
        elif "download report" in q or "export report" in q:
            # generate a simple CSV of full df
            csv = df.to_csv(index=False)
            st.download_button("Download full dataset as CSV", csv, "report.csv")

        # --- Quarter parsing ---
        elif re.search(r"q[1-4]-\d{4}", q):
            sm = re.search(r"q([1-4])-(\d{4})", q)
            qn, qy = int(sm.group(1)), int(sm.group(2))
            months = {1:(1,2,3),2:(4,5,6),3:(7,8,9),4:(10,11,12)}[qn]
            date_cols = [c for c in df.columns if re.match(rf"[A-Za-z]+-{qy}$", str(c))]
            sel = []
            for c in date_cols:
                mname, year = c.split('-')
                if int(year)==qy and (datetime.datetime.strptime(mname, "%B").month in months): sel.append(c)
            if sel:
                rev = df[sel].clip(lower=0).sum().sum()
                cost = df[sel].clip(upper=0).abs().sum().sum()
                profit = rev - cost
                st.metric(f"Profitability Q{qn}-{qy}", f"{profit:,.2f}")
            else:
                st.error("No data found for that quarter.")

        # --- Rolling window ---
        elif m := re.search(r"last\s+(\d+)\s+months", q):
            n = int(m.group(1))
            # select most recent n date-like cols
            date_cols = [c for c in df.columns if re.match(r"[A-Za-z]+-\d{4}$", str(c))]
            month_order = {m:i+1 for i,m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"]) }
            sorted_cols = sorted(date_cols, key=lambda c:(int(c.split('-')[1]), month_order.get(c.split('-')[0],0)))
            sel = sorted_cols[-n:]
            rev = df[sel].clip(lower=0).sum().sum()
            cost = df[sel].clip(upper=0).abs().sum().sum()
            profit = rev - cost
            st.metric(f"Profitability last {n} months", f"{profit:,.2f}")

        # ... other direct computations unchanged ...
        else:
            with st.spinner("Thinking…"):
                try:
                    ans = agent.run(query)
                    ans = re.sub(r"\.([A-Za-z])", r". \1", ans)
                except Exception as e:
                    logging.error("Agent run failed", exc_info=True)
                    st.error(f"❌ Error: {e}")
                else:
                    st.subheader("LLM Answer")
                    st.write(ans)
else:
    st.info("👉 Upload a spreadsheet to get started!")
