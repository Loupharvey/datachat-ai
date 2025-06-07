import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("No OPENAI_API_KEY in .env")

print("🧭 Current working directory:", os.getcwd())
df = pd.read_csv("sales.csv")

llm = ChatOpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

while True:
    query = input("\nAsk about your data (or 'exit'): ")
    if query.lower() == "exit":
        break
    print("\n📊 Response:", agent.run(query))
