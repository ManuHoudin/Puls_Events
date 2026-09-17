import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

# Teste si Qwen fonctionne avec LangChain en utilisant l'API Dashscope

llm = ChatOpenAI(
    model="qwen3.8-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)


response = llm.invoke(
    "Réponds simplement : Bonjour, LangChain fonctionne avec Qwen."
)

print(response.content)