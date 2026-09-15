import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="qwen3.8-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)


@tool
def rechercher_evenements(question: str) -> str:
    """Recherche des événements culturels dans la base."""

    print(">>> TOOL APPELÉ")
    print(">>> QUESTION :", question)

    return "Concert de musique classique à Vannes le samedi 12 septembre à 20h."


llm_with_tools = llm.bind_tools([rechercher_evenements])


response = llm_with_tools.invoke(
    "Je cherche un concert de musique classique à Vannes."
)

print("\n>>> CONTENU :")
print(response.content)

print("\n>>> TOOL CALLS :")
print(response.tool_calls)