import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen3.8-max",
    messages=[
        {
            "role": "user",
            "content": "Réponds simplement : Bonjour, Qwen fonctionne.",
        }
    ],
)

print(response.choices[0].message.content)