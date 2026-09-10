import os, traceback
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

try:
    response = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print(response.choices[0].message.content)
except Exception:
    traceback.print_exc()