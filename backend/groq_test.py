import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"GROQ_API_KEY not found. Check this file: {ENV_PATH}"
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": (
                "Reply with exactly: "
                "ShopMind AI LLM connection successful!"
            ),
        }
    ],
)

print(response.choices[0].message.content)