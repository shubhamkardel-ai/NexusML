import os
import json

from dotenv import load_dotenv
from groq import Groq

from services.copilot_context import build_copilot_context

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


client = Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """
You are NexusML Copilot, an AI assistant for a production
machine learning lifecycle and model reliability platform.

Your job is to help engineers understand the current state
of NexusML.

IMPORTANT RULES:

1. Use the provided NexusML context as the source of truth.
2. Do not invent metrics, model versions, experiments,
   drift results, or deployment information.
3. If the context does not contain enough information,
   clearly say that the information is unavailable.
4. Explain MLOps decisions in simple but technically accurate
   language.
5. When discussing model promotion or retraining, distinguish
   between the current production model and candidate models.
6. Never reveal API keys, credentials, or secrets.
7. Keep answers concise unless the user asks for more detail.

You are an engineering assistant, not an autonomous deployment
agent. Do not claim that you changed production unless an
explicit tool/action actually performed that operation.
"""


def ask_copilot(question: str) -> str:
    context = build_copilot_context()

    context_json = json.dumps(
        context,
        indent=2,
        default=str,
    )

    user_prompt = f"""
NexusML operational context:

{context_json}

User question:

{question}

Answer the user's question using only the NexusML context
provided above.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    except Exception as error:
        raise RuntimeError(
            f"Copilot LLM request failed: {error}"
        ) from error


if __name__ == "__main__":
    print("=" * 70)
    print("NexusML — LLM Copilot Test")
    print("=" * 70)

    questions = [
        "What model is currently in production?",
        "Why was the latest candidate not promoted?",
        "What is the current production F1 score?",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        print(f"Copilot: {ask_copilot(question)}")

    print("\n" + "=" * 70)