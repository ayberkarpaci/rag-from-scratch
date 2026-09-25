"""Checks the connection to the model server and access to each model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import certifi
import httpx
from openai import OpenAI

from src import config


def build_client() -> OpenAI:
    """On networks that inspect TLS the certificate bundle must be given explicitly."""
    http_client = httpx.Client(verify=certifi.where(), timeout=60.0)

    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        http_client=http_client,
    )


def main():
    client = build_client()

    print("LLM test...")
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        print(f"  {config.LLM_MODEL} -> {response.choices[0].message.content}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print("\nEmbedding test...")
    try:
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=["test sentence"],
        )
        print(f"  {config.EMBEDDING_MODEL} -> dimension {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print("\nJudge test...")
    try:
        response = client.chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=200,
            temperature=0.0,
        )
        print(f"  {config.JUDGE_MODEL} -> {response.choices[0].message.content}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
