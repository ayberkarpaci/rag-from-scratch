"""Generates the LLM answer from the retrieved context."""

import time
from typing import List

import certifi
import httpx
from openai import OpenAI, RateLimitError

from src import config


class Generator:
    """Puts the context and question into the prompt and asks the LLM."""

    def __init__(self, prompt_variant: str = "baseline", max_retries: int = 5):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            http_client=httpx.Client(verify=certifi.where(), timeout=120.0),
        )
        self.model = config.LLM_MODEL
        self.system_prompt = config.SYSTEM_PROMPTS[prompt_variant]
        self.max_retries = max_retries

    @staticmethod
    def build_prompt(question: str, contexts: List[str]) -> str:
        """Numbers the context blocks and joins them with the question."""
        blocks = [f"[{i + 1}] {text}" for i, text in enumerate(contexts)]
        context_section = "\n\n".join(blocks)

        return f"Context:\n{context_section}\n\nQuestion: {question}\n\nAnswer:"

    def generate(self, question: str, contexts: List[str]) -> str:
        """Retries with a growing wait when rate-limited."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.build_prompt(question, contexts)},
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                wait = 30 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)

        return ""
