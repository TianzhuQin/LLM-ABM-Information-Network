import os
from typing import List, Dict

from openai import OpenAI


def load_api_key() -> str:
    key = os.getenv("OPENAI_OPENAI_API_KEY", "").strip() if False else os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        key_file = os.getenv("OPENAI_API_KEY_FILE", "").strip() or "api-key.txt"
        if os.path.exists(key_file):
            with open(key_file) as f:
                line = f.readline().strip()
                if line:
                    key = line
    if not key and os.path.exists("api-key.txt"):
        with open("api-key.txt") as f:
            line = f.readline().strip()
            if line:
                key = line
    if not key or len(key) < 20:
        raise RuntimeError("Missing OPENAI_API_KEY; set env var or add api-key*.txt")
    return key


def build_client() -> OpenAI:
    api_key = load_api_key()
    return OpenAI(api_key=api_key)


def call_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens_requested: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens_requested,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Empty response (chat.completions)")
    return content





