from __future__ import annotations
# minimal HTTP embedding to keep things simple; you can swap with HF later
from litellm import embedding
from typing import List

# default to OpenAI small embed; or set LITE_LLM/OPENROUTER provider via env
EMBED_MODEL = "text-embedding-3-small"

def embed_words(words: List[str]) -> list[list[float]]:
    # LiteLLM normalizes provider; set OPENAI_API_KEY or OPENROUTER_API_KEY
    resp = embedding(model=EMBED_MODEL, input=words)
    # LiteLLM returns {"data":[{"embedding":[...]}...]}
    return [d["embedding"] for d in resp["data"]]
