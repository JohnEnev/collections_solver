# src/core/env.py
from __future__ import annotations
import os
from dotenv import load_dotenv

KNOWN_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "GEMINI_API_KEY",        # alias for GOOGLE API key (AI Studio)
    "GOOGLE_API_KEY",        # AI Studio
    "VERTEXAI_PROJECT",      # Vertex AI
    "VERTEXAI_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
]

def load_env(dotenv_path: str | None = None) -> dict[str, str]:
    """
    Load .env once. Returns a dict of which keys are present (masked).
    """
    load_dotenv(dotenv_path or os.getenv("DOTENV_PATH", ".env"), override=False)
    found = {}
    for k in KNOWN_KEYS:
        v = os.getenv(k)
        if v:
            mask = v[:4] + "…" if len(v) > 4 else "…"
            found[k] = mask
    return found
