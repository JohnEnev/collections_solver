from __future__ import annotations

from typing import List, Dict, Any
import json, re
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM = (
    "You solve NYT Connections. Given 16 words, return 4 disjoint groups of 4.\n"
    "Return STRICT JSON only, no extra text. Provide short, non-revealing rationales "
    "for each group (one sentence). Do NOT reveal chain-of-thought."
)

PROMPT = """Words (16):
{words}

Return STRICT JSON only:
{{
  "groups": [
    ["w1","w2","w3","w4"],
    ["w5","w6","w7","w8"],
    ["w9","w10","w11","w12"],
    ["w13","w14","w15","w16"]
  ],
  "rationales": [
    "≤20 words on the common theme.",
    "…",
    "…",
    "…"
  ]
}}
"""

def _provider(model: str) -> str:
    if model.startswith("anthropic/"):
        return "anthropic"
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("gemini/"):      # Google AI Studio
        return "gemini"
    if model.startswith("vertex_ai/"):   # Vertex AI
        return "vertex_ai"
    if model.startswith("openrouter/"):
        return "openrouter"
    return "other"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def solve_with_llm(words: List[str], model: str = "openai/gpt-4o") -> Dict[str, Any]:
    """
    Call a model via LiteLLM and return:
      { "groups": list[list[str]] (4x4), "rationales": list[str] (len 4), "raw": str }
    Raises on invalid/missing JSON; retries on transient errors.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(words=", ".join(words))},
    ]

    prov = _provider(model)
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=0.2,
        timeout=120,
    )

    # Provider-specific sane defaults
    if prov == "anthropic":
        kwargs["max_tokens"] = 512
    elif prov == "openai":
        kwargs["max_tokens"] = 512
        kwargs["response_format"] = {"type": "json_object"}
    elif prov == "gemini":
        # Gemini (AI Studio) prefers this for structured JSON
        kwargs["max_tokens"] = 1024
        kwargs["response_mime_type"] = "application/json"
    elif prov == "vertex_ai":
        # Vertex also supports response_mime_type
        kwargs["max_tokens"] = 1024
        kwargs["response_mime_type"] = "application/json"

    try:
        resp = completion(**kwargs)
    except Exception as e:
        detail = getattr(e, "message", None) or getattr(e, "body", None) or str(e)
        raise RuntimeError(f"Provider error calling {model}: {detail}") from e

    txt = resp["choices"][0]["message"]["content"]

    # Extract JSON from the response defensively
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError(f"Missing JSON in model response: {txt[:200]}")

    obj = json.loads(m.group(0))

    groups = obj.get("groups", [])
    if not (isinstance(groups, list) and len(groups) == 4 and all(isinstance(g, list) and len(g) == 4 for g in groups)):
        raise ValueError(f"Bad groups format: {obj}")

    rats = obj.get("rationales", [""] * 4)
    if not (isinstance(rats, list) and len(rats) == 4):
        rats = [""] * 4

    return {"groups": groups, "rationales": rats, "raw": txt}
