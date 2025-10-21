from __future__ import annotations
from typing import Protocol, List, Dict, Any
from dataclasses import dataclass

@dataclass
class SolveResult:
    """Standardized result from any model client."""
    groups: List[List[str]]      # 4 groups of 4 words each
    rationales: List[str]         # 4 rationales (one per group)
    raw_response: str            # Raw model output for debugging
    model: str                   # Model identifier used


class BaseClient(Protocol):
    """Protocol defining the interface all model clients must implement."""
    
    def solve(self, words: List[str], model: str, **kwargs) -> SolveResult:
        """
        Solve a Connections puzzle.
        
        Args:
            words: List of 16 words to group
            model: Model identifier (client-specific format)
            **kwargs: Additional model-specific parameters
            
        Returns:
            SolveResult with groups, rationales, and metadata
            
        Raises:
            ValueError: If response is malformed
            RuntimeError: If API call fails
        """
        ...


# Shared prompts used across all clients
SYSTEM_PROMPT = (
    "You solve NYT Connections. Given 16 words, return 4 disjoint groups of 4.\n"
    "Return STRICT JSON only, no extra text. Provide short, non-revealing rationales "
    "for each group (one sentence). Do NOT reveal chain-of-thought."
)

USER_PROMPT_TEMPLATE = """Words (16):
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