from __future__ import annotations
import os
import json
import re
from typing import List, Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import SolveResult, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class OpenAIClient:
    """Client for OpenAI models using official SDK."""
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def solve(
        self, 
        words: List[str], 
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs
    ) -> SolveResult:
        """
        Solve Connections puzzle using OpenAI model.
        
        Args:
            words: 16 words to group
            model: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Sampling temperature
            max_tokens: Max response tokens
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            SolveResult with groups and rationales
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(words=", ".join(words))}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  # Force JSON mode
            **kwargs
        )
        
        raw_text = response.choices[0].message.content
        groups, rationales = self._parse_response(raw_text)
        
        return SolveResult(
            groups=groups,
            rationales=rationales,
            raw_response=raw_text,
            model=model
        )
    
    def _parse_response(self, text: str) -> tuple[List[List[str]], List[str]]:
        """Extract and validate groups and rationales from JSON response."""
        # Extract JSON (defensive in case model adds extra text)
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        
        # Validate groups structure
        groups = data.get("groups", [])
        if not (isinstance(groups, list) and len(groups) == 4):
            raise ValueError(f"Expected 4 groups, got {len(groups)}")
        
        for i, group in enumerate(groups):
            if not (isinstance(group, list) and len(group) == 4):
                raise ValueError(f"Group {i} must have exactly 4 words, got {len(group)}")
            if not all(isinstance(w, str) for w in group):
                raise ValueError(f"All words in group {i} must be strings")
        
        # Get rationales (optional, default to empty strings)
        rationales = data.get("rationales", [""] * 4)
        if not isinstance(rationales, list):
            rationales = [""] * 4
        while len(rationales) < 4:
            rationales.append("")
        rationales = rationales[:4]
        
        return groups, rationales