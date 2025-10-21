from __future__ import annotations
import os
import json
import re
from typing import List
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import SolveResult, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class AnthropicClient:
    """Client for Anthropic Claude models using official SDK."""
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def solve(
        self,
        words: List[str],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs
    ) -> SolveResult:
        """
        Solve Connections puzzle using Claude.
        
        Args:
            words: 16 words to group
            model: Claude model name (e.g., "claude-3-5-sonnet-20241022")
            temperature: Sampling temperature
            max_tokens: Max response tokens
            **kwargs: Additional Anthropic API parameters
            
        Returns:
            SolveResult with groups and rationales
        """
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(words=", ".join(words))
            }],
            **kwargs
        )
        
        raw_text = response.content[0].text
        groups, rationales = self._parse_response(raw_text)
        
        return SolveResult(
            groups=groups,
            rationales=rationales,
            raw_response=raw_text,
            model=model
        )
    
    def _parse_response(self, text: str) -> tuple[List[List[str]], List[str]]:
        """Extract and validate groups and rationales from JSON response."""
        # Extract JSON
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        
        # Validate groups
        groups = data.get("groups", [])
        if not (isinstance(groups, list) and len(groups) == 4):
            raise ValueError(f"Expected 4 groups, got {len(groups)}")
        
        for i, group in enumerate(groups):
            if not (isinstance(group, list) and len(group) == 4):
                raise ValueError(f"Group {i} must have exactly 4 words, got {len(group)}")
            if not all(isinstance(w, str) for w in group):
                raise ValueError(f"All words in group {i} must be strings")
        
        # Get rationales
        rationales = data.get("rationales", [""] * 4)
        if not isinstance(rationales, list):
            rationales = [""] * 4
        while len(rationales) < 4:
            rationales.append("")
        rationales = rationales[:4]
        
        return groups, rationales