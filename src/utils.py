"""
Utility functions for parsing model outputs.
"""

import re
import json
from typing import Optional, List, Dict, Any


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Extract JSON object from model response.
    Handles both ```json blocks and raw JSON.
    """
    # Try to find JSON in markdown code blocks first
    matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object (last occurrence)
    try:
        start = text.rfind('{')
        if start != -1:
            depth = 0
            for i, c in enumerate(text[start:]):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                if depth == 0:
                    return json.loads(text[start:start + i + 1])
    except json.JSONDecodeError:
        pass
    
    return None


def extract_guess_from_parsed(parsed: Optional[Dict]) -> List[str]:
    """
    Extract the 4-word guess from parsed JSON.
    Handles multiple JSON formats the model might output.
    """
    if not parsed:
        return []
    
    # Check for common keys
    for key in ["group", "words"]:
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    
    # Get first value that's a list of 4 items
    for value in parsed.values():
        if isinstance(value, list) and len(value) == 4:
            return value
    
    # Fallback: first list value
    for value in parsed.values():
        if isinstance(value, list):
            return value
    
    return []


def normalize_words(words: List[str]) -> List[str]:
    """Normalize words to uppercase."""
    return [w.upper().strip() for w in words]


def check_one_away(guess_set: frozenset, solution_groups: Dict[frozenset, str]) -> bool:
    """Check if guess is ONE AWAY (3 correct, 1 wrong) from any solution group."""
    return any(len(guess_set & sol) == 3 for sol in solution_groups.keys())
