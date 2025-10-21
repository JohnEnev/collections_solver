#!/usr/bin/env python3
"""Generate synthetic NYT Connections puzzles using LLMs with LLM-as-Judge validation."""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import your model clients  
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models import solve_with_model

app = typer.Typer()
console = Console()


GENERATOR_PROMPT = """You are an expert at creating NYT Connections puzzles.

Create a NEW, ORIGINAL puzzle with these requirements:
1. Exactly 16 unique words
2. Exactly 4 groups of 4 words each
3. Groups should have DIVERSE thematic connections (avoid common themes like "types of fruit" unless necessary)
4. Include some "red herrings" (words that seem related but aren't in same group)
5. Vary difficulty across groups (level 0=easiest, level 3=hardest)

IMPORTANT: Create CREATIVE and UNIQUE themes! Avoid overly simple categories.

Difficulty guidelines:
- Level 0 (easiest): Straightforward categories (types of fruit, colors, countries)
- Level 1 (easy-medium): Common associations (things in a kitchen, sports terms)
- Level 2 (medium-hard): Requires more thought (words that follow X, homophones)
- Level 3 (hardest): Tricky connections (wordplay, obscure trivia, misleading)

Example themes (USE DIFFERENT ONES):
- Oscar-winning movies
- Words that can follow "FIRE"
- U.S. state capitals
- NBA teams
- Synonyms for specific emotions
- Things with wheels
- Words ending in specific suffixes
- Homophones of body parts
- Collective nouns (murder, pride, school)
- Words that precede "WOOD"
- Greek gods
- Programming languages
- Dance styles
- Chemical elements
- Harry Potter characters

Return ONLY valid JSON (no other text):
{{
  "words": ["WORD1", "WORD2", ..., "WORD16"],
  "groups": [
    {{
      "members": ["W1", "W2", "W3", "W4"],
      "connection": "Theme description",
      "level": 0,
      "reasoning": "One brief sentence (max 15 words) explaining the connection"
    }},
    {{
      "members": ["W5", "W6", "W7", "W8"],
      "connection": "Theme description",
      "level": 1,
      "reasoning": "One brief sentence (max 15 words)"
    }},
    {{
      "members": ["W9", "W10", "W11", "W12"],
      "connection": "Theme description",
      "level": 2,
      "reasoning": "One brief sentence (max 15 words)"
    }},
    {{
      "members": ["W13", "W14", "W15", "W16"],
      "connection": "Theme description",
      "level": 3,
      "reasoning": "One brief sentence (max 15 words)"
    }}
  ]
}}"""


JUDGE_PROMPT = """You are a quality judge for NYT Connections puzzles.

Evaluate this puzzle on a scale of 1-10 for:
1. **Validity**: Are the connections logical and correct?
2. **Clarity**: Are group themes clear and unambiguous?
3. **Difficulty**: Is there good variation in difficulty?
4. **Red Herrings**: Are there plausible but incorrect groupings?
5. **Uniqueness**: Is this puzzle interesting and original?

Puzzle:
{puzzle}

Return ONLY valid JSON:
{{
  "score": <1-10>,
  "validity": <1-10>,
  "clarity": <1-10>,
  "difficulty_balance": <1-10>,
  "red_herrings": <1-10>,
  "uniqueness": <1-10>,
  "feedback": "Brief explanation of score",
  "accept": <true/false>
}}

A puzzle is accepted if score >= 7."""


def generate_puzzle(generator_model: str = "gpt4o", used_themes: set = None, temperature: float = 0.9) -> Dict[str, Any]:
    """Generate a synthetic puzzle using an LLM with diversity enforcement."""
    try:
        # Use a direct API call with JSON mode
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        client = OpenAI(api_key=api_key)
        
        # Build diversity prompt if we have used themes
        diversity_note = ""
        if used_themes:
            # Filter to only string themes (not tuple combinations)
            string_themes = [t for t in used_themes if isinstance(t, str)]
            if string_themes:
                diversity_note = f"\n\nIMPORTANT: Create UNIQUE themes. AVOID these already-used themes: {', '.join(list(string_themes)[:10])}"
        
        prompt = GENERATOR_PROMPT + diversity_note
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # Higher temp for more diversity
            response_format={"type": "json_object"}
        )
        
        puzzle_str = response.choices[0].message.content
        puzzle = json.loads(puzzle_str)
        
        # Validate structure
        assert "words" in puzzle and len(puzzle["words"]) == 16
        assert "groups" in puzzle and len(puzzle["groups"]) == 4
        assert all(len(g["members"]) == 4 for g in puzzle["groups"])
        
        return puzzle
    
    except Exception as e:
        console.print(f"[red]Error generating puzzle: {e}[/]")
        return None


def judge_puzzle(puzzle: Dict[str, Any], judge_model: str = "sonnet") -> Dict[str, Any]:
    """Judge puzzle quality using an LLM."""
    try:
        # Try Anthropic first
        if os.getenv("ANTHROPIC_API_KEY") and "claude" in judge_model.lower():
            from anthropic import Anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key)
            
            prompt = JUDGE_PROMPT.format(puzzle=json.dumps(puzzle, indent=2))
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            judgment_str = response.content[0].text
        
        # Fall back to OpenAI
        else:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"accept": True, "score": 8, "feedback": "No judge available"}
            
            client = OpenAI(api_key=api_key)
            
            prompt = JUDGE_PROMPT.format(puzzle=json.dumps(puzzle, indent=2))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            judgment_str = response.choices[0].message.content
        
        # Extract JSON
        import re
        match = re.search(r"\{.*\}", judgment_str, flags=re.S)
        if match:
            judgment = json.loads(match.group(0))
            return judgment
        else:
            return {"accept": False, "feedback": "Failed to parse judgment"}
    
    except Exception as e:
        console.print(f"[red]Error judging puzzle: {e}[/]")
        return {"accept": False, "feedback": str(e)}


def add_adversarial_examples(puzzle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create adversarial training examples from a valid puzzle.
    
    IMPORTANT: We DON'T create broken puzzles with wrong answers!
    Instead, we create HARD puzzles with ambiguous words.
    
    Returns list of adversarial puzzles (could be empty if no good adversarial version).
    """
    adversarial_puzzles = []
    
    # Strategy 1: Create a puzzle with one "tempting but wrong" word
    # This requires the LLM to generate puzzles with ambiguous words
    # For now, we'll skip this and let the LLM naturally create hard puzzles
    
    # Strategy 2: Ask for a harder variant
    # We can prompt the LLM to create a variant with more red herrings
    
    # For now, return empty list - adversarial examples should come from
    # real data where we KNOW the correct answers
    return adversarial_puzzles


@app.command()
def main(
    n_puzzles: int = typer.Option(50, help="Number of puzzles to generate"),
    output_path: str = "data/training/synthetic_puzzles.jsonl",
    generator_model: str = "gpt4o",
    judge_model: str = "sonnet",
    min_score: float = 7.0,
    skip_judge: bool = typer.Option(False, help="Skip LLM judge (accept all)"),
    seed: int = 42,
):
    """
    Generate synthetic training data using LLM generation + LLM-as-judge.
    
    Process:
    1. Generator LLM creates puzzle with difficulty levels (0-3)
    2. Judge LLM evaluates quality (score 1-10) [optional]
    3. If score >= min_score, accept puzzle
    4. Save to training data format matching real data structure
    
    Example:
        python generate_synthetic_puzzles.py --n-puzzles 100
        python generate_synthetic_puzzles.py --n-puzzles 50 --skip-judge  # No judge
    """
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if not has_openai:
        console.print("[red]Error: OPENAI_API_KEY not set![/]")
        console.print("Set it in your .env file or environment")
        raise typer.Exit(1)
    
    # Determine judge
    if skip_judge:
        judge_info = "DISABLED"
    elif has_anthropic and "claude" in judge_model.lower():
        judge_info = f"{judge_model} (Anthropic)"
    else:
        judge_info = "gpt-4o (OpenAI fallback)"
        if not has_anthropic:
            console.print("[yellow]Note: Using GPT-4o as judge (Anthropic key not set)[/]")
    
    console.rule("[bold cyan]Synthetic Puzzle Generation")
    console.print(f"Target: {n_puzzles} puzzles")
    console.print(f"Generator: {generator_model}")
    console.print(f"Judge: {judge_info}")
    console.print(f"Min score: {min_score if not skip_judge else 'N/A'}")
    console.print("")
    
    accepted_puzzles = []
    rejected_puzzles = []
    used_themes = set()  # Track themes to ensure diversity
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task(f"Generating puzzles...", total=n_puzzles)
        
        attempts = 0
        max_attempts = n_puzzles * 3  # Allow 3x attempts
        
        while len(accepted_puzzles) < n_puzzles and attempts < max_attempts:
            attempts += 1
            
            # Vary temperature for more diversity
            temp = 0.8 + (attempts % 3) * 0.1  # Cycles through 0.8, 0.9, 1.0
            
            # Generate puzzle with diversity enforcement
            progress.update(task, description=f"Generating ({len(accepted_puzzles)}/{n_puzzles})...")
            puzzle = generate_puzzle(generator_model, used_themes, temperature=temp)
            
            if puzzle is None:
                continue
            
            # Judge puzzle (or auto-accept if skipping)
            if skip_judge:
                judgment = {"accept": True, "score": 8, "feedback": "Auto-accepted (no judge)"}
            else:
                progress.update(task, description=f"Judging ({len(accepted_puzzles)}/{n_puzzles})...")
                judgment = judge_puzzle(puzzle, judge_model)
            
            if judgment.get("accept", False) and judgment.get("score", 0) >= min_score:
                # Check for duplicate themes (simple deduplication)
                puzzle_themes = tuple(sorted([g.get("connection", "").lower() for g in puzzle["groups"]]))
                
                # Skip if we've seen very similar themes
                if puzzle_themes in used_themes:
                    console.print(f"[yellow]Skipping duplicate theme combination[/]")
                    continue
                
                # Accept puzzle
                used_themes.add(puzzle_themes)
                
                # Also track individual themes
                for g in puzzle["groups"]:
                    theme = g.get("connection", "").lower()
                    used_themes.add(theme)
                
                accepted_puzzles.append({
                    "puzzle": puzzle,
                    "judgment": judgment,
                })
                
                progress.update(task, advance=1)
            else:
                rejected_puzzles.append({
                    "puzzle": puzzle,
                    "judgment": judgment
                })
    
    # Convert to training format (matching real data structure)
    training_examples = []
    
    for item in accepted_puzzles:
        puzzle = item["puzzle"]
        
        # Sort groups by difficulty level (0=easiest to 3=hardest)
        sorted_groups = sorted(puzzle["groups"], key=lambda g: g.get("level", 0))
        
        # Build CONCISE reasoning (max 50 words total)
        # Just mention the approach, not every detail
        easiest = sorted_groups[0]["connection"]
        hardest = sorted_groups[3]["connection"]
        
        reasoning = f"Starting with {easiest}, then working through harder groups, ending with {hardest}."
        
        # Create training example in Alpaca format with CoT
        example = {
            "instruction": (
                "You solve NYT Connections puzzles. Given 16 words, partition them into exactly 4 groups of 4 words each. "
                "Each word must be used exactly once. First explain your reasoning, then return strict JSON with 4 groups and their rationales."
            ),
            "input": f"Words: {', '.join(puzzle['words'])}",
            "output": json.dumps({
                "reasoning": reasoning,
                "groups": [g["members"] for g in sorted_groups],
                "rationales": [g["connection"] for g in sorted_groups]
            }, indent=2),
            "metadata": {
                "synthetic": True,
                "score": item["judgment"].get("score", 0),
                "difficulty_levels": [g.get("level", 0) for g in sorted_groups]
            }
        }
        training_examples.append(example)
    
    # Save
    with open(output_path, "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Summary
    console.rule("[bold green]Generation Complete")
    console.print(f"✓ Generated: {len(accepted_puzzles)} puzzles")
    console.print(f"✗ Rejected: {len(rejected_puzzles)} puzzles")
    console.print(f"📊 Acceptance rate: {len(accepted_puzzles)/attempts*100:.1f}%")
    console.print(f"🎨 Unique themes: {len([t for t in used_themes if isinstance(t, str)])}")
    console.print(f"\n💾 Saved to: {output_path}")
    
    # Show difficulty distribution
    if training_examples and "difficulty_levels" in training_examples[0]["metadata"]:
        all_levels = [level for ex in training_examples for level in ex["metadata"]["difficulty_levels"]]
        from collections import Counter
        level_counts = Counter(all_levels)
        console.print(f"\n📊 Difficulty distribution:")
        for level in sorted(level_counts.keys()):
            console.print(f"  Level {level}: {level_counts[level]} groups")
    
    # Show sample themes (first 10)
    theme_samples = [t for t in used_themes if isinstance(t, str)][:10]
    if theme_samples:
        console.print(f"\n🎭 Sample themes generated:")
        for theme in theme_samples:
            console.print(f"  • {theme}")
    
    # Show sample
    if training_examples:
        console.print("\n[cyan]Sample generated puzzle:[/]")
        sample = training_examples[0]
        console.print(f"[bold]Input:[/] {sample['input'][:100]}...")
        console.print(f"[bold]Output:[/] {sample['output'][:150]}...")
        console.print(f"[bold]Score:[/] {sample['metadata']['score']}")


if __name__ == "__main__":
    app()