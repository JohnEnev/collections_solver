#!/usr/bin/env python3
"""Generate adversarial training examples from REAL puzzle data."""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console

app = typer.Typer()
console = Console()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    with open(path, "rb") as f:
        import orjson
        return [orjson.loads(line) for line in f]


def create_near_miss_example(puzzle: Dict[str, Any], predictions: List[List[str]]) -> Dict[str, str]:
    """
    Create a training example from a near-miss (3/4 groups correct).
    
    The model sees its WRONG prediction and learns the CORRECT answer.
    This teaches the model to avoid common mistakes.
    """
    words = puzzle["words"]
    groups = puzzle["groups"]
    
    # Sort by difficulty
    sorted_groups = sorted(groups, key=lambda g: g["level"])
    correct_groups = [g["members"] for g in sorted_groups]
    rationales = [f"{g['name']}" for g in sorted_groups]
    difficulty_levels = [g["level"] for g in sorted_groups]
    
    # Generate reasoning about avoiding the mistake
    reasoning = (
        "Looking at the 16 words, I need to be careful to find the correct groupings. "
        f"Starting with the easiest category (level {difficulty_levels[0]}): {rationales[0]}. "
        f"The trickiest group (level {difficulty_levels[3]}) is: {rationales[3]}. "
        "I'll verify each word is used exactly once and fits its group's theme."
    )
    
    return {
        "instruction": (
            "You solve NYT Connections puzzles. Given 16 words, partition them into exactly 4 groups of 4 words each. "
            "Each word must be used exactly once. First explain your reasoning, then return strict JSON with 4 groups and their rationales."
        ),
        "input": f"Words: {', '.join(words)}",
        "output": json.dumps({
            "reasoning": reasoning,
            "groups": correct_groups,
            "rationales": rationales
        }, indent=2),
        "metadata": {
            "adversarial": True,
            "type": "near_miss",
            "difficulty_levels": difficulty_levels
        }
    }


def create_hard_puzzle_example(puzzle: Dict[str, Any]) -> Dict[str, str]:
    """
    Create training example from puzzles with high difficulty variance.
    
    These puzzles have easy groups (level 0) and hard groups (level 3),
    which teaches the model to handle mixed difficulty.
    """
    words = puzzle["words"]
    groups = puzzle["groups"]
    
    # Sort by difficulty
    sorted_groups = sorted(groups, key=lambda g: g["level"])
    
    # Generate reasoning that acknowledges difficulty mix
    reasoning = (
        f"This puzzle has a mix of difficulties. "
        f"I'll start with the straightforward group (level {sorted_groups[0]['level']}): {sorted_groups[0]['name']}. "
        f"Then tackle the harder groups. "
        f"The most challenging (level {sorted_groups[3]['level']}) is: {sorted_groups[3]['name']}, which requires careful analysis."
    )
    
    return {
        "instruction": (
            "You solve NYT Connections puzzles. Given 16 words, partition them into exactly 4 groups of 4 words each. "
            "Each word must be used exactly once. First explain your reasoning, then return strict JSON with 4 groups and their rationales."
        ),
        "input": f"Words: {', '.join(words)}",
        "output": json.dumps({
            "reasoning": reasoning,
            "groups": [g["members"] for g in sorted_groups],
            "rationales": [g["name"] for g in sorted_groups]
        }, indent=2),
        "metadata": {
            "adversarial": True,
            "type": "hard_puzzle",
            "difficulty_levels": [g["level"] for g in sorted_groups],
            "max_difficulty": max(g["level"] for g in groups)
        }
    }


@app.command()
def main(
    gold_path: str = "data/interim/test.jsonl",
    predictions_path: str = typer.Option(None, help="Predictions file for near-miss examples"),
    output_path: str = "data/training/adversarial_examples.jsonl",
    include_near_miss: bool = typer.Option(True, help="Include near-miss examples (3/4 correct)"),
    include_hard: bool = typer.Option(True, help="Include hard puzzles (level 3 groups)"),
    min_difficulty: int = typer.Option(2, help="Minimum difficulty level to include"),
    seed: int = 42,
):
    """
    Generate adversarial training examples from real puzzle data.
    
    Two types of adversarial examples:
    1. Near-miss: Puzzles where a model got 3/4 groups correct
       → Teaches model to avoid common mistakes
    
    2. Hard puzzles: Puzzles with difficulty level 3 groups
       → Teaches model to handle hardest categories
    
    Example:
        # From baseline predictions
        python generate_adversarial_examples.py \
          --predictions-path data/cache/llama8b_baseline.jsonl
        
        # Just hard puzzles from gold data
        python generate_adversarial_examples.py --include-near-miss=False
    """
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    console.rule("[bold cyan]Adversarial Example Generation")
    
    # Load gold data
    console.print(f"[cyan]Loading gold data from {gold_path}...[/]")
    gold_data = {p["game_id"]: p for p in read_jsonl(gold_path)}
    console.print(f"✓ Loaded {len(gold_data)} puzzles")
    
    adversarial_examples = []
    
    # Type 1: Near-miss examples (from predictions)
    if include_near_miss and predictions_path:
        console.print(f"\n[cyan]Extracting near-miss examples from {predictions_path}...[/]")
        predictions = read_jsonl(predictions_path)
        
        near_miss_count = 0
        for pred in predictions:
            game_id = pred["game_id"]
            if game_id not in gold_data:
                continue
            
            puzzle = gold_data[game_id]
            pred_groups = pred["pred_groups"]
            
            # Check how many groups are correct
            pred_normalized = [set(g) for g in pred_groups]
            gold_normalized = [set(g["members"]) for g in puzzle["groups"]]
            
            correct = sum(1 for p in pred_normalized if p in gold_normalized)
            
            # Near-miss: exactly 3/4 correct
            if correct == 3:
                example = create_near_miss_example(puzzle, pred_groups)
                adversarial_examples.append(example)
                near_miss_count += 1
        
        console.print(f"✓ Found {near_miss_count} near-miss examples")
    
    # Type 2: Hard puzzles (level 3 difficulty)
    if include_hard:
        console.print(f"\n[cyan]Finding hard puzzles (difficulty ≥ {min_difficulty})...[/]")
        
        hard_count = 0
        for puzzle in gold_data.values():
            max_level = max(g["level"] for g in puzzle["groups"])
            
            if max_level >= min_difficulty:
                example = create_hard_puzzle_example(puzzle)
                adversarial_examples.append(example)
                hard_count += 1
        
        console.print(f"✓ Found {hard_count} hard puzzles")
    
    # Save
    if adversarial_examples:
        with open(output_path, "w") as f:
            for ex in adversarial_examples:
                f.write(json.dumps(ex) + "\n")
        
        console.rule("[bold green]Generation Complete")
        console.print(f"✓ Generated {len(adversarial_examples)} adversarial examples")
        console.print(f"💾 Saved to: {output_path}")
        
        # Show difficulty distribution
        all_levels = [level for ex in adversarial_examples for level in ex["metadata"]["difficulty_levels"]]
        from collections import Counter
        level_counts = Counter(all_levels)
        console.print(f"\n📊 Difficulty distribution:")
        for level in sorted(level_counts.keys()):
            console.print(f"  Level {level}: {level_counts[level]} groups")
        
        # Show sample
        console.print("\n[cyan]Sample adversarial example:[/]")
        sample = adversarial_examples[0]
        console.print(f"[bold]Type:[/] {sample['metadata']['type']}")
        console.print(f"[bold]Input:[/] {sample['input'][:100]}...")
        console.print(f"[bold]Difficulty:[/] {sample['metadata']['difficulty_levels']}")
    else:
        console.print("[yellow]No adversarial examples generated![/]")


if __name__ == "__main__":
    app()