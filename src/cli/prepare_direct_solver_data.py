#!/usr/bin/env python3
"""Prepare training data for direct puzzle solver."""

import orjson
import typer
from pathlib import Path
from rich import print

app = typer.Typer()

def read_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)

def is_valid_puzzle(puzzle):
    """Check if puzzle has valid data (no None values)."""
    # Check words
    if not puzzle.get("words") or len(puzzle["words"]) != 16:
        return False
    if any(w is None for w in puzzle["words"]):
        return False
    
    # Check groups
    if not puzzle.get("groups") or len(puzzle["groups"]) != 4:
        return False
    
    for group in puzzle["groups"]:
        members = group.get("members", [])
        if len(members) != 4:
            return False
        if any(m is None for m in members):
            return False
    
    return True

def format_for_training(puzzle):
    """Format with few-shot examples."""
    words_str = ", ".join(puzzle["words"])
    
    groups = [g["members"] for g in puzzle["groups"]]
    rationales = [g["name"] for g in puzzle["groups"]]
    
    output = {
        "groups": groups,
        "rationales": rationales
    }
    
    # Few-shot examples in the instruction
    instruction = """
        Solve this NYT Connections puzzle. Group the 16 words into 4 groups of 4 words each. Each word must be used exactly once.

        Example:
        Words: SHIFT, TAB, RETURN, OPTION, HEAT, BUCKS, JAZZ, NETS, LEVEL, KAYAK, RACECAR, MOM, SNOW, HAIL, RAIN, SLEET
        Solution: {"groups":[["SHIFT","TAB","RETURN","OPTION"],["HEAT","BUCKS","JAZZ","NETS"],["LEVEL","KAYAK","RACECAR","MOM"],["SNOW","HAIL","RAIN","SLEET"]],"rationales":["KEYBOARD KEYS","NBA TEAMS","PALINDROMES","WET WEATHER"]}

        Now solve this puzzle. Respond with JSON containing 'groups' and 'rationales'.
        """
    
    return {
        "game_id": puzzle["game_id"],
        "instruction": instruction,
        "input": f"Words: {words_str}",
        "output": orjson.dumps(output).decode("utf-8")
    }

@app.command()
def main(
    train_path: str = "data/interim/train.jsonl",
    dev_path: str = "data/interim/dev.jsonl",
    out_train: str = "data/interim/solver_train.jsonl",
    out_val: str = "data/interim/solver_val.jsonl",
):
    """Prepare direct solver training data using existing train/dev split."""
    
    # Process training data
    train_puzzles = list(read_jsonl(train_path))
    print(f"[cyan]Loaded {len(train_puzzles)} training puzzles from {train_path}[/]")
    
    # Filter valid puzzles
    valid_train = [p for p in train_puzzles if is_valid_puzzle(p)]
    invalid_train = len(train_puzzles) - len(valid_train)
    if invalid_train > 0:
        print(f"[yellow]⚠ Filtered out {invalid_train} invalid training puzzles (with None values)[/]")
    
    train_data = [format_for_training(p) for p in valid_train]
    
    # Process validation data
    dev_puzzles = list(read_jsonl(dev_path))
    print(f"[cyan]Loaded {len(dev_puzzles)} dev puzzles from {dev_path}[/]")
    
    # Filter valid puzzles
    valid_dev = [p for p in dev_puzzles if is_valid_puzzle(p)]
    invalid_dev = len(dev_puzzles) - len(valid_dev)
    if invalid_dev > 0:
        print(f"[yellow]⚠ Filtered out {invalid_dev} invalid dev puzzles (with None values)[/]")
    
    val_data = [format_for_training(p) for p in valid_dev]
    
    print(f"[green]Final split: {len(train_data)} train, {len(val_data)} validation[/]")
    
    # Write train
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "wb") as f:
        for item in train_data:
            f.write(orjson.dumps(item) + b"\n")
    
    # Write val
    with open(out_val, "wb") as f:
        for item in val_data:
            f.write(orjson.dumps(item) + b"\n")
    
    print(f"[green]✓ Training data saved to:[/]")
    print(f"  Train: {out_train}")
    print(f"  Val:   {out_val}")
    
    # Show sample
    print(f"\n[yellow]Sample training example:[/]")
    sample = train_data[0]
    print(f"[bold]Game ID:[/] {sample['game_id']}")
    print(f"[bold]Input:[/] {sample['input'][:100]}...")
    print(f"[bold]Output:[/] {sample['output'][:150]}...")

if __name__ == "__main__":
    app()