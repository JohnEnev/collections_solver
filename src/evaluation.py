"""
Batch evaluation for NYT Connections models.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from .game_loop import play_game


def load_puzzles(filepath: str) -> List[Dict]:
    """Load puzzles from JSONL file."""
    puzzles = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                puzzles.append(json.loads(line))
    return puzzles


def load_progress(progress_file: str) -> Dict:
    """Load evaluation progress from file."""
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"completed_ids": [], "results": []}


def save_progress(progress: Dict, progress_file: str):
    """Save evaluation progress to file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f)


def append_trace(puzzle: Dict, result: Dict, traces_file: str):
    """Append game trace to JSONL file."""
    with open(traces_file, "a") as f:
        f.write(json.dumps({
            "puzzle_id": puzzle["game_id"],
            "date": puzzle.get("date"),
            "words": puzzle["words"],
            "solution": puzzle["solution"],
            "solved": result["solved"],
            "groups_found": result["groups_found"],
            "mistakes": result["mistakes"],
            "trace": result.get("trace", [])
        }) + "\n")


def print_running_stats(results: List[Dict]):
    """Print running statistics."""
    n = len(results)
    solved = sum(r["solved"] for r in results)
    avg_groups = sum(r["groups_found"] for r in results) / n
    avg_mistakes = sum(r["mistakes"] for r in results) / n
    total_valid = sum(r["valid_outputs"] for r in results)
    total_invalid = sum(r["invalid_outputs"] for r in results)
    validity_rate = total_valid / (total_valid + total_invalid) * 100 if (total_valid + total_invalid) > 0 else 0
    
    print(f"  📊 Running: {solved}/{n} ({solved/n*100:.1f}%) solved | "
          f"Avg groups: {avg_groups:.2f} | Avg mistakes: {avg_mistakes:.1f} | "
          f"Valid outputs: {validity_rate:.1f}%")


def run_evaluation(
    model,
    tokenizer,
    puzzles: List[Dict],
    model_name: str = "model",
    output_dir: str = "data",
    save_traces: bool = True
) -> List[Dict]:
    """
    Run evaluation on a list of puzzles.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        puzzles: List of puzzle dicts
        model_name: Name for output files
        output_dir: Directory for output files
        save_traces: Whether to save full game traces
    
    Returns:
        List of result dicts
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    progress_file = f"{output_dir}/eval_{model_name}_progress.json"
    traces_file = f"{output_dir}/eval_{model_name}_traces.jsonl"
    
    # Load progress for resume capability
    progress = load_progress(progress_file)
    completed_ids = set(progress["completed_ids"])
    results = progress["results"]
    
    remaining = [p for p in puzzles if p["game_id"] not in completed_ids]
    
    print(f"Resuming: {len(completed_ids)} done, {len(remaining)} remaining")
    if results:
        print_running_stats(results)
    print()
    
    for i, puzzle in enumerate(remaining):
        puzzle_id = puzzle["game_id"]
        
        # Play game
        result = play_game(puzzle, model, tokenizer, return_trace=save_traces)
        
        # Update progress
        results.append({
            "puzzle_id": puzzle_id,
            "solved": result["solved"],
            "groups_found": result["groups_found"],
            "mistakes": result["mistakes"],
            "valid_outputs": result["valid_outputs"],
            "invalid_outputs": result["invalid_outputs"]
        })
        completed_ids.add(puzzle_id)
        
        # Save immediately (crash recovery)
        save_progress({"completed_ids": list(completed_ids), "results": results}, progress_file)
        if save_traces:
            append_trace(puzzle, result, traces_file)
        
        # Print status
        status = "✅" if result["solved"] else "❌"
        total_done = len(results)
        print(f"{status} [{total_done}/{len(puzzles)}] Puzzle {puzzle_id}: "
              f"Groups {result['groups_found']}/4, Mistakes {result['mistakes']}")
        
        # Running stats every 10 puzzles
        if total_done % 10 == 0:
            print_running_stats(results)
            print()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    n = len(results)
    solved = sum(r["solved"] for r in results)
    avg_groups = sum(r["groups_found"] for r in results) / n
    avg_mistakes = sum(r["mistakes"] for r in results) / n
    total_valid = sum(r["valid_outputs"] for r in results)
    total_invalid = sum(r["invalid_outputs"] for r in results)
    validity_rate = total_valid / (total_valid + total_invalid) * 100
    
    print(f"Solved:        {solved}/{n} ({solved/n*100:.1f}%)")
    print(f"Avg groups:    {avg_groups:.2f}/4")
    print(f"Avg mistakes:  {avg_mistakes:.1f}")
    print(f"Valid outputs: {validity_rate:.1f}%")
    print(f"{'='*60}")
    
    return results


def analyze_one_away_recovery(traces_file: str) -> Dict[str, Any]:
    """
    Analyze ONE AWAY recovery rates from trace file.
    
    Returns dict with total ONE AWAY situations and recovery count.
    """
    one_away_total = 0
    one_away_recovered = 0
    
    with open(traces_file) as f:
        for line in f:
            t = json.loads(line)
            trace = t.get("trace", [])
            
            for i, turn in enumerate(trace):
                if turn.get("result") == "ONE_AWAY":
                    one_away_total += 1
                    # Check if next non-duplicate guess was correct
                    for next_turn in trace[i+1:]:
                        if next_turn.get("result") == "CORRECT":
                            one_away_recovered += 1
                            break
                        elif next_turn.get("result") in ["WRONG", "ONE_AWAY"]:
                            break
    
    recovery_rate = one_away_recovered / one_away_total * 100 if one_away_total > 0 else 0
    
    return {
        "total": one_away_total,
        "recovered": one_away_recovered,
        "recovery_rate": recovery_rate
    }
