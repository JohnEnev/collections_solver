from __future__ import annotations

import random
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Iterable

import orjson
import typer
from rich import print
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..models import solve_with_model, MODEL_PRESETS
from .evaluate import evaluate_predictions
from ..core.env import load_env

app = typer.Typer()
console = Console()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def validate_partition(words: List[str], groups: List[List[str]]) -> tuple[bool, str]:
    """
    Sanity-check that predicted groups form a valid partition of the 16 input words.
    """
    flat = [w for g in groups for w in g]
    if len(flat) != 16:
        return False, f"Expected 16 words but got {len(flat)}"
    if len(set(flat)) != 16:
        dupes = [w for w, c in Counter(flat).items() if c > 1]
        return False, f"Duplicate words: {dupes}"
    if Counter(flat) != Counter(words):
        extra = list((Counter(flat) - Counter(words)).elements())
        missing = list((Counter(words) - Counter(flat)).elements())
        return False, f"Word mismatch. extra={extra} missing={missing}"
    return True, "ok"


@app.command()
def main(
    in_path: str = "data/interim/test.jsonl",
    out_path: str = "data/cache/llm_baseline.jsonl",
    model: str = "gpt4o",
    limit: int = 50,
    seed: int = 42,
    show_sample: bool = True,
    log_raw: bool = True,
    samples: int = 3,
    temperature: float = 0.2,
    debug: bool = False,
):
    """
    Run LLM baseline using clean provider-specific clients.
    
    Supports:
    - OpenAI: gpt4o, gpt4o-mini, etc.
    - Anthropic: sonnet, opus, haiku
    - Google: gemini, gemini-pro, gemini-flash
    - OSS: llama-70b, llama-8b, qwen-72b, mistral-7b
    - Local: local/your-model-name (requires vLLM/Ollama running)
    """
    seen = load_env()
    if debug:
        from rich import print as rprint
        rprint({"env_keys_detected": seen})
        rprint({"available_presets": list(MODEL_PRESETS.keys())})

    random.seed(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    gold_list = list(read_jsonl(in_path))
    if not gold_list:
        console.print(f"[red]No puzzles found at[/] {in_path}")
        raise SystemExit(1)

    puzzles = gold_list.copy()
    random.shuffle(puzzles)
    puzzles = puzzles[:limit]
    gold_by_id = {p["game_id"]: p for p in gold_list}

    # Show which model we're using (resolved from preset if applicable)
    resolved = MODEL_PRESETS.get(model, model)
    console.rule(
        f"[bold green]LLM Baseline[/]\n"
        f"Model: {model} → {resolved}\n"
        f"Puzzles: {len(puzzles)} | Seed: {seed} | Temp: {temperature}"
    )

    preds: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=False,
    ) as prog:
        task = prog.add_task("Solving", total=len(puzzles))
        
        for p in puzzles:
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Use the new unified solve_with_model function
                    res = solve_with_model(
                        p["words"], 
                        model=model,
                        temperature=temperature
                    )
                    
                    ok, msg = validate_partition(p["words"], res["groups"])
                    if not ok:
                        if attempt < max_retries - 1 and "Duplicate" in msg:
                            # Retry with slightly higher temperature to get different response
                            continue
                        raise ValueError(f"Invalid partition: {msg}")

                    preds.append({
                        "game_id": p["game_id"],
                        "pred_groups": res["groups"],
                        "rationales": res.get("rationales", []),
                    })
                    
                    if log_raw:
                        raw_path = Path(out_path).with_suffix(".raw.jsonl")
                        with raw_path.open("ab") as rf:
                            rf.write(
                                orjson.dumps({
                                    "game_id": p["game_id"], 
                                    "raw": res.get("raw", ""),
                                    "model": res.get("model", model)
                                }) + b"\n"
                            )
                    
                    success = True
                    break  # Success, exit retry loop
                        
                except Exception as e:
                    msg = str(e)
                    hint = ""
                    
                    # Extract actual error from RetryError
                    actual_error = msg
                    if hasattr(e, '__cause__') and e.__cause__:
                        actual_error = str(e.__cause__)
                    
                    # Better error messages
                    if "credit balance is too low" in actual_error.lower():
                        hint = (
                            f"💳 Anthropic account is out of credits! "
                            f"Add credits at https://console.anthropic.com/settings/billing"
                        )
                    elif "API key" in actual_error or "authentication" in actual_error.lower():
                        hint = (
                            "Hint: Check your API key in .env file. "
                            f"For model '{model}', you need the appropriate key set."
                        )
                    elif "BadRequestError" in actual_error or "404" in actual_error or "NotFound" in actual_error:
                        hint = (
                            f"Hint: Model '{model}' (→ {resolved}) may not exist or isn't available on your account. "
                            "Try: gpt4o, gpt4o-mini, opus, haiku, or gemini"
                        )
                    elif "rate_limit" in actual_error.lower() or "429" in actual_error:
                        hint = "Hint: Rate limited. Try --limit 10 or wait a moment."
                    
                    # If this is the last retry, record the failure
                    if attempt == max_retries - 1:
                        error_msg = actual_error if debug else msg
                        failures.append({
                            "game_id": p.get("game_id", "?"),
                            "error": error_msg + (f"\n{hint}" if hint else ""),
                            "trace": traceback.format_exc() if debug else "",
                        })
                        break
            
            # Always update progress after puzzle is done (success or failure)
            prog.update(task, advance=1)

    # Write predictions
    with open(out_path, "wb") as f:
        for r in preds:
            f.write(orjson.dumps(r) + b"\n")

    # Evaluate
    console.rule("[bold cyan]Evaluation")
    if preds:
        stats = evaluate_predictions(out_path, in_path)
        print(stats)
    else:
        console.print("[red]No successful predictions to evaluate![/]")

    # Show failures
    if failures:
        console.rule(f"[bold red]Failures ({len(failures)})")
        
        if debug:
            # In debug mode, print full errors
            for f in failures[:10]:
                console.print(f"[bold red]Game {f['game_id']}:[/]")
                console.print(f["error"])
                if f.get("trace"):
                    console.print("\n[dim]Full trace:[/]")
                    console.print(f["trace"])
                console.print("")
        else:
            # Normal mode: show table with truncated errors
            table = Table("game_id", "error")
            for f in failures[:10]:
                table.add_row(str(f["game_id"]), f["error"][:120])
            console.print(table)
        
        if len(failures) > 10:
            console.print(f"...and {len(failures) - 10} more.")

    # Show samples
    if show_sample and preds:
        def is_exact(pred_groups: List[List[str]], gold_groups: List[Dict[str, Any]]) -> bool:
            def norm(gs): return sorted([tuple(sorted(g)) for g in gs])
            return norm(pred_groups) == norm([g["members"] for g in gold_groups])

        exact_samples: List[Dict[str, Any]] = []
        non_exact_samples: List[Dict[str, Any]] = []
        for r in preds:
            gid = r["game_id"]
            gold = gold_by_id.get(gid)
            if not gold:
                continue
            (exact_samples if is_exact(r["pred_groups"], gold["groups"]) else non_exact_samples).append(r)

        chosen: List[Dict[str, Any]] = []
        if exact_samples:
            chosen.append(exact_samples[0])

        pool = non_exact_samples if exact_samples else preds
        for r in pool:
            if len(chosen) >= samples:
                break
            if r not in chosen:
                chosen.append(r)

        console.rule(f"[bold magenta]Samples ({len(chosen)}) — predictions vs ground truth")
        for i, samp in enumerate(chosen, start=1):
            gid = samp["game_id"]
            gold = gold_by_id[gid]
            table = Table("Group #", "Predicted (4 words)", "Rationale", "Gold (4 words)")
            gold_groups = [g["members"] for g in gold["groups"]]
            for j in range(4):
                pred_grp = samp["pred_groups"][j]
                rat = (samp.get("rationales", [""] * 4)[j] or "")[:120]
                gold_grp = gold_groups[j]
                table.add_row(str(j), ", ".join(pred_grp), rat, ", ".join(gold_grp))
            console.print(f"[bold]Sample {i} — game_id:[/] {gid}")
            console.print(table)


def cli():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli()