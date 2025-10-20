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

import litellm  # for optional debug toggle

from src.models.llm_client import solve_with_llm
from src.cli.evaluate import evaluate_predictions  # expects your updated evaluator

app = typer.Typer()
console = Console()

# Handy aliases so you don't have to remember full provider IDs
PRESETS: Dict[str, str] = {
    "gpt4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "gpt5": "openai/gpt-5",                   # requires access on your key
    "gpt5-mini": "openai/gpt-5-mini",         # requires access
    "sonnet35": "anthropic/claude-3-5-sonnet-20241022",
    # Add more here if you like (Together, Fireworks, OpenRouter routes, etc.)
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def validate_partition(words: List[str], groups: List[List[str]]) -> tuple[bool, str]:
    """
    Sanity-check that predicted groups form a valid partition of the 16 input words:
    - exactly 4 groups of 4 (enforced earlier)
    - no duplicates
    - bag-of-words equality with the input
    Returns (ok, message).
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
    out_path: str = "data/cache/llm_frontier.jsonl",
    model: str = "gpt4o",
    limit: int = 50,
    seed: int = 42,
    show_sample: bool = True,
    log_raw: bool = True,
    samples: int = 3,
    debug: bool = False,
):
    """
    Run a hosted LLM baseline using LiteLLM for N puzzles and evaluate.
    - Progress bar + per-puzzle error capture
    - Partition sanity checks (duplicates/missing/extra)
    - Optional caching of raw model responses for debugging
    - Prints overall stats and multiple sample predictions with ground truth
    """
    # Resolve preset aliases
    if model in PRESETS:
        model = PRESETS[model]

    if debug:
        # Turn on detailed LiteLLM logging for troubleshooting
        try:
            litellm._turn_on_debug()
        except Exception:
            pass

    random.seed(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    gold_list = list(read_jsonl(in_path))
    if not gold_list:
        console.print("[red]No puzzles found at[/] ", in_path)
        raise SystemExit(1)

    puzzles = gold_list.copy()
    random.shuffle(puzzles)
    puzzles = puzzles[:limit]
    gold_by_id = {p["game_id"]: p for p in gold_list}

    console.rule(f"[bold green]LLM baseline :: {model}[/]  puzzles={len(puzzles)}  seed={seed}")
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
            try:
                res = solve_with_llm(p["words"], model=model)
                ok, msg = validate_partition(p["words"], res["groups"])
                if not ok:
                    raise ValueError(f"Invalid partition: {msg}")

                preds.append(
                    {
                        "game_id": p["game_id"],
                        "pred_groups": res["groups"],
                        "rationales": res.get("rationales", []),
                    }
                )
                if log_raw:
                    raw_path = Path(out_path).with_suffix(".raw.jsonl")
                    with raw_path.open("ab") as rf:
                        rf.write(
                            orjson.dumps(
                                {"game_id": p["game_id"], "raw": res.get("raw", "")}
                            )
                            + b"\n"
                        )
            except Exception as e:
                msg = str(e)
                hint = ""
                if "UnsupportedParamsError" in msg or "not supported" in msg.lower():
                    hint = (
                        "Hint: model id or required params are invalid for this provider/key. "
                        "Try --model sonnet35 (with ANTHROPIC_API_KEY) or --model gpt4o (with OPENAI_API_KEY)."
                    )
                failures.append(
                    {
                        "game_id": p.get("game_id", "?"),
                        "error": msg + (f"  {hint}" if hint else ""),
                        "trace": traceback.format_exc(),
                    }
                )
            finally:
                prog.update(task, advance=1)

    # Write predictions (even if some failed)
    with open(out_path, "wb") as f:
        for r in preds:
            f.write(orjson.dumps(r) + b"\n")

    # Evaluate
    console.rule("[bold cyan]Evaluation")
    stats = evaluate_predictions(out_path, in_path)
    print(stats)

    # Show failures (first 10)
    if failures:
        console.rule(f"[bold red]Failures ({len(failures)})")
        table = Table("game_id", "error")
        for f in failures[:10]:
            table.add_row(str(f["game_id"]), f["error"][:120])
        console.print(table)
        if len(failures) > 10:
            console.print(f"...and {len(failures) - 10} more.")

    # Show multiple samples (try to include at least one exact solve if any)
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
            chosen.append(exact_samples[0])  # ensure at least one exact if exists

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
            # Side-by-side by index (display only)
            for j in range(4):
                pred_grp = samp["pred_groups"][j]
                rat = (samp.get("rationales", [""] * 4)[j] or "")[:120]
                gold_grp = gold_groups[j]
                table.add_row(str(j), ", ".join(pred_grp), rat, ", ".join(gold_grp))
            console.print(f"[bold]Sample {i} — game_id:[/] {gid}")
            console.print(table)


if __name__ == "__main__":
    app()
