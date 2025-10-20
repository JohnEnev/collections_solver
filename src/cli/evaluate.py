from __future__ import annotations

import orjson
import typer
from collections import Counter
from typing import Dict, Any, Iterable, List, Tuple

app = typer.Typer()


def read_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with open(p, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def normalize_groups(groups: List[List[str]]) -> List[Tuple[str, ...]]:
    # sort words inside group + sort group list for comparison
    return sorted([tuple(sorted(g)) for g in groups])


def eval_one(pred_groups: List[List[str]], gold_groups: List[Dict[str, Any]]):
    pred = normalize_groups(pred_groups)
    gold = normalize_groups([g["members"] for g in gold_groups])
    correct = sum(1 for g in pred if g in gold)
    exact = int(correct == 4)
    return exact, correct


def simulate_mistakes_best_case(
    pred_groups: List[List[str]], gold_groups: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Simulate mistakes in a best-case ordering:
    - We assume you can choose the order of your 4 predicted groups to minimize mistakes.
    - A 'guess' equals selecting one predicted group.
    - If the predicted group exactly equals any remaining gold group -> it's correct and removed (no mistake).
    - Else it's a mistake (+1). The real game ends after 3 mistakes; we track whether solved within 0/1/2/3 mistakes.
    Returns:
      {
        "mistakes": int (minimum mistakes needed to get all 4 correct; 3+ means you would bust in-game),
        "solved_within_0": 0/1,
        "solved_within_1": 0/1,
        "solved_within_2": 0/1,
        "solved_within_3": 0/1
      }
    """
    gold_sets = [set(g["members"]) for g in gold_groups]
    pred_sets = [set(g) for g in pred_groups]

    # put exact matches first (best-case)
    exact_first = sorted(
        pred_sets, key=lambda s: any(s == g for g in gold_sets), reverse=True
    )

    mistakes = 0
    remaining_gold = gold_sets.copy()

    for s in exact_first:
        # if already busted, we can stop counting further (but we compute total min mistakes)
        if any(s == g for g in remaining_gold):
            # correct guess, remove that gold group
            idx = next(i for i, g in enumerate(remaining_gold) if g == s)
            remaining_gold.pop(idx)
        else:
            mistakes += 1

    solved = int(len(remaining_gold) == 0)
    out = {
        "mistakes": mistakes,
        "solved_within_0": int(solved and mistakes <= 0),
        "solved_within_1": int(solved and mistakes <= 1),
        "solved_within_2": int(solved and mistakes <= 2),
        "solved_within_3": int(solved and mistakes <= 3),
    }
    return out


def evaluate_predictions(pred_path: str, gold_path: str = "data/interim/test.jsonl"):
    gold = {p["game_id"]: p for p in read_jsonl(gold_path)}
    stats = Counter()
    per_level = Counter()
    n = 0

    mistake_sum = 0
    solved_within = Counter({0: 0, 1: 0, 2: 0, 3: 0})

    for r in read_jsonl(pred_path):
        gid = r["game_id"]
        if gid not in gold:
            continue
        g = gold[gid]
        exact, correct = eval_one(r["pred_groups"], g["groups"])
        stats["exact"] += exact
        stats[f"groups_{correct}"] += 1

        # per-difficulty: count only when exact (simple, conservative)
        if exact:
            for gg in g["groups"]:
                per_level[f"level_{gg['level']}_correct"] += 1

        # mistake simulation (best-case ordering of the 4 predicted groups)
        sim = simulate_mistakes_best_case(r["pred_groups"], g["groups"])
        mistake_sum += sim["mistakes"]
        for k in (0, 1, 2, 3):
            solved_within[k] += sim[f"solved_within_{k}"]

        n += 1

    out = {
        "n_puzzles": n,
        "exact_rate": (stats["exact"] / n) if n else 0.0,
        "groups_0": stats["groups_0"],
        "groups_1": stats["groups_1"],
        "groups_2": stats["groups_2"],
        "groups_3": stats["groups_3"],
        "groups_4": stats["groups_4"],
        "per_level_correct_counts": dict(per_level),
        # new mistake metrics
        "avg_mistakes_until_solve": (mistake_sum / n) if n else 0.0,
        "solved_within_mistakes": {
            "k0": (solved_within[0] / n) if n else 0.0,
            "k1": (solved_within[1] / n) if n else 0.0,
            "k2": (solved_within[2] / n) if n else 0.0,
            "k3": (solved_within[3] / n) if n else 0.0,
        },
    }
    return out


@app.command()
def main(pred_path: str, gold_path: str = "data/interim/test.jsonl"):
    print(evaluate_predictions(pred_path, gold_path))


if __name__ == "__main__":
    app()
