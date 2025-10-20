from __future__ import annotations
import orjson, pathlib, typer
from collections import Counter
from rich import print

app = typer.Typer()

def read_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)

@app.command()
def main(path: str = "data/interim/train.jsonl"):
    puzzles = list(read_jsonl(path))
    levels = Counter()
    word_lengths = []
    for p in puzzles:
        for g in p["groups"]:
            levels[g["level"]] += 1
        word_lengths += [len(w) for w in p["words"] if w is not None]

    print(f"[bold]Puzzles[/]: {len(puzzles)}")
    print(f"[bold]Group levels[/]: {dict(levels)} (0=easiest … 3=hardest)")
    print(f"[bold]Avg word len[/]: {sum(word_lengths)/len(word_lengths):.2f}")

if __name__ == "__main__":
    app()
