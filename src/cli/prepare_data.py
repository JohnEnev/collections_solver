from __future__ import annotations
import pandas as pd, orjson, pathlib, typer
from rich import print
from datetime import datetime

app = typer.Typer()

def write_jsonl(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")

@app.command()
def main(csv_path: str = "data/raw/connections.csv",
         out_dir: str = "data/interim",
         train_ratio: float = 0.7,
         dev_ratio: float = 0.1):
    df = pd.read_csv(csv_path)
    # Normalize dates
    df["Puzzle Date"] = pd.to_datetime(df["Puzzle Date"]).dt.date

    # Build puzzles
    puzzles = []
    for gid, g in df.groupby("Game ID"):
        words = g["Word"].tolist()
        groups = []
        for name, gg in g.groupby("Group Name"):
            lvl = int(gg["Group Level"].iloc[0])
            groups.append({
                "name": name,
                "level": lvl,
                "members": gg["Word"].tolist()
            })
        puzzles.append({
            "game_id": str(gid),
            "date": str(min(g["Puzzle Date"])),  # one per puzzle
            "words": words,
            "groups": groups
        })

    # Temporal split by date
    puzzles = sorted(puzzles, key=lambda p: p["date"])
    n = len(puzzles)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = puzzles[:n_train]
    dev = puzzles[n_train:n_train+n_dev]
    test = puzzles[n_train+n_dev:]

    out = pathlib.Path(out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)

    print(f"[green]Wrote[/] train={len(train)} dev={len(dev)} test={len(test)} to {out}")

if __name__ == "__main__":
    app()
