from __future__ import annotations
import orjson, typer, random
from pathlib import Path
from collections import defaultdict
from src.models.embeddings import embed_words
from src.models.clustering import kmeans_groups
from src.cli.evaluate import evaluate_predictions  # reuse evaluator

app = typer.Typer()

def read_jsonl(p): 
    for line in open(p, "rb"): yield orjson.loads(line)

@app.command()
def main(in_path: str = "data/interim/test.jsonl", out_path: str = "data/cache/emb_kmeans.jsonl"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    preds = []
    for p in read_jsonl(in_path):
        words = p["words"]
        embs = embed_words(words)
        labels = kmeans_groups(embs, k=4)
        # convert labels -> 4 groups of 4 by majority bucketing
        buckets = defaultdict(list)
        for w, lab in zip(words, labels):
            buckets[lab].append(w)
        # ensure exactly 4x4 (fallback: random balance)
        groups = list(buckets.values())
        if any(len(g)!=4 for g in groups) or len(groups)!=4:
            # simple rebalance
            flat = [w for g in groups for w in g]
            random.shuffle(flat)
            groups = [flat[i*4:(i+1)*4] for i in range(4)]
        preds.append({"game_id": p["game_id"], "pred_groups": groups})

    with open(out_path, "wb") as f:
        for r in preds: f.write(orjson.dumps(r) + b"\n")

    print(evaluate_predictions(out_path, in_path))

if __name__ == "__main__":
    app()
