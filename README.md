# WIP


Connections-SOLVER

Goal: build a small, open-source model + solver that can reliably solve NYT Connections (16 words → 4 groups of 4) — with a clean evaluation harness, reproducible baselines, and two improvement stages: SFT on synthetic data and RL fine-tuning in a simulator.


# Data

Use public/permissioned datasets and synthetic puzzles; avoid scraping NYT (ToS).

Generate good puzzles + hard negatives with an LLM-as-generator and LLM-as-judge.

# Baselines

(B0) K-Means on embeddings (K=4) + merge heuristics.

(B1) Tuple-coherence scorer + beam search solver.

(B2) “LLM solver” (few-shot prompt) for reference.

# SFT

LoRA-tune a small, open model (e.g., Llama-3-8B-Instruct, Mistral-7B, or Phi-3-mini) to score tuples / propose groups. Use Unsloth for fast adapters.

# RL

Treat the game as an MDP (pick next 4-set). Train with PPO/AWR on a synthetic simulator, optionally with offline RL from demonstrations.

# Metrics

Exact solve rate; partial credit (#groups correct); average mistakes; per-difficulty succe
