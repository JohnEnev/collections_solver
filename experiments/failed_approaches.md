# Failed Experiments & Lessons Learned

This document chronicles approaches that didn't work, to save others time and provide context for the final solution.

## 1. Reward-Weighted Regression (RWR) with Embeddings

### Approach
- Use sentence embeddings (all-MiniLM-L6-v2) to compute word similarity
- Score candidate groupings based on intra-group similarity
- Generate multiple candidate solutions, weight by similarity scores
- Train model to prefer higher-scored outputs

### Why It Failed
- **Embedding similarity doesn't capture wordplay**: NYT Connections often uses puns, phrases, and cultural references that don't have semantic similarity
- Example: "JAZZ", "HEAT", "BUCKS", "NETS" (NBA teams) have low embedding similarity
- Example: "MOM", "KAYAK", "RACECAR", "LEVEL" (palindromes) have zero semantic relationship
- **The hard puzzles are hard precisely because the connections aren't semantic**

### Lesson
Embedding-based approaches work for surface-level semantic grouping but fail on the cleverness that makes Connections interesting.

---

## 2. Too Many Game Traces (Dilution Effect)

### Approach
- Generate 150 game traces (multi-turn with feedback)
- Combine with 358 golden traces (one-shot reasoning)
- Train on the combined dataset

### Results
| Training Data | Solve Rate |
|--------------|------------|
| 358 golden only | 27.3% |
| 358 golden + 150 game | 28.0% (3ep) |
| 358 golden + 150 game | 30.0% (2.5ep) |
| 358 golden + 50 game | 22.7% |

### Why 50 Game Traces Performed Worse
Initially hypothesized that 150 traces were "diluting" the golden traces. Tested with only 50 game traces.

**Result**: 22.7% — significantly worse!

### Actual Finding
- More game traces = better (up to a point)
- The issue wasn't dilution, it was **overfitting at 3 epochs**
- 2.5 epochs hit the sweet spot for the 150-trace mix

### Lesson
When mixing data types, tune epochs carefully. More data isn't always worse — you may just need fewer epochs.

---

## 3. Overfitting at 3 Epochs

### Observation
Training Beta model (358 golden + 150 game):
- 3 epochs: 28.0% solve rate
- 2.5 epochs: 30.0% solve rate

### The Accidental 5.5 Epoch Model
Accidentally trained an already-fine-tuned model for 3 more epochs (total 5.5):
- Starting loss: ~0.40 (should be ~1.0 for fresh model)
- Ending loss: ~0.15 (too low = memorization)
- **Did not test** but expected to overfit badly

### Lesson
- Watch starting loss to verify you're training from base
- Loss of 0.15-0.20 on this task = likely overfitting
- 2.5 epochs was optimal for ~500 examples

---

## 4. Structured State Format (Early Experiment)

### Approach
Instead of natural language prompts, used structured state:
```
STATE:
- REMAINING: [16 words]
- FOUND: [groups found]
- MISTAKES: 2/4
- WRONG_GUESSES: [list]

ACTION: Output {"group": [...]}
```

### Why It Failed
- Model struggled to follow rigid format
- Lost the "think step by step" reasoning that helps find patterns
- Natural language prompts let the model explore associations

### Lesson
For reasoning tasks, natural language prompts > structured formats.

---

## 5. ONE AWAY Recovery Didn't Improve

### Hypothesis
Training on game traces with ONE AWAY feedback would teach the model to recover from near-misses.

### Result
| Model | ONE AWAY Situations | Recovery Rate |
|-------|---------------------|---------------|
| Alpha (no game traces) | 228 | 21.5% |
| Beta (with game traces) | 257 | 17.9% |

**Beta was actually worse at ONE AWAY recovery!**

### Analysis
- The model encounters more ONE AWAY situations (257 vs 228) — perhaps more exploratory
- But recovery rate dropped from 21.5% to 17.9%
- Game traces may have taught the format but not the reasoning for swaps

### Lesson
Teaching a model the feedback format ≠ teaching it to reason about corrections. May need more targeted ONE AWAY training data.

---

## What Actually Worked

1. **Distillation from a stronger model** (Sonnet → Qwen)
2. **One-shot reasoning traces** with step-by-step thinking
3. **Careful epoch tuning** (2.5 epochs for 500 examples)
4. **Higher LoRA rank** (32 vs 16) for this task
5. **Temporal train/test split** for realistic evaluation
