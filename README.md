# NYT Connections Solver: Fine-tuning Qwen 14B via Distillation

A fine-tuned Qwen 2.5 14B model that solves NYT Connections puzzles, trained through distillation from Claude Sonnet 4.5.

## Results

| Model | Solve Rate | Avg Groups | Avg Mistakes |
|-------|-----------|------------|--------------|
| Base Qwen 14B | 9.3% | 0.75 | 3.4 |
| GPT-4o-mini | 10.0% | 0.78 | 3.0 |
| Claude Haiku 3.5 | 13.3% | 1.07 | 3.1 |
| GPT-4o | 22.7% | 1.49 | 3.1 |
| **Alpha (fine-tuned)** | 27.3% | 1.75 | 3.2 |
| **Beta (fine-tuned)** | **30.0%** | **1.91** | 3.2 |
| Claude Sonnet 4.5 (teacher) | 87.3% | 3.61 | 0.9 |

**Key findings:**
- 🏆 Fine-tuned Qwen 14B **beats GPT-4o** (30% vs 22.7%)
- 📈 **3.2x improvement** over base model through distillation
- 💰 Zero inference cost after one-time training (~$5 compute on RunPod)

## Approach

### 1. Data Collection
- 913 NYT Connections puzzles (June 2023 - December 2025)
- Temporal train/test split: 763 train, 150 test (test = most recent puzzles)

### 2. Training Data Generation (Distillation)
- **Golden traces**: 358 one-shot puzzle solutions from Claude Sonnet
- **Game traces**: 150 multi-turn game transcripts with feedback (CORRECT, WRONG, ONE AWAY)

### 3. Model Training
| Model | Training Data | Epochs |
|-------|--------------|--------|
| Alpha | 358 golden traces | 3 |
| Beta | 358 golden + 150 game traces | 2.5 |

Training: QLoRA (rank 32) with Unsloth on A100 80GB (~20 min per model)

### 4. Evaluation
Game loop with:
- 4 mistakes allowed
- ONE AWAY feedback and targeted recovery prompts
- Temperature 0.3 → 0.7 "hail mary" on last attempt
- Auto-complete when 3 groups found

## Repository Structure

```
nyt-connections-solver/
├── README.md
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Load puzzles, train/test split
│   ├── 02_generate_traces.ipynb     # Play games with Sonnet, save traces
│   ├── 03_training.ipynb            # Fine-tune with Unsloth
│   └── 04_evaluation.ipynb          # Run game loop evaluation
├── src/
│   ├── game_loop.py                 # Core game logic
│   ├── evaluation.py                # Batch evaluation
│   └── utils.py                     # JSON extraction helpers
├── experiments/
│   └── failed_rwr_approach.md       # What didn't work
└── results/
    └── summary.json                 # Final numbers
```

## Quick Start

### Install dependencies
```bash
pip install unsloth anthropic openai transformers datasets
```

### Run evaluation with a fine-tuned model
```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, "path/to/lora-weights")
FastLanguageModel.for_inference(model)

# Run game
from src.game_loop import play_game
result = play_game(puzzle, model, tokenizer)
```

## What Didn't Work

See `experiments/` for documented failed approaches:
- Reward-weighted regression with embedding similarity
- Too many synthetic game traces (dilution effect)
- 3 epochs on beta data (slight overfitting)

## Training Details

- **Base model**: Qwen 2.5 14B Instruct (4-bit quantized)
- **LoRA config**: rank=32, alpha=64, dropout=0.05
- **Training**: AdamW 8-bit, lr=2e-4, batch=8 (2×4 grad accum)
- **Hardware**: NVIDIA A100 80GB
- **Cost**: ~$5 on RunPod

## License

MIT

## Acknowledgments

- Training data generated using Claude Sonnet 4.5 (Anthropic)
- Puzzle data from NYT Connections (thanks to Eric Nunes) at [Kaggle](https://www.kaggle.com/datasets/eric27n/the-new-york-times-connections/data)
- Fine-tuning accelerated by [Unsloth](https://github.com/unslothai/unsloth)
