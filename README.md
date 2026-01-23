# NYT Connections Solver

Fine-tuning language models to solve NYT Connections puzzles.

## Project Structure

```
nyt-connections-solver/
├── README.md                    # Project overview + results
├── data/
│   ├── sample_puzzles.jsonl     # Example puzzles
│   └── results_summary.json     # Final numbers for all models
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_generate_traces.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── game_loop.py             # Clean version of play_game_with_trace
│   ├── evaluation.py            # run_evaluation function
│   └── utils.py                 # JSON extraction, etc.
├── results/
│   ├── base_summary.json
│   ├── alpha_summary.json
│   ├── beta_summary.json
│   └── example_traces.json      # Cherry-picked interesting examples
└── blog_post.md                 # Blog post or link
```

## Overview

This project explores fine-tuning language models to improve their performance on NYT Connections puzzles.

## Results

*Results will be added here*

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt  # or use uv/poetry
```

## Usage

See the notebooks in `notebooks/` for the complete workflow:
1. Data preparation
2. Generating reasoning traces
3. Fine-tuning models
4. Evaluation

## License

MIT
