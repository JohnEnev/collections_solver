"""
CLI commands for NYT Connections experiments.
"""

from .evaluate import evaluate_predictions
from .run_baseline_llm import main as run_baseline
from .compare_models import main as compare_models

__all__ = [
    "evaluate_predictions",
    "run_baseline",
    "compare_models",
]