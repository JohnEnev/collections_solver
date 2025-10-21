#!/usr/bin/env python3
"""Compare multiple model baselines on the SAME set of puzzles."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Set
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import orjson

from .evaluate import evaluate_predictions, read_jsonl

app = typer.Typer()
console = Console()


def get_game_ids(pred_file: str) -> Set[str]:
    """Get set of game_ids that were successfully solved."""
    if not Path(pred_file).exists():
        return set()
    return {p["game_id"] for p in read_jsonl(pred_file)}


def filter_to_common_puzzles(pred_files: List[str], output_dir: str = "data/cache/common/") -> List[str]:
    """
    Filter all prediction files to only include puzzles that ALL models completed.
    Returns list of filtered file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find common game_ids across all models
    all_ids = [get_game_ids(f) for f in pred_files if Path(f).exists()]
    if not all_ids:
        return []
    
    common_ids = set.intersection(*all_ids)
    console.print(f"[cyan]Found {len(common_ids)} puzzles completed by all {len(pred_files)} models[/]")
    
    # Create filtered versions
    filtered_files = []
    for pred_file in pred_files:
        if not Path(pred_file).exists():
            continue
            
        # Read all predictions
        all_preds = list(read_jsonl(pred_file))
        
        # Filter to common puzzles only
        common_preds = [p for p in all_preds if p["game_id"] in common_ids]
        
        # Write filtered version
        model_name = Path(pred_file).stem
        filtered_path = Path(output_dir) / f"{model_name}_common.jsonl"
        
        with open(filtered_path, "wb") as f:
            for pred in common_preds:
                f.write(orjson.dumps(pred) + b"\n")
        
        filtered_files.append(str(filtered_path))
    
    return filtered_files


def format_percentage(value: float) -> str:
    """Format float as percentage with color."""
    pct = value * 100
    if pct >= 50:
        return f"[green]{pct:.1f}%[/]"
    elif pct >= 25:
        return f"[yellow]{pct:.1f}%[/]"
    else:
        return f"[red]{pct:.1f}%[/]"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with appropriate color."""
    formatted = f"{value:.{decimals}f}"
    if value <= 1.0:
        return f"[green]{formatted}[/]"
    elif value <= 2.0:
        return f"[yellow]{formatted}[/]"
    else:
        return f"[red]{formatted}[/]"


@app.command()
def main(
    pred_files: List[str] = typer.Argument(..., help="Prediction files to compare"),
    gold_path: str = typer.Option("data/interim/test.jsonl", help="Ground truth file"),
    model_names: List[str] = typer.Option(None, "--name", "-n", help="Custom model names"),
    export_csv: str = typer.Option(None, help="Export results to CSV"),
    skip_filtering: bool = typer.Option(False, help="Skip filtering to common puzzles"),
):
    """
    Compare multiple model baselines on the SAME set of puzzles.
    
    By default, filters all predictions to only include puzzles that ALL models completed.
    
    Example:
        python -m src.cli.compare_models data/cache/*_baseline.jsonl
        python -m src.cli.compare_models gpt4o.jsonl sonnet.jsonl -n GPT-4o -n Claude
    """
    console.rule("[bold cyan]Model Comparison")
    
    # Filter to common puzzles first
    if not skip_filtering:
        console.print("[yellow]Filtering to common puzzles across all models...[/]")
        filtered_files = filter_to_common_puzzles(pred_files)
        if not filtered_files:
            console.print("[red]No common puzzles found![/]")
            raise typer.Exit(1)
        pred_files = filtered_files
    
    # Evaluate all models
    results = {}
    for i, pred_file in enumerate(pred_files):
        # Extract model name
        if model_names and i < len(model_names):
            name = model_names[i]
        else:
            name = Path(pred_file).stem.replace("_baseline", "").replace("_common", "")
        
        if not Path(pred_file).exists():
            console.print(f"[red]Warning: {pred_file} not found, skipping...[/]")
            continue
        
        stats = evaluate_predictions(pred_file, gold_path)
        results[name] = stats
    
    if not results:
        console.print("[red]No valid prediction files found![/]")
        raise typer.Exit(1)
    
    # Sort by exact_rate descending
    sorted_models = sorted(results.items(), key=lambda x: x[1]["exact_rate"], reverse=True)
    
    # Main comparison table
    table = Table(title="Model Performance Comparison", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Puzzles", justify="right")
    table.add_column("Exact Solve\n(All 4 Groups)", justify="right")
    table.add_column("Avg Groups\nCorrect", justify="right")
    table.add_column("≥1 Group", justify="right")
    table.add_column("≥2 Groups", justify="right")
    table.add_column("≥3 Groups", justify="right")
    
    for model_name, stats in sorted_models:
        n = stats["n_puzzles"]
        # Calculate cumulative percentages
        at_least_1 = (n - stats["groups_0"]) / n if n > 0 else 0
        at_least_2 = (stats["groups_2"] + stats["groups_3"] + stats["groups_4"]) / n if n > 0 else 0
        at_least_3 = (stats["groups_3"] + stats["groups_4"]) / n if n > 0 else 0
        avg_groups = (stats["groups_1"] + 2*stats["groups_2"] + 3*stats["groups_3"] + 4*stats["groups_4"]) / n if n > 0 else 0
        
        table.add_row(
            model_name,
            str(n),
            format_percentage(stats["exact_rate"]),
            format_number(avg_groups, 2),
            format_percentage(at_least_1),
            format_percentage(at_least_2),
            format_percentage(at_least_3),
        )
    
    console.print(table)
    
    # Group distribution table
    console.print("\n")
    dist_table = Table(title="Groups Correct Distribution", show_lines=True)
    dist_table.add_column("Model", style="bold")
    dist_table.add_column("0 Groups", justify="right")
    dist_table.add_column("1 Group", justify="right")
    dist_table.add_column("2 Groups", justify="right")
    dist_table.add_column("3 Groups", justify="right")
    dist_table.add_column("4 Groups", justify="right")
    
    for model_name, stats in sorted_models:
        n = stats["n_puzzles"]
        dist_table.add_row(
            model_name,
            f"{stats['groups_0']} ({stats['groups_0']/n*100:.1f}%)" if n > 0 else "0",
            f"{stats['groups_1']} ({stats['groups_1']/n*100:.1f}%)" if n > 0 else "0",
            f"{stats['groups_2']} ({stats['groups_2']/n*100:.1f}%)" if n > 0 else "0",
            f"{stats['groups_3']} ({stats['groups_3']/n*100:.1f}%)" if n > 0 else "0",
            f"{stats['groups_4']} ({stats['groups_4']/n*100:.1f}%)" if n > 0 else "0",
        )
    
    console.print(dist_table)
    
    # Summary insights
    console.print("\n[bold cyan]Key Insights:[/]")
    best_model = sorted_models[0]
    console.print(f"🏆 Best Exact Solve Rate: [bold]{best_model[0]}[/] ({best_model[1]['exact_rate']*100:.1f}%)")
    console.print(f"📊 All models tested on: {best_model[1]['n_puzzles']} common puzzles")
    
    # Find model with most groups correct on average
    if sorted_models[0][1]["n_puzzles"] > 0:
        best_avg = max(sorted_models, key=lambda x: (
            x[1]["groups_1"] + 2*x[1]["groups_2"] + 3*x[1]["groups_3"] + 4*x[1]["groups_4"]
        ) / x[1]["n_puzzles"] if x[1]["n_puzzles"] > 0 else 0)
        avg_groups = (best_avg[1]["groups_1"] + 2*best_avg[1]["groups_2"] + 3*best_avg[1]["groups_3"] + 4*best_avg[1]["groups_4"]) / best_avg[1]["n_puzzles"]
        console.print(f"📈 Best Average Groups: [bold]{best_avg[0]}[/] ({avg_groups:.2f} groups/puzzle)")
    
    console.print(f"\n[dim]Note: Current metrics show 'one-shot' performance (no feedback loop).[/]")
    console.print(f"[dim]For interactive simulation (like real game), see run_interactive.py (coming soon)[/]")
    
    # Export to CSV if requested
    if export_csv:
        import csv
        with open(export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "n_puzzles", "exact_rate", "avg_groups_correct",
                "groups_0", "groups_1", "groups_2", "groups_3", "groups_4",
                "pct_at_least_1", "pct_at_least_2", "pct_at_least_3"
            ])
            for model_name, stats in sorted_models:
                n = stats["n_puzzles"]
                avg_groups = (stats["groups_1"] + 2*stats["groups_2"] + 3*stats["groups_3"] + 4*stats["groups_4"]) / n if n > 0 else 0
                at_least_1 = (n - stats["groups_0"]) / n if n > 0 else 0
                at_least_2 = (stats["groups_2"] + stats["groups_3"] + stats["groups_4"]) / n if n > 0 else 0
                at_least_3 = (stats["groups_3"] + stats["groups_4"]) / n if n > 0 else 0
                
                writer.writerow([
                    model_name,
                    stats["n_puzzles"],
                    f"{stats['exact_rate']:.4f}",
                    f"{avg_groups:.2f}",
                    stats["groups_0"],
                    stats["groups_1"],
                    stats["groups_2"],
                    stats["groups_3"],
                    stats["groups_4"],
                    f"{at_least_1:.4f}",
                    f"{at_least_2:.4f}",
                    f"{at_least_3:.4f}",
                ])
        console.print(f"\n✓ Exported to: {export_csv}")


def cli():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli()