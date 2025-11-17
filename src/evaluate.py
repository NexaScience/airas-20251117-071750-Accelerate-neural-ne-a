import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

plt.style.use("seaborn-v0_8-paper")


def load_history(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)
    return pd.DataFrame(data)


def auc_accuracy(df: pd.DataFrame) -> float:
    """Area under the accuracy-vs-rounds curve (trapezoidal)."""
    return sk_metrics.auc(df["round"], df["test_accuracy"])


def make_line_plot(histories: Dict[str, pd.DataFrame], results_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for run_id, df in histories.items():
        ax.plot(df["round"], df["test_accuracy"], label=run_id)
        ax.annotate(f"{df['test_accuracy'].iat[-1]*100:.2f}%",
                    xy=(df['round'].iat[-1], df['test_accuracy'].iat[-1]),
                    textcoords="offset points", xytext=(0, 5))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test Accuracy")
    ax.legend()
    plt.tight_layout()
    img_dir = results_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig_path = img_dir / "accuracy_over_rounds.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure {fig_path.relative_to(results_dir)}")
    plt.close(fig)


def make_bar_plot(final_acc: Dict[str, float], results_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    run_ids = list(final_acc.keys())
    values = [final_acc[r] for r in run_ids]
    bars = ax.bar(run_ids, values)
    ax.set_ylabel("Final Test Accuracy")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val*100:.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    img_dir = results_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig_path = img_dir / "final_accuracy.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure {fig_path.relative_to(results_dir)}")
    plt.close(fig)


def consolidate_metrics(histories: Dict[str, pd.DataFrame]):
    metrics = {}
    for run_id, df in histories.items():
        metrics[run_id] = {
            "final_accuracy": float(df["test_accuracy"].iat[-1]),
            "best_accuracy": float(df["test_accuracy"].max()),
            "auc_accuracy": float(auc_accuracy(df)),
            "num_rounds": int(df["round"].iat[-1]),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate & compare results of all experiment variations")
    parser.add_argument("--results-dir", required=True, help="Root directory containing per-run sub-directories")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # Collect histories
    histories = {}
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            histories[run_dir.name] = load_history(res_file)

    if not histories:
        print("No results.json files found – nothing to evaluate.")
        return

    # Produce figures
    make_line_plot(histories, results_dir)
    final_acc = {k: v["test_accuracy"].iat[-1] for k, v in histories.items()}
    make_bar_plot(final_acc, results_dir)

    # Consolidated metrics
    metrics = consolidate_metrics(histories)
    print(json.dumps({"comparison_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
