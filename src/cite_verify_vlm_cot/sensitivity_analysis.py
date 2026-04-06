"""
sensitivity_analysis.py
───────────────────────
Experiment 4 — CaVeScore weight sensitivity analysis.

Usage
-----
Run this script after you have collected experiment results from Phoenix
(or any source) and exported the per-sample metric components to a JSON or
CSV file.

    # From saved JSON (list of per-sample dicts)
    python sensitivity_analysis.py --results-file outputs/cave_results.json

    # From a Phoenix-exported CSV
    python sensitivity_analysis.py --results-file outputs/cave_results.csv --format csv

    # Quick smoke-test with synthetic data
    python sensitivity_analysis.py --smoke-test

Output
------
Prints a ranked table of weight configurations × aggregate statistics,
and writes the results to outputs/weight_sensitivity.json and
outputs/weight_sensitivity.csv.

Required keys in each per-sample dict
--------------------------------------
    accuracy            float  0–1   final answer correct?
    citation_precision  float  0–1   (text + QI) citation precision
    citation_recall     float  0–1   citations vs factual sentences
    ais                 float  0–1   NLI-based attribution score
    grounding_score     float  0–1   evidence grounding check

These are exactly the keys returned by ``compute_cave_score(state)``.
"""

import argparse
import json
import os
import sys
import csv

# Make sure the project root is on the path when run directly.
sys.path.insert(0, os.path.dirname(__file__))

from solver.solver_evals import WEIGHT_CONFIGS, run_weight_sensitivity_analysis


# I/O helpers

def load_results_json(path: str):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Phoenix sometimes wraps results in {"results": [...]}
        data = data.get("results", list(data.values()))
    return data


def load_results_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                k: float(v) if v not in ("", None) else 0.0
                for k, v in row.items()
            })
    return rows


def _synthetic_samples(n: int = 200, seed: int = 42):
    """Generate synthetic per-sample metric dicts for smoke-testing."""
    import random
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        acc = rng.choices([0.0, 1.0], weights=[0.35, 0.65])[0]
        samples.append({
            "accuracy":           acc,
            "citation_precision": rng.uniform(0.4, 1.0) if acc else rng.uniform(0.0, 0.6),
            "citation_recall":    rng.uniform(0.3, 0.9),
            "ais":                rng.uniform(0.4, 1.0) if acc else rng.uniform(0.0, 0.5),
            "grounding_score":    rng.uniform(0.4, 1.0) if acc else rng.uniform(0.0, 0.5),
        })
    return samples


# Formatting

_COL_WIDTHS = {
    "config":           22,
    "mean":              8,
    "std":               6,
    "min":               6,
    "max":               6,
    "delta":             8,
    "w_acc":             6,
    "w_cprec":           7,
    "w_crec":            7,
    "w_ais":             6,
    "w_gnd":             6,
}

def _header():
    h = (
        f"{'Config':<{_COL_WIDTHS['config']}}"
        f"{'Mean':>{_COL_WIDTHS['mean']}}"
        f"{'Std':>{_COL_WIDTHS['std']}}"
        f"{'Min':>{_COL_WIDTHS['min']}}"
        f"{'Max':>{_COL_WIDTHS['max']}}"
        f"{'Δdefault':>{_COL_WIDTHS['delta']}}"
        f"  "
        f"{'w_acc':>{_COL_WIDTHS['w_acc']}}"
        f"{'w_cprec':>{_COL_WIDTHS['w_cprec']}}"
        f"{'w_crec':>{_COL_WIDTHS['w_crec']}}"
        f"{'w_ais':>{_COL_WIDTHS['w_ais']}}"
        f"{'w_gnd':>{_COL_WIDTHS['w_gnd']}}"
    )
    return h


def _row(config_name: str, stats: dict) -> str:
    w = stats["weights"]
    delta = stats["delta_vs_default"]
    delta_str = f"{delta:+.4f}"
    return (
        f"{config_name:<{_COL_WIDTHS['config']}}"
        f"{stats['mean_cave_score']:>{_COL_WIDTHS['mean']}.4f}"
        f"{stats['std_cave_score']:>{_COL_WIDTHS['std']}.4f}"
        f"{stats['min_cave_score']:>{_COL_WIDTHS['min']}.4f}"
        f"{stats['max_cave_score']:>{_COL_WIDTHS['max']}.4f}"
        f"{delta_str:>{_COL_WIDTHS['delta']}}"
        f"  "
        f"{w['accuracy']:>{_COL_WIDTHS['w_acc']}.2f}"
        f"{w['citation_precision']:>{_COL_WIDTHS['w_cprec']}.2f}"
        f"{w['citation_recall']:>{_COL_WIDTHS['w_crec']}.2f}"
        f"{w['ais']:>{_COL_WIDTHS['w_ais']}.2f}"
        f"{w['grounding']:>{_COL_WIDTHS['w_gnd']}.2f}"
    )


def print_table(analysis: dict):
    sep = "─" * 85
    print(f"\n{sep}")
    print("CaVeScore Weight Sensitivity Analysis")
    print(sep)
    print(_header())
    print(sep)
    # Sort: default first, then descending mean
    rows = sorted(
        analysis.items(),
        key=lambda kv: (kv[0] != "default", -kv[1]["mean_cave_score"]),
    )
    for name, stats in rows:
        print(_row(name, stats))
    print(sep)
    print(f"  n_samples = {list(analysis.values())[0]['n_samples']}")
    print()


# Saving

def save_results(analysis: dict, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # JSON — full precision
    json_path = os.path.join(out_dir, "weight_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved JSON → {json_path}")

    # CSV — flat summary table
    csv_path = os.path.join(out_dir, "weight_sensitivity.csv")
    fieldnames = [
        "config", "mean_cave_score", "std_cave_score",
        "min_cave_score", "max_cave_score", "delta_vs_default",
        "n_samples",
        "w_accuracy", "w_citation_precision", "w_citation_recall",
        "w_ais", "w_grounding",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in analysis.items():
            w = stats["weights"]
            writer.writerow({
                "config":              name,
                "mean_cave_score":     round(stats["mean_cave_score"],  4),
                "std_cave_score":      round(stats["std_cave_score"],   4),
                "min_cave_score":      round(stats["min_cave_score"],   4),
                "max_cave_score":      round(stats["max_cave_score"],   4),
                "delta_vs_default":    round(stats["delta_vs_default"], 4),
                "n_samples":           stats["n_samples"],
                "w_accuracy":          w["accuracy"],
                "w_citation_precision": w["citation_precision"],
                "w_citation_recall":   w["citation_recall"],
                "w_ais":               w["ais"],
                "w_grounding":         w["grounding"],
            })
    print(f"Saved CSV  → {csv_path}")


# Add a custom weight configuration at runtime

def add_custom_config(name: str, acc: float, cprec: float, crec: float,
                      ais: float, gnd: float) -> None:
    """
    Register a one-off weight configuration at runtime.
    Example::
        python sensitivity_analysis.py \\
            --smoke-test \\
            --custom-config my_config 0.5 0.15 0.15 0.1 0.1
    Weights must sum to 1.0 (checked by run_weight_sensitivity_analysis).
    """
    WEIGHT_CONFIGS[name] = {
        "accuracy":           acc,
        "citation_precision": cprec,
        "citation_recall":    crec,
        "ais":                ais,
        "grounding":          gnd,
    }
    print(f"Registered custom config '{name}': {WEIGHT_CONFIGS[name]}")


# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="CaVeScore weight sensitivity analysis (Experiment 4)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results-file", type=str,
        help="Path to per-sample metric file (JSON list or CSV with header row).",
    )
    source.add_argument(
        "--smoke-test", action="store_true",
        help="Run on 200 synthetic samples for a quick sanity check.",
    )

    parser.add_argument(
        "--format", type=str, default="json", choices=["json", "csv"],
        help="File format when --results-file is used (default: json).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs",
        help="Directory to write weight_sensitivity.json / .csv (default: outputs).",
    )
    parser.add_argument(
        "--custom-config", nargs=6,
        metavar=("NAME", "W_ACC", "W_CPREC", "W_CREC", "W_AIS", "W_GND"),
        help=(
            "Add a custom weight config before running. Provide the config name "
            "followed by five floats that sum to 1.0. "
            "Example: --custom-config my_cfg 0.5 0.15 0.15 0.1 0.1"
        ),
    )
    args = parser.parse_args()

    # Optional custom config
    if args.custom_config:
        name, *vals = args.custom_config
        add_custom_config(name, *[float(v) for v in vals])

    # Load data
    if args.smoke_test:
        print("Running smoke test with 200 synthetic samples...")
        samples = _synthetic_samples()
    elif args.format == "csv":
        samples = load_results_csv(args.results_file)
    else:
        samples = load_results_json(args.results_file)

    print(f"Loaded {len(samples)} samples.")

    # Run analysis across all named weight configs
    analysis = run_weight_sensitivity_analysis(samples)

    # Display + save
    print_table(analysis)
    save_results(analysis, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
