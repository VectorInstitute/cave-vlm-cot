"""
Aggregate metrics from multiple Phoenix experiment CSV exports (one per shard)
into a single summary table — equivalent to what Phoenix would show for a
single n=5000 run.

Phoenix CSV format (confirmed from export):
  - Each evaluator produces 9 columns: annotation_{name}_score,
    annotation_{name}_label, annotation_{name}_explanation, etc.
  - The score column for eval_cave_score is annotation_eval_cave_score_score
    (the evaluator name already ends in _score, so Phoenix appends another _score).

Usage
-----
    # Point at one or more CSV files
    python aggregate_shards.py shard0.csv shard1.csv shard2.csv

    # Or use a glob
    python aggregate_shards.py "exports/shard_*.csv"

    # Save combined per-row CSV
    python aggregate_shards.py shard*.csv --save-csv combined.csv

How to get the CSVs from Phoenix
---------------------------------
    1. Open your experiment in the Phoenix UI
    2. Click the download/export button (top-right of the experiment table)
    3. Select "Export as CSV"
    4. Repeat for each shard experiment
    5. Run this script pointing at those files

To execute:
python aggregate_shards.py shard0.csv shard1.csv shard2.csv

# Or if you save them with a pattern:
python aggregate_shards.py "exports/cave_shard_*.csv"

# To also get the full per-row combined CSV:
python aggregate_shards.py shard0.csv shard1.csv shard2.csv --save-csv combined_5000.csv
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# CLI
parser = argparse.ArgumentParser(description="Aggregate CaVe-VLM-CoT shard CSVs")
parser.add_argument("csvs", nargs="+",
                    help="One or more Phoenix experiment CSV export files (supports globs)")
parser.add_argument("--save-csv", default=None,
                    help="Save the combined per-row DataFrame to this path")
args = parser.parse_args()

# Expand any globs (needed on Windows; Linux shells expand automatically)
csv_paths = []
for pattern in args.csvs:
    expanded = glob.glob(pattern)
    csv_paths.extend(expanded if expanded else [pattern])

csv_paths = [p for p in csv_paths if Path(p).exists()]
if not csv_paths:
    sys.exit(f"No CSV files found matching: {args.csvs}")


# Load and concatenate
dfs = []
for path in csv_paths:
    df = pd.read_csv(path)
    df["_source_file"] = Path(path).name
    dfs.append(df)
    print(f"  Loaded {len(df):>5} rows  <- {path}")

combined = pd.concat(dfs, ignore_index=True)
print(f"\nCombined: {len(combined)} rows from {len(dfs)} file(s)\n")

# Warn about duplicates (same example_id appearing in multiple shards)
if "example_id" in combined.columns:
    dupes = combined["example_id"].duplicated().sum()
    if dupes:
        print(f"[WARN] {dupes} duplicate example_ids detected — "
              f"rows may overlap across shards. Dropping duplicates.\n")
        combined = combined.drop_duplicates(subset="example_id", keep="first")
        print(f"After dedup: {len(combined)} rows\n")


# Identify score columns
# Phoenix naming: annotation_{evaluator_name}_score
score_cols = [c for c in combined.columns
              if c.startswith("annotation_") and c.endswith("_score")]

if not score_cols:
    sys.exit(
        "No 'annotation_*_score' columns found. "
        "Make sure you exported the full CSV from Phoenix (not just the summary)."
    )

# Map column to clean display name
# annotation_eval_accuracy_score    -> eval_accuracy
# annotation_eval_cave_score_score  -> eval_cave_score  (strip one _score suffix)
def col_to_name(col):
    name = col[len("annotation_"):]   # strip annotation_ prefix
    if name.endswith("_score"):
        name = name[:-len("_score")]  # strip trailing _score
    return name


# Compute and print aggregate metrics
WIDTH = 47

print("=" * (WIDTH + 20))
print(f" CaVe-VLM-CoT  Aggregated ({len(dfs)} shard(s), n={len(combined)})")
print("=" * (WIDTH + 20))

# Primary metrics in a meaningful display order, then anything else
PRIMARY = [
    "annotation_eval_accuracy_score",
    "annotation_eval_cave_score_score",
    "annotation_eval_citation_precision_score",
    "annotation_eval_citation_recall_score",
    "annotation_eval_ais_score",
    "annotation_eval_hallucination_rate_score",
    "annotation_eval_grounding_score_score",
    "annotation_eval_is_grounded_score",
    "annotation_eval_planner_hit_score",
    "annotation_eval_planner_coverage_score",
    "annotation_eval_recall_score",
    "annotation_eval_recall_pass_score",
    "annotation_eval_precision_score",
    "annotation_eval_mrr_score",
    "annotation_eval_ndcg_score",
    "annotation_eval_text_citation_precision_score",
    "annotation_eval_qi_citation_coverage_score",
    "annotation_eval_qi_citation_count_score",
    "annotation_eval_qi_citation_precision_score",
    "annotation_eval_decision_correct_score",
    "annotation_eval_hallucination_detection_correct_score",
    "annotation_eval_confidence_appropriate_score",
    "annotation_eval_feedback_quality_score",
    "annotation_eval_subquery_count_score",
    "annotation_eval_num_question_images_score",
    "annotation_eval_coverage_pass_score",
]

printed = set()
for col in PRIMARY + sorted(score_cols):
    if col in printed or col not in combined.columns:
        continue
    printed.add(col)
    series = combined[col].dropna()
    mean_val = series.mean()
    n = len(series)
    name = col_to_name(col)
    print(f"  {name:<{WIDTH}}  {mean_val:.4f}  (n={n})")

print("=" * (WIDTH + 20))

# Per-source-file breakdown (useful for checking shard balance)
acc_col = "annotation_eval_accuracy_score"
if acc_col in combined.columns and "_source_file" in combined.columns:
    print("\nPer-shard accuracy breakdown:")
    for src, grp in combined.groupby("_source_file"):
        acc = grp[acc_col].dropna().mean()
        n = len(grp)
        print(f"  {src:<55}  acc={acc:.4f}  (n={n})")

# Save combined CSV
if args.save_csv:
    combined.to_csv(args.save_csv, index=False)
    print(f"\nCombined per-row CSV saved -> {args.save_csv}")