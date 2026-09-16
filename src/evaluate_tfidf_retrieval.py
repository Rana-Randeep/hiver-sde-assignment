import json
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_FILE = PROJECT_ROOT / "results" / "tfidf_retrieval_golden_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "tfidf_retrieval_metrics.json"


# -------------------------------------------------------------------
# Load human relevance judgments
# -------------------------------------------------------------------

df = pd.read_csv(REVIEW_FILE)

if len(df) != 156:
    raise ValueError(
        f"Expected 156 golden examples, but found {len(df)}."
    )

if df["relevance"].isna().any():
    raise ValueError("Some relevance values are still blank.")

valid_scores = {0, 1, 2}

if not set(df["relevance"].unique()).issubset(valid_scores):
    raise ValueError("Relevance must contain only 0, 1, or 2.")


# -------------------------------------------------------------------
# Calculate metrics
# -------------------------------------------------------------------

total_examples = len(df)

irrelevant_count = int((df["relevance"] == 0).sum())
somewhat_relevant_count = int((df["relevance"] == 1).sum())
clearly_relevant_count = int((df["relevance"] == 2).sum())

useful_count = int((df["relevance"] >= 1).sum())

clearly_relevant_rate = clearly_relevant_count / total_examples
useful_retrieval_rate = useful_count / total_examples
mean_relevance = float(df["relevance"].mean())


# -------------------------------------------------------------------
# Save metrics
# -------------------------------------------------------------------

metrics = {
    "evaluation_set_size": total_examples,
    "relevance_scale": {
        "0": "irrelevant",
        "1": "somewhat_relevant",
        "2": "clearly_relevant",
    },
    "counts": {
        "irrelevant": irrelevant_count,
        "somewhat_relevant": somewhat_relevant_count,
        "clearly_relevant": clearly_relevant_count,
        "useful": useful_count,
    },
    "metrics": {
        "clearly_relevant_rate": round(clearly_relevant_rate, 4),
        "useful_retrieval_rate": round(useful_retrieval_rate, 4),
        "mean_relevance": round(mean_relevance, 4),
    },
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


# -------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------

print("=" * 70)
print("TF-IDF HISTORICAL RESPONSE RETRIEVAL EVALUATION")
print("=" * 70)

print(f"\nEvaluation set size: {total_examples}")

print("\nRelevance distribution:")
print(f"  0 - Irrelevant:          {irrelevant_count}")
print(f"  1 - Somewhat relevant:   {somewhat_relevant_count}")
print(f"  2 - Clearly relevant:    {clearly_relevant_count}")

print("\nMetrics:")
print(f"  Clearly relevant rate:   {clearly_relevant_rate:.2%}")
print(f"  Useful retrieval rate:   {useful_retrieval_rate:.2%}")
print(f"  Mean relevance:          {mean_relevance:.3f}")

print("\nMetrics file created:")
print(OUTPUT_FILE)