from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "vector_retrieval_golden_review.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "vector_retrieval_review.csv"
)


# ============================================================
# LOAD VECTOR RESULTS
# ============================================================

print("Loading vector retrieval results...")

df = pd.read_csv(INPUT_PATH)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_tweet_id",
    "golden_customer_text",
    "golden_intent",
    "vector_top1_distance",
    "vector_top1_customer_text",
    "vector_top1_response",
    "vector_top1_tweet_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CREATE HUMAN REVIEW FILE
# ============================================================

review = df[
    [
        "customer_tweet_id",
        "golden_customer_text",
        "golden_intent",
        "vector_top1_distance",
        "vector_top1_customer_text",
        "vector_top1_response",
        "vector_top1_tweet_id",
    ]
].copy()


# Columns to be completed during human review.
review["vector_relevance"] = pd.NA
review["review_notes"] = ""


# ============================================================
# SAVE REVIEW FILE
# ============================================================

review.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VECTOR HUMAN REVIEW FILE CREATED")
print("=" * 70)

print(f"Review rows: {len(review):,}")

print()
print("Output:")
print(OUTPUT_PATH)

print()
print("Review columns:")
for column in review.columns:
    print(f"- {column}")

print()
print("Relevance labels:")
print("0 = irrelevant")
print("1 = useful")
print("2 = clearly relevant")

print()
print(
    "IMPORTANT: vector_relevance and review_notes "
    "are intentionally blank."
)