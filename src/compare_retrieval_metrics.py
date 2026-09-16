from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TFIDF_PATH = (
    PROJECT_ROOT
    / "results"
    / "tfidf_retrieval_golden_review.csv"
)

VECTOR_PATH = (
    PROJECT_ROOT
    / "results"
    / "vector_retrieval_golden_review.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading TF-IDF evaluation results...")

tfidf = pd.read_csv(TFIDF_PATH)

print(
    f"TF-IDF examples: {len(tfidf):,}"
)

print()
print("Loading vector evaluation results...")

vector = pd.read_csv(VECTOR_PATH)

print(
    f"Vector examples: {len(vector):,}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(tfidf) != len(vector):
    raise ValueError(
        "TF-IDF and vector evaluation sets "
        "have different numbers of examples."
    )

if not (
    tfidf["customer_tweet_id"].values
    == vector["customer_tweet_id"].values
).all():
    raise ValueError(
        "TF-IDF and vector examples are not aligned."
    )


# ============================================================
# RELEVANCE LABELS
# ============================================================

relevance = tfidf["relevance"]


print()
print("=" * 70)
print("GOLDEN SET RELEVANCE DISTRIBUTION")
print("=" * 70)

print(
    relevance.value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# TF-IDF METRICS
# ============================================================

tfidf_mean = tfidf["relevance"].mean()

tfidf_clearly_relevant = (
    tfidf["relevance"] == 2
).mean()

tfidf_useful = (
    tfidf["relevance"] >= 1
).mean()


# ============================================================
# VECTOR METRICS
# ============================================================

# The vector retrieval file currently contains the original
# human relevance label for each golden example.
#
# At this stage, the vector retrieval itself has not been
# manually relabelled.
#
# Therefore, we cannot legitimately calculate vector
# relevance metrics yet from this file alone.


print()
print("=" * 70)
print("CURRENT EVALUATION STATUS")
print("=" * 70)

print()
print("TF-IDF:")
print(
    f"Mean relevance: "
    f"{tfidf_mean:.3f} / 2"
)

print(
    f"Clearly relevant: "
    f"{tfidf_clearly_relevant * 100:.2f}%"
)

print(
    f"Useful retrieval: "
    f"{tfidf_useful * 100:.2f}%"
)

print()
print("Vector retrieval:")
print(
    "Human relevance labels for the vector results "
    "have not yet been assigned."
)

print()
print(
    "Therefore a direct TF-IDF vs vector relevance "
    "comparison cannot be calculated yet."
)