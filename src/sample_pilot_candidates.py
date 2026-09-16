import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "data/glocare_pairs.csv"
OUTPUT_PATH = "results/pilot_candidate_pool.csv"

RANDOM_STATE = 42

# Size of the candidate pool.
CANDIDATE_POOL_SIZE = 250

# Number of semantic clusters used only for sampling diversity.
N_CLUSTERS = 12


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# Basic preparation
# ============================================================

df = df.copy()

df["customer_text"] = df["customer_text"].astype(str)

df["text_length"] = df["customer_text"].str.len()

df["word_count"] = df["customer_text"].str.split().str.len()


# ============================================================
# Identify customer messages with multiple responses
# ============================================================

response_count = (
    df.groupby("customer_tweet_id")
      .size()
      .rename("response_count")
)

df = df.merge(
    response_count,
    left_on="customer_tweet_id",
    right_index=True,
    how="left"
)

df["has_multiple_responses"] = df["response_count"] > 1


# ============================================================
# Identify duplicate customer texts
# ============================================================

text_frequency = (
    df.groupby("customer_text")["customer_text"]
      .transform("size")
)

df["text_frequency"] = text_frequency

df["has_duplicate_text"] = df["text_frequency"] > 1


# ============================================================
# Length buckets
# ============================================================

df["length_bucket"] = pd.qcut(
    df["text_length"],
    q=4,
    labels=["short", "medium_short", "medium_long", "long"],
    duplicates="drop"
)


# ============================================================
# TF-IDF semantic representation
#
# IMPORTANT:
# KMeans is being used only as a sampling/diversity signal.
# It is NOT an intent label.
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_features=5000
)

X = vectorizer.fit_transform(df["customer_text"])


# ============================================================
# KMeans clustering
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster_id"] = kmeans.fit_predict(X)


# ============================================================
# Build sampling priority
#
# Higher score = more useful for manual inspection.
# ============================================================

df["sampling_priority"] = 0

# Multiple-response examples are useful for inspecting
# context-rich support situations.
df.loc[
    df["has_multiple_responses"],
    "sampling_priority"
] += 3

# Duplicate texts are useful for leakage / correlation inspection.
df.loc[
    df["has_duplicate_text"],
    "sampling_priority"
] += 2

# Very short messages are often ambiguous/context-dependent.
df.loc[
    df["text_length"] <= 40,
    "sampling_priority"
] += 2

# Very long messages may contain richer context.
df.loc[
    df["text_length"] >= 180,
    "sampling_priority"
] += 1


# ============================================================
# Deterministic random generator
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# Sampling strategy
#
# We first guarantee representation from every cluster.
# Then we add targeted examples from useful edge cases.
# Finally we fill the remaining slots randomly.
# ============================================================

selected_indices = set()


def add_random_rows(candidate_df, n):
    """
    Add up to n random rows from candidate_df that have
    not already been selected.
    """
    available = candidate_df[
        ~candidate_df.index.isin(selected_indices)
    ]

    if available.empty:
        return

    n = min(n, len(available))

    sampled_indices = rng.choice(
        available.index.to_numpy(),
        size=n,
        replace=False
    )

    selected_indices.update(sampled_indices.tolist())


# ============================================================
# Stage 1:
# Semantic diversity
#
# Take up to 10 examples from every cluster.
# ============================================================

for cluster_id in sorted(df["cluster_id"].unique()):

    cluster_rows = df[
        df["cluster_id"] == cluster_id
    ].sort_values(
        by="sampling_priority",
        ascending=False
    )

    add_random_rows(cluster_rows, 10)


# ============================================================
# Stage 2:
# Targeted difficult / informative cases
# ============================================================

# Multiple-response examples
multi_response_rows = df[
    df["has_multiple_responses"]
].sort_values(
    by="sampling_priority",
    ascending=False
)

add_random_rows(multi_response_rows, 30)


# Duplicate customer-text examples
duplicate_text_rows = df[
    df["has_duplicate_text"]
].sort_values(
    by="sampling_priority",
    ascending=False
)

add_random_rows(duplicate_text_rows, 25)


# Very short messages
short_rows = df[
    df["text_length"] <= 40
].sort_values(
    by="sampling_priority",
    ascending=False
)

add_random_rows(short_rows, 30)


# Very long / information-rich messages
long_rows = df[
    df["text_length"] >= 180
].sort_values(
    by="sampling_priority",
    ascending=False
)

add_random_rows(long_rows, 20)


# ============================================================
# Stage 3:
# Fill remaining slots randomly
# ============================================================

remaining = CANDIDATE_POOL_SIZE - len(selected_indices)

if remaining > 0:
    add_random_rows(df, remaining)


# ============================================================
# Create candidate pool
# ============================================================

candidate_pool = df.loc[
    sorted(selected_indices)
].copy()


# Shuffle final pool so sampling categories are not grouped
candidate_pool = candidate_pool.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# Add a simple candidate ID for manual labelling.
candidate_pool.insert(
    0,
    "candidate_id",
    range(1, len(candidate_pool) + 1)
)


# ============================================================
# Add blank pilot-label column
# ============================================================

candidate_pool["pilot_intent"] = ""


# ============================================================
# Select useful columns for manual inspection
# ============================================================

output_columns = [
    "candidate_id",
    "customer_tweet_id",
    "customer_text",
    "brand_response",
    "customer_created_at",
    "brand_created_at",
    "response_count",
    "has_multiple_responses",
    "text_frequency",
    "has_duplicate_text",
    "text_length",
    "word_count",
    "length_bucket",
    "cluster_id",
    "sampling_priority",
    "pilot_intent"
]

candidate_pool = candidate_pool[output_columns]


# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

candidate_pool.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Reporting
# ============================================================

print()
print("=" * 70)
print("CANDIDATE POOL")
print("=" * 70)

print(f"Candidate rows: {len(candidate_pool)}")

print()
print("Cluster distribution:")
print(
    candidate_pool["cluster_id"]
    .value_counts()
    .sort_index()
)

print()
print("Length distribution:")
print(
    candidate_pool["length_bucket"]
    .value_counts()
)

print()
print("Multiple-response examples:")
print(
    candidate_pool["has_multiple_responses"]
    .value_counts()
)

print()
print("Duplicate-text examples:")
print(
    candidate_pool["has_duplicate_text"]
    .value_counts()
)

print()
print("Priority distribution:")
print(
    candidate_pool["sampling_priority"]
    .value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Saved to: {OUTPUT_PATH}")