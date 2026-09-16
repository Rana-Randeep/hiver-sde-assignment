import pandas as pd


INPUT_PATH = "results/pilot_candidate_pool.csv"
OUTPUT_PATH = "results/pilot_candidates_for_labeling.csv"

PILOT_SIZE = 100
RANDOM_STATE = 42


# ============================================================
# Load candidate pool
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("PILOT CANDIDATE INSPECTION")
print("=" * 70)

print(f"Candidate pool rows: {len(df)}")


# ============================================================
# Select pilot examples
#
# We want diversity across:
#   - semantic clusters
#   - message length
#   - multiple responses
#   - duplicate text
#   - sampling priority
#
# This is still NOT final golden-set selection.
# ============================================================

selected_indices = set()


def add_rows(rows, n):
    """Add up to n rows without exceeding the pilot-size budget."""

    available = rows[
        ~rows.index.isin(selected_indices)
    ]

    if available.empty:
        return

    # Remaining capacity in the pilot sample.
    remaining_capacity = PILOT_SIZE - len(selected_indices)

    if remaining_capacity <= 0:
        return

    # Never select more than:
    # 1. requested rows
    # 2. available rows
    # 3. remaining pilot capacity
    n = min(
        n,
        len(available),
        remaining_capacity
    )

    sampled = available.sample(
        n=n,
        random_state=RANDOM_STATE + len(selected_indices)
    )

    selected_indices.update(
        sampled.index.tolist()
    )


# ============================================================
# 1. Cover every semantic cluster
# ============================================================

for cluster_id in sorted(df["cluster_id"].unique()):

    cluster_rows = df[
        df["cluster_id"] == cluster_id
    ].sort_values(
        by="sampling_priority",
        ascending=False
    )

    add_rows(cluster_rows, 5)


# ============================================================
# 2. Add difficult / informative examples
# ============================================================

add_rows(
    df[df["has_multiple_responses"]]
    .sort_values(
        by="sampling_priority",
        ascending=False
    ),
    15
)

add_rows(
    df[df["has_duplicate_text"]]
    .sort_values(
        by="sampling_priority",
        ascending=False
    ),
    15
)

add_rows(
    df[df["length_bucket"] == "short"]
    .sort_values(
        by="sampling_priority",
        ascending=False
    ),
    15
)

add_rows(
    df[df["length_bucket"] == "long"]
    .sort_values(
        by="sampling_priority",
        ascending=False
    ),
    10
)


# ============================================================
# 3. Fill remaining slots
# ============================================================

remaining = PILOT_SIZE - len(selected_indices)

if remaining > 0:

    add_rows(
        df.sort_values(
            by="sampling_priority",
            ascending=False
        ),
        remaining
    )


# ============================================================
# Create pilot dataframe
# ============================================================

pilot = df.loc[
    sorted(selected_indices)
].copy()


# Shuffle rows so selection order is not obvious
pilot = pilot.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# Create fresh pilot IDs
# ============================================================

pilot.insert(
    0,
    "pilot_id",
    range(1, len(pilot) + 1)
)


# ============================================================
# Add labelling columns
# ============================================================

pilot["pilot_intent"] = ""
pilot["label_confidence"] = ""
pilot["label_notes"] = ""


# ============================================================
# Keep useful columns
# ============================================================

columns = [
    "pilot_id",
    "customer_tweet_id",
    "customer_text",
    "brand_response",
    "response_count",
    "has_multiple_responses",
    "text_frequency",
    "has_duplicate_text",
    "text_length",
    "length_bucket",
    "cluster_id",
    "sampling_priority",
    "pilot_intent",
    "label_confidence",
    "label_notes"
]

pilot = pilot[columns]


# ============================================================
# Save
# ============================================================

pilot.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("PILOT SAMPLE")
print("=" * 70)

print(f"Pilot rows: {len(pilot)}")

print()
print("Cluster distribution:")
print(
    pilot["cluster_id"]
    .value_counts()
    .sort_index()
)

print()
print("Length distribution:")
print(
    pilot["length_bucket"]
    .value_counts()
)

print()
print("Multiple-response examples:")
print(
    pilot["has_multiple_responses"]
    .value_counts()
)

print()
print("Duplicate-text examples:")
print(
    pilot["has_duplicate_text"]
    .value_counts()
)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Saved to: {OUTPUT_PATH}")