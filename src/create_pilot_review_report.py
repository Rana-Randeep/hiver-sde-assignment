import pandas as pd


INPUT_PATH = "results/pilot_candidates_for_labeling.csv"
OUTPUT_PATH = "results/pilot_review_report.csv"


# ============================================================
# Load pilot data
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("CREATING PILOT REVIEW REPORT")
print("=" * 70)

print(f"Input rows: {len(df)}")


# ============================================================
# Review priority
#
# The purpose is NOT to say that a label is wrong.
# It only identifies examples that deserve closer inspection.
# ============================================================

df["review_priority"] = "normal"


# Low confidence = highest review priority
df.loc[
    df["label_confidence"].astype(str).str.lower() == "low",
    "review_priority"
] = "high"


# Medium confidence = medium review priority
df.loc[
    (
        df["label_confidence"].astype(str).str.lower() == "medium"
    )
    &
    (
        df["review_priority"] == "normal"
    ),
    "review_priority"
] = "medium"


# ============================================================
# Add review reasons
# ============================================================

df["review_reason"] = ""


df.loc[
    df["label_confidence"].astype(str).str.lower() == "low",
    "review_reason"
] = "Low annotation confidence"


df.loc[
    (
        df["label_confidence"].astype(str).str.lower() == "medium"
    )
    &
    (df["review_reason"] == ""),
    "review_reason"
] = "Medium annotation confidence"


# Multiple-response cases deserve contextual inspection
mask = (
    df["has_multiple_responses"].astype(str).str.lower() == "true"
)

df.loc[
    mask & (df["review_reason"] == ""),
    "review_reason"
] = "Multiple historical responses"


# Duplicate-text cases deserve correlation inspection
mask = (
    df["has_duplicate_text"].astype(str).str.lower() == "true"
)

df.loc[
    mask & (df["review_reason"] == ""),
    "review_reason"
] = "Duplicate customer text"


# Very short messages can be context dependent
mask = df["text_length"] <= 40

df.loc[
    mask & (df["review_reason"] == ""),
    "review_reason"
] = "Very short customer message"


# ============================================================
# Add a simple ambiguity flag
#
# This is only a review aid.
# It does NOT claim that the example is incorrectly labelled.
# ============================================================

df["needs_taxonomy_review"] = False

df.loc[
    df["label_confidence"].astype(str).str.lower().isin(
        ["low", "medium"]
    ),
    "needs_taxonomy_review"
] = True


# ============================================================
# Sort:
#   1. taxonomy-review cases first
#   2. high priority before medium
#   3. pilot ID for reproducibility
# ============================================================

priority_order = {
    "high": 0,
    "medium": 1,
    "normal": 2
}

df["_priority_sort"] = df["review_priority"].map(
    priority_order
)

df = df.sort_values(
    by=["_priority_sort", "pilot_id"]
).drop(
    columns=["_priority_sort"]
)


# ============================================================
# Select report columns
# ============================================================

columns = [
    "pilot_id",
    "customer_tweet_id",
    "customer_text",
    "brand_response",

    "pilot_intent",
    "label_confidence",
    "label_notes",

    "review_priority",
    "review_reason",
    "needs_taxonomy_review",

    "cluster_id",
    "length_bucket",
    "text_length",
    "has_multiple_responses",
    "has_duplicate_text",
    "response_count",
    "text_frequency",
    "sampling_priority"
]

report = df[columns]


# ============================================================
# Save
# ============================================================

report.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("REVIEW SUMMARY")
print("=" * 70)

print()
print("Review priority:")
print(
    report["review_priority"]
    .value_counts()
)

print()
print("Taxonomy review required:")
print(
    report["needs_taxonomy_review"]
    .value_counts()
)

print()
print("Intent distribution:")
print(
    report["pilot_intent"]
    .value_counts()
)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Saved to: {OUTPUT_PATH}")