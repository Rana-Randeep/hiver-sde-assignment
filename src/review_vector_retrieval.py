from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    PROJECT_ROOT
    / "results"
    / "vector_retrieval_review.csv"
)


# ============================================================
# LOAD REVIEW FILE
# ============================================================

df = pd.read_csv(REVIEW_PATH)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "customer_tweet_id",
    "golden_customer_text",
    "golden_intent",
    "vector_top1_distance",
    "vector_top1_customer_text",
    "vector_top1_response",
    "vector_top1_tweet_id",
    "vector_relevance",
    "review_notes",
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
# NORMALIZE EXISTING LABELS
# ============================================================

df["vector_relevance"] = pd.to_numeric(
    df["vector_relevance"],
    errors="coerce",
)
df["review_notes"] = df["review_notes"].fillna("").astype(str)



# ============================================================
# SAVE FUNCTION
# ============================================================

def save_progress():
    df.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# REVIEW LOOP
# ============================================================

while True:

    # Find first unanswered example.
    unanswered = df[
        df["vector_relevance"].isna()
    ]

    if unanswered.empty:

        print()
        print("=" * 70)
        print("ALL EXAMPLES HAVE BEEN REVIEWED")
        print("=" * 70)

        break

    row_index = unanswered.index[0]
    row = df.loc[row_index]

    reviewed = (
        df["vector_relevance"]
        .notna()
        .sum()
    )

    total = len(df)

    print()
    print("=" * 80)
    print(
        f"VECTOR RETRIEVAL REVIEW "
        f"[{reviewed + 1}/{total}]"
    )
    print("=" * 80)

    print()
    print("Golden customer message:")
    print("-" * 80)
    print(row["golden_customer_text"])

    print()
    print("Golden intent:")
    print("-" * 80)
    print(row["golden_intent"])

    print()
    print("Vector top-1 distance:")
    print("-" * 80)
    print(
        f"{float(row['vector_top1_distance']):.4f}"
    )

    print()
    print("Retrieved historical customer message:")
    print("-" * 80)
    print(row["vector_top1_customer_text"])

    print()
    print("Historical GloCare response:")
    print("-" * 80)
    print(row["vector_top1_response"])

    print()
    print("Retrieved tweet ID:")
    print("-" * 80)
    print(row["vector_top1_tweet_id"])

    print()
    print("-" * 80)
    print("RELEVANCE")
    print("-" * 80)
    print("0 = irrelevant")
    print("1 = useful")
    print("2 = clearly relevant")
    print("s = skip")
    print("q = quit and save")

    while True:

        choice = input(
            "\nEnter 0, 1, 2, s, or q: "
        ).strip().lower()

        if choice == "q":

            save_progress()

            print()
            print(
                "Progress saved. "
                "You can resume later."
            )

            raise SystemExit

        if choice == "s":

            print(
                "Skipped. This example remains "
                "unlabelled."
            )

            break

        if choice in {"0", "1", "2"}:

            df.at[
                row_index,
                "vector_relevance"
            ] = int(choice)

            note = input(
                "Optional review note "
                "(press Enter for none): "
            ).strip()

            df.at[
                row_index,
                "review_notes"
            ] = note

            save_progress()

            print(
                f"Saved relevance={choice}"
            )

            break

        print(
            "Invalid input. Please enter "
            "0, 1, 2, s, or q."
        )