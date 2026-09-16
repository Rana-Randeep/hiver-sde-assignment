from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_FILES = [
    RESULTS_DIR / "golden_batch_1_annotated.csv",
    RESULTS_DIR / "golden_batch_2_annotated.csv",
    RESULTS_DIR / "golden_batch_3_annotated.csv",
]

OUTPUT_FILE = RESULTS_DIR / "golden_set_working.csv"


def main():
    print("=== MERGING ALL GOLDEN ANNOTATION BATCHES ===")

    selected_batches = []

    for batch_file in BATCH_FILES:
        if not batch_file.exists():
            raise FileNotFoundError(
                f"Missing batch file: {batch_file}"
            )

        df = pd.read_csv(batch_file)

        selected = df[
            df["golden_selected"].astype(str).str.lower().eq("yes")
        ].copy()

        print(
            f"{batch_file.name}: "
            f"{len(df)} total, "
            f"{len(selected)} selected"
        )

        selected_batches.append(selected)

    combined = pd.concat(
        selected_batches,
        ignore_index=True
    )

    print(f"\nCombined selected rows: {len(combined)}")

    # Check duplicate customer tweet IDs before removing anything.
    duplicate_mask = combined["customer_tweet_id"].duplicated(
        keep=False
    )

    duplicate_rows = combined[duplicate_mask]

    print(
        f"Duplicate customer_tweet_id rows: "
        f"{len(duplicate_rows)}"
    )

    print(
        f"Unique duplicate customer_tweet_ids: "
        f"{duplicate_rows['customer_tweet_id'].nunique()}"
    )

    if len(duplicate_rows) > 0:
        print("\nDuplicate IDs found:")
        print(
            duplicate_rows[
                [
                    "customer_tweet_id",
                    "customer_text",
                    "golden_intent",
                ]
            ].sort_values("customer_tweet_id").to_string(index=False)
        )

    # Remove duplicate customer tweets.
    before_dedup = len(combined)

    combined = combined.drop_duplicates(
        subset=["customer_tweet_id"],
        keep="first"
    ).copy()

    after_dedup = len(combined)

    print(
        f"\nRows removed during deduplication: "
        f"{before_dedup - after_dedup}"
    )

    # Stable golden-set IDs.
    combined = combined.reset_index(drop=True)
    combined.insert(
        0,
        "golden_id",
        [
            f"G{i:03d}"
            for i in range(1, len(combined) + 1)
        ]
    )

    # Put the most useful columns first.
    preferred_columns = [
        "golden_id",
        "customer_tweet_id",
        "customer_text",
        "golden_intent",
        "confidence",
        "case_type",
        "escalation_expected",
        "annotation_notes",
        "brand_response",
    ]

    remaining_columns = [
        c for c in combined.columns
        if c not in preferred_columns
    ]

    combined = combined[
        [
            c for c in preferred_columns
            if c in combined.columns
        ]
        + remaining_columns
    ]

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nFinal golden set rows: {len(combined)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n=== FINAL INTENT DISTRIBUTION ===")
    print(
        combined["golden_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== FINAL CASE TYPE DISTRIBUTION ===")
    print(
        combined["case_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== FINAL ESCALATION DISTRIBUTION ===")
    print(
        combined["escalation_expected"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()