from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = (
    PROJECT_ROOT
    / "results"
    / "golden_set_working.csv"
)

BATCH4_FILE = (
    PROJECT_ROOT
    / "results"
    / "batch4_review_candidates_labeled.csv"
)

REVIEW_SHEET_FILE = (
    PROJECT_ROOT
    / "results"
    / "golden_review_sheet.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "golden_set_working_v2.csv"
)


# ============================================================
# EXPECTED TAXONOMY
# ============================================================

EXPECTED_INTENTS = {
    "network_issue",
    "data_issue",
    "data_plan_issue",
    "unauthorized_charge_or_vas",
    "recharge_or_airtime_issue",
    "sim_or_device_issue",
    "voice_or_line_issue",
    "bonus_or_promotion_issue",
    "account_or_general_support",
    "non_actionable_or_context_required",
}

EXPECTED_CASE_TYPES = {
    "ambiguous",
    "escalation_relevant",
    "low_information",
    "rare",
    "routine",
}

ALLOWED_ESCALATION = {
    "yes",
    "no",
}


# ============================================================
# LOAD
# ============================================================

def main():

    print("=" * 70)
    print("MERGE GOLDEN SET + BATCH 4")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    for file_path, description in [
        (GOLDEN_FILE, "existing golden set"),
        (BATCH4_FILE, "Batch 4 labeled file"),
        (REVIEW_SHEET_FILE, "golden review sheet"),
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {description}:\n{file_path}"
            )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    golden = pd.read_csv(GOLDEN_FILE)
    batch4 = pd.read_csv(BATCH4_FILE)
    review_sheet = pd.read_csv(REVIEW_SHEET_FILE)

    print(f"\nExisting golden rows : {len(golden)}")
    print(f"Batch 4 rows         : {len(batch4)}")
    print(f"Review sheet rows    : {len(review_sheet)}")


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if len(golden) != 137:
        raise ValueError(
            f"Expected current golden set to contain 137 rows, "
            f"found {len(golden)}."
        )

    if len(batch4) != 19:
        raise ValueError(
            f"Expected Batch 4 to contain exactly 19 rows, "
            f"found {len(batch4)}."
        )


    # ========================================================
    # REQUIRED COLUMNS - EXISTING GOLDEN
    # ========================================================

    golden_required_columns = [
        "customer_tweet_id",
        "customer_text",
        "golden_intent",
        "case_type",
        "escalation_expected",
    ]

    for column in golden_required_columns:

        if column not in golden.columns:
            raise ValueError(
                f"Missing required column in golden set: {column}"
            )


    # ========================================================
    # REQUIRED COLUMNS - BATCH 4 LABELS
    # ========================================================

    batch4_required_columns = [
        "customer_tweet_id",
        "golden_intent",
        "case_type",
        "escalation_expected",
    ]

    for column in batch4_required_columns:

        if column not in batch4.columns:
            raise ValueError(
                f"Missing required column in Batch 4: {column}"
            )


    # ========================================================
    # REQUIRED COLUMNS - REVIEW SHEET
    # ========================================================

    review_required_columns = [
        "customer_tweet_id",
        "customer_text",
    ]

    for column in review_required_columns:

        if column not in review_sheet.columns:
            raise ValueError(
                f"Missing required column in review sheet: {column}"
            )


    # ========================================================
    # NORMALIZE JOIN KEY
    # ========================================================

    golden["customer_tweet_id"] = (
        golden["customer_tweet_id"].astype(str).str.strip()
    )

    batch4["customer_tweet_id"] = (
        batch4["customer_tweet_id"].astype(str).str.strip()
    )

    review_sheet["customer_tweet_id"] = (
        review_sheet["customer_tweet_id"].astype(str).str.strip()
    )


    # ========================================================
    # BATCH 4 DUPLICATE CHECK
    # ========================================================

    duplicate_batch4 = batch4[
        batch4["customer_tweet_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicate_batch4) > 0:

        print("\nFAIL - Duplicate IDs inside Batch 4:")
        print(
            duplicate_batch4[
                [
                    "customer_tweet_id",
                    "golden_intent",
                    "case_type",
                    "escalation_expected",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            "Batch 4 contains duplicate customer_tweet_id values."
        )

    print("\nPASS - No duplicate IDs inside Batch 4.")


    # ========================================================
    # REVIEW SHEET DUPLICATE CHECK
    # ========================================================

    duplicate_review = review_sheet[
        review_sheet["customer_tweet_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicate_review) > 0:

        print("\nFAIL - Duplicate IDs inside review sheet:")
        print(
            duplicate_review[
                ["customer_tweet_id", "customer_text"]
            ].to_string(index=False)
        )

        raise ValueError(
            "Review sheet contains duplicate customer_tweet_id "
            "values. Cannot perform a deterministic join."
        )

    print("PASS - Review sheet has unique customer IDs.")


    # ========================================================
    # JOIN BATCH 4 LABELS WITH CUSTOMER TEXT
    # ========================================================

    batch4_ids = set(batch4["customer_tweet_id"])
    review_ids = set(review_sheet["customer_tweet_id"])

    missing_review_ids = batch4_ids - review_ids

    if missing_review_ids:

        print("\nFAIL - Batch 4 IDs missing from review sheet:")

        for tweet_id in sorted(missing_review_ids):
            print(f"  {tweet_id}")

        raise ValueError(
            f"{len(missing_review_ids)} Batch 4 customer_tweet_id "
            "values could not be found in golden_review_sheet.csv."
        )

    print(
        "PASS - All Batch 4 IDs found in golden review sheet."
    )

    batch4 = batch4.merge(
        review_sheet[
            [
                "customer_tweet_id",
                "customer_text",
            ]
        ],
        on="customer_tweet_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"PASS - Batch 4 customer_text recovered for "
        f"{len(batch4)} rows."
    )


    # ========================================================
    # CUSTOMER TEXT CHECK
    # ========================================================

    missing_text = batch4["customer_text"].isna().sum()

    if missing_text > 0:

        print("\nFAIL - Missing customer_text after join:")
        print(
            batch4[
                batch4["customer_text"].isna()
            ][["customer_tweet_id"]].to_string(index=False)
        )

        raise ValueError(
            f"Batch 4 has {missing_text} rows with missing "
            "customer_text after the join."
        )

    print(
        "PASS - Batch 4 customer_text contains no missing values."
    )


    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    golden_ids = set(
        golden["customer_tweet_id"]
    )

    batch4_ids = set(
        batch4["customer_tweet_id"]
    )

    overlap = golden_ids.intersection(batch4_ids)

    if overlap:

        print("\nFAIL - Batch 4 overlaps with existing golden set:")

        for tweet_id in sorted(overlap):
            print(f"  {tweet_id}")

        raise ValueError(
            "Batch 4 contains customer_tweet_id values "
            "already present in the golden set."
        )

    print("PASS - No overlap with existing golden set.")


    # ========================================================
    # MISSING LABEL CHECK
    # ========================================================

    for column in [
        "golden_intent",
        "case_type",
        "escalation_expected",
    ]:

        missing = batch4[column].isna().sum()

        if missing > 0:
            raise ValueError(
                f"Batch 4 has {missing} missing values in {column}."
            )

        print(
            f"PASS - Batch 4 {column}: no missing values."
        )


    # ========================================================
    # INTENT CHECK
    # ========================================================

    batch4_intents = set(
        batch4["golden_intent"].astype(str)
    )

    unknown_intents = (
        batch4_intents - EXPECTED_INTENTS
    )

    if unknown_intents:

        print("\nFAIL - Unknown Batch 4 intents:")

        for intent in sorted(unknown_intents):
            print(f"  {intent}")

        raise ValueError(
            "Batch 4 contains intent values outside "
            "the approved taxonomy."
        )

    print(
        "PASS - Batch 4 intents match approved taxonomy."
    )


    # ========================================================
    # CASE TYPE CHECK
    # ========================================================

    batch4_case_types = set(
        batch4["case_type"].astype(str)
    )

    unknown_case_types = (
        batch4_case_types - EXPECTED_CASE_TYPES
    )

    if unknown_case_types:

        print("\nFAIL - Unknown Batch 4 case types:")

        for case_type in sorted(unknown_case_types):
            print(f"  {case_type}")

        raise ValueError(
            "Batch 4 contains unknown case_type values."
        )

    print(
        "PASS - Batch 4 case types are valid."
    )


    # ========================================================
    # ESCALATION CHECK
    # ========================================================

    batch4_escalation = set(
        batch4["escalation_expected"]
        .astype(str)
        .str.lower()
    )

    unknown_escalation = (
        batch4_escalation - ALLOWED_ESCALATION
    )

    if unknown_escalation:

        print("\nFAIL - Unknown escalation values:")

        for value in sorted(unknown_escalation):
            print(f"  {value}")

        raise ValueError(
            "Batch 4 contains invalid escalation values."
        )

    print(
        "PASS - Batch 4 escalation values are valid."
    )


    # ========================================================
    # ALIGN COLUMNS
    # ========================================================

    # Preserve the existing golden-set schema.
    # Batch 4 sampling/review metadata is intentionally excluded
    # if it is not part of the existing golden-set schema.

    common_columns = [
        column
        for column in golden.columns
        if column in batch4.columns
    ]

    combined = pd.concat(
        [
            golden[common_columns],
            batch4[common_columns],
        ],
        ignore_index=True,
    )


    # ========================================================
    # FINAL DUPLICATE CHECK
    # ========================================================

    duplicate_final = combined[
        combined["customer_tweet_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicate_final) > 0:

        print("\nFAIL - Duplicate IDs after merge:")

        print(
            duplicate_final[
                ["customer_tweet_id", "customer_text"]
            ].to_string(index=False)
        )

        raise ValueError(
            "Final merged golden set contains duplicate IDs."
        )

    print(
        "PASS - Final merged set has unique customer IDs."
    )


    # ========================================================
    # REBUILD STABLE GOLDEN IDs
    # ========================================================

    if "golden_id" in combined.columns:
        combined = combined.drop(columns=["golden_id"])

    combined.insert(
        0,
        "golden_id",
        [
            f"G{i:03d}"
            for i in range(1, len(combined) + 1)
        ],
    )


    # ========================================================
    # FINAL SIZE CHECK
    # ========================================================

    expected_size = 137 + 19

    if len(combined) != expected_size:
        raise ValueError(
            f"Expected final golden set size {expected_size}, "
            f"found {len(combined)}."
        )


    # ========================================================
    # SAVE
    # ========================================================

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MERGE SUCCESSFUL")
    print("=" * 70)

    print("\nBatches 1-3 rows : 137")
    print("Batch 4 rows     : 19")
    print(f"Final rows       : {len(combined)}")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nGolden ID range:")
    print(
        f"{combined['golden_id'].iloc[0]} "
        f"-> "
        f"{combined['golden_id'].iloc[-1]}"
    )

    print("\nIntent distribution:")
    print(
        combined["golden_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCase type distribution:")
    print(
        combined["case_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nEscalation expectation:")
    print(
        combined["escalation_expected"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()