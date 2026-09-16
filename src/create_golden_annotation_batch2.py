from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_review_sheet.csv")
BATCH1_FILE = Path("results/golden_batch_1_annotated.csv")
OUTPUT_FILE = Path("results/golden_batch_2.csv")

BATCH_SIZE = 50


def main():
    df = pd.read_csv(INPUT_FILE)
    batch1 = pd.read_csv(BATCH1_FILE)

    # Exclude everything already reviewed in Batch 1.
    reviewed_ids = set(
        batch1["customer_tweet_id"].astype(str)
    )

    df["customer_tweet_id"] = (
        df["customer_tweet_id"].astype(str)
    )

    remaining = df[
        ~df["customer_tweet_id"].isin(reviewed_ids)
    ].copy()

    # We want Batch 2 to improve coverage of the
    # underrepresented taxonomy areas.
    #
    # These are candidate signals only, NOT labels.

    priority_groups = [
        (
            "account_or_general_candidate",
            remaining["signal_account_general"]
            & ~remaining["signal_network"]
            & ~remaining["signal_data"]
            & ~remaining["signal_recharge"]
            & ~remaining["signal_sim_device"]
            & ~remaining["signal_vas_charge"]
            & ~remaining["signal_bonus_promo"],
            10,
        ),
        (
            "sim_device_candidate",
            remaining["signal_sim_device"]
            & ~remaining["signal_network"],
            8,
        ),
        (
            "voice_line_candidate",
            remaining["signal_network"]
            & remaining["has_question_mark"],
            8,
        ),
        (
            "data_service_candidate",
            remaining["signal_data"]
            & ~remaining["signal_recharge"]
            & ~remaining["signal_bonus_promo"]
            & ~remaining["signal_vas_charge"],
            10,
        ),
        (
            "data_plan_candidate",
            remaining["signal_data"]
            & (
                remaining["signal_bonus_promo"]
                | remaining["signal_account_general"]
            ),
            8,
        ),
    ]

    selected_indices = set()
    selected_group_counts = {}

    for group_name, mask, target_count in priority_groups:
        candidates = remaining[mask].copy()

        # Prefer candidates with a question or longer description,
        # because they tend to provide more context for annotation.
        candidates = candidates.sort_values(
            [
                "has_question_mark",
                "text_length",
                "review_priority",
            ],
            ascending=[False, False, False],
        )

        count = 0

        for idx in candidates.index:
            if idx in selected_indices:
                continue

            selected_indices.add(idx)
            count += 1

            if count >= target_count:
                break

        selected_group_counts[group_name] = count

    # Fill remaining slots with diverse unreviewed examples.
    if len(selected_indices) < BATCH_SIZE:
        fallback = (
            remaining[
                ~remaining.index.isin(selected_indices)
            ]
            .sort_values(
                ["review_priority", "topic_signal_count"],
                ascending=[False, False],
            )
        )

        for idx in fallback.index:
            selected_indices.add(idx)

            if len(selected_indices) >= BATCH_SIZE:
                break

    batch = remaining.loc[
        list(selected_indices)
    ].copy()

    batch = batch.sort_values(
        [
            "signal_account_general",
            "signal_sim_device",
            "signal_data",
            "signal_network",
            "review_priority",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    batch.insert(
        0,
        "annotation_status",
        "",
    )

    batch.insert(
        1,
        "review_order",
        range(1, len(batch) + 1),
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT_FILE, index=False)

    print("Golden annotation Batch 2 created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Batch 1 file: {BATCH1_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Remaining unreviewed candidates: {len(remaining)}")
    print(f"Batch size: {len(batch)}")
    print()

    print("Target candidate groups:")
    for group_name, count in selected_group_counts.items():
        print(f"  {group_name}: {count}")

    print()

    print("Observable signal counts in Batch 2:")

    signals = {
        "network": "signal_network",
        "data": "signal_data",
        "recharge": "signal_recharge",
        "sim/device": "signal_sim_device",
        "VAS/charge": "signal_vas_charge",
        "bonus/promotion": "signal_bonus_promo",
        "account/general": "signal_account_general",
    }

    for name, column in signals.items():
        count = int(batch[column].sum())
        print(f"  {name:20s}: {count}")

    print()

    print("First 10 candidates:")
    print(
        batch[
            [
                "review_order",
                "customer_tweet_id",
                "customer_text",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()