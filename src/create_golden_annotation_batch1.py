from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_review_sheet.csv")
OUTPUT_FILE = Path("results/golden_batch_1.csv")

BATCH_SIZE = 50


def main():
    df = pd.read_csv(INPUT_FILE)

    # We use observable review signals only for ordering.
    # They are NOT labels.
    #
    # First prioritize potentially difficult cases,
    # then preserve topic diversity.

    selected_indices = set()

    signal_columns = [
        "signal_network",
        "signal_data",
        "signal_recharge",
        "signal_sim_device",
        "signal_vas_charge",
        "signal_bonus_promo",
        "signal_account_general",
    ]

    # Start with high-priority candidates.
    high_priority = (
        df.sort_values(
            ["review_priority", "topic_signal_count"],
            ascending=[False, False],
        )
        .index
        .tolist()
    )

    # Take high-priority examples while avoiding excessive
    # concentration from one topic.
    topic_counts = {column: 0 for column in signal_columns}

    for idx in high_priority:
        row = df.loc[idx]

        active_topics = [
            column
            for column in signal_columns
            if bool(row[column])
        ]

        # Prefer examples that add topic diversity.
        if active_topics:
            current_max = max(topic_counts[t] for t in active_topics)

            if current_max >= 12 and len(selected_indices) < 35:
                continue

            for topic in active_topics:
                topic_counts[topic] += 1

        selected_indices.add(idx)

        if len(selected_indices) >= 35:
            break

    # Add short / low-information candidates.
    low_info_candidates = (
        df[
            (
                df["signal_low_information"]
                | df["signal_short_message"]
            )
            & (~df.index.isin(selected_indices))
        ]
        .sort_values(
            ["review_priority", "text_length"],
            ascending=[False, True],
        )
        .index
        .tolist()
    )

    for idx in low_info_candidates:
        selected_indices.add(idx)

        if len(selected_indices) >= 43:
            break

    # Add escalation-relevant candidates.
    escalation_candidates = (
        df[
            df["signal_possible_escalation"]
            & (~df.index.isin(selected_indices))
        ]
        .sort_values(
            ["review_priority", "topic_signal_count"],
            ascending=[False, False],
        )
        .index
        .tolist()
    )

    for idx in escalation_candidates:
        selected_indices.add(idx)

        if len(selected_indices) >= BATCH_SIZE:
            break

    # Safety fallback in case the above selection produces fewer rows.
    if len(selected_indices) < BATCH_SIZE:
        remaining = [
            idx for idx in df.index
            if idx not in selected_indices
        ]

        for idx in remaining:
            selected_indices.add(idx)

            if len(selected_indices) >= BATCH_SIZE:
                break

    batch = df.loc[list(selected_indices)].copy()

    # Reorder so the most useful cases appear first.
    batch = batch.sort_values(
        ["review_priority", "topic_signal_count", "text_length"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Add explicit annotation instructions as metadata columns.
    batch.insert(0, "annotation_status", "")
    batch.insert(1, "review_order", range(1, len(batch) + 1))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT_FILE, index=False)

    print("Golden annotation Batch 1 created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Batch size: {len(batch)}")
    print()

    print("Review columns:")
    print("  annotation_status")
    print("  golden_selected")
    print("  golden_intent")
    print("  confidence")
    print("  case_type")
    print("  escalation_expected")
    print("  annotation_notes")
    print()

    print("Batch characteristics:")
    print(
        f"High-priority rows: "
        f"{(batch['review_priority'] >= 2).sum()}"
    )
    print(
        f"Short-message rows: "
        f"{batch['signal_short_message'].sum()}"
    )
    print(
        f"Low-information signal rows: "
        f"{batch['signal_low_information'].sum()}"
    )
    print(
        f"Possible-escalation rows: "
        f"{batch['signal_possible_escalation'].sum()}"
    )
    print()

    print("First 10 rows for review:")
    print(
        batch[
            [
                "review_order",
                "customer_tweet_id",
                "customer_text",
                "review_priority",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()