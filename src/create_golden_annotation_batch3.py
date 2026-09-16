from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_review_sheet.csv")
WORKING_GOLDEN_FILE = Path("results/golden_set_working.csv")
OUTPUT_FILE = Path("results/golden_batch_3.csv")


TARGETS = {
    "account_or_general_candidate": 15,
    "sim_device_candidate": 15,
    "voice_line_candidate": 15,
    "data_service_candidate": 15,
    "data_plan_candidate": 10,
}


def main():
    candidates = pd.read_csv(INPUT_FILE)
    golden = pd.read_csv(WORKING_GOLDEN_FILE)

    # Remove examples already present in the working golden set.
    used_ids = set(golden["customer_tweet_id"].astype(str))

    candidates = candidates[
        ~candidates["customer_tweet_id"].astype(str).isin(used_ids)
    ].copy()

    # Start with no candidate group.
    candidates["candidate_group"] = "other"

    # ---------------------------------------------------------
    # 1. Account / general support
    # ---------------------------------------------------------
    candidates.loc[
        candidates["signal_account_general"] == 1,
        "candidate_group"
    ] = "account_or_general_candidate"

    # ---------------------------------------------------------
    # 2. SIM / device
    # ---------------------------------------------------------
    candidates.loc[
        candidates["signal_sim_device"] == 1,
        "candidate_group"
    ] = "sim_device_candidate"

    # ---------------------------------------------------------
    # 3. Voice / line
    #
    # There is no signal_voice_line column in the dataset.
    # Therefore we use explicit voice/call/line keywords only
    # as a sampling heuristic.
    # ---------------------------------------------------------
    voice_pattern = (
        r"\b(call|calls|calling|called|voice|line|dial|dialing|"
        r"ring|rings|sms|message|messages|cannot call|can't call|"
        r"call drop|call drops|call dropped|call.*completed)\b"
    )

    voice_mask = (
        candidates["customer_text"]
        .fillna("")
        .str.lower()
        .str.contains(voice_pattern, regex=True)
    )

    candidates.loc[
        voice_mask,
        "candidate_group"
    ] = "voice_line_candidate"

    # ---------------------------------------------------------
    # 4. Data candidates
    # ---------------------------------------------------------
    data_mask = candidates["signal_data"] == 1

    # Data-plan wording
    plan_pattern = (
        r"\b(plan|plans|bundle|bundles|subscribe|subscription|"
        r"subscribed|renew|renewal|package|packages|"
        r"mb|gb|data plan|auto.?renew|carry over|expiry|expired)\b"
    )

    plan_mask = (
        candidates["customer_text"]
        .fillna("")
        .str.lower()
        .str.contains(plan_pattern, regex=True)
    )

    # Data-plan candidates get their own group.
    candidates.loc[
        data_mask & plan_mask,
        "candidate_group"
    ] = "data_plan_candidate"

    # Remaining data candidates are data-service candidates.
    candidates.loc[
        data_mask & ~plan_mask,
        "candidate_group"
    ] = "data_service_candidate"

    # ---------------------------------------------------------
    # Prevent accidental overwriting by giving each group
    # an explicit priority.
    # ---------------------------------------------------------
    priority = {
        "account_or_general_candidate": 1,
        "sim_device_candidate": 2,
        "voice_line_candidate": 3,
        "data_service_candidate": 4,
        "data_plan_candidate": 5,
        "other": 6,
    }

    candidates["group_priority"] = (
        candidates["candidate_group"]
        .map(priority)
        .fillna(99)
    )

    # Highest review priority first, then stronger topic signal.
    candidates = candidates.sort_values(
        ["group_priority", "review_priority", "topic_signal_count"],
        ascending=[True, False, False]
    )

    selected_parts = []

    for group, target_count in TARGETS.items():
        group_df = candidates[
            candidates["candidate_group"] == group
        ].head(target_count).copy()

        selected_parts.append(group_df)

    batch = pd.concat(
        selected_parts,
        ignore_index=True
    )

    # Remove duplicate customer tweet IDs if any overlap occurs.
    batch = batch.drop_duplicates(
        subset=["customer_tweet_id"],
        keep="first"
    ).reset_index(drop=True)

    batch.insert(
        0,
        "review_order",
        range(1, len(batch) + 1)
    )

    batch.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("Golden Batch 3 candidate file created.")
    print(f"Input candidate pool: {INPUT_FILE}")
    print(f"Existing golden rows: {len(golden)}")
    print(f"Remaining candidates: {len(candidates)}")
    print(f"Batch 3 size: {len(batch)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nCandidate group distribution:")
    print(
        batch["candidate_group"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()