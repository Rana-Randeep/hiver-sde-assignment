import pandas as pd


def main():
    file_path = "data/glocare_pairs.csv"
    output_path = "results/pilot_sample.csv"

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 1. Work at customer-interaction level
    # ---------------------------------------------------------
    #
    # One customer_tweet_id = one interaction anchor.
    #
    # If an interaction has multiple brand responses,
    # keep it as one pilot example.
    # ---------------------------------------------------------

    interaction_counts = (
        df.groupby("customer_tweet_id")
        .size()
        .rename("response_count")
    )

    interactions = (
        df.groupby("customer_tweet_id", as_index=False)
        .first()
    )

    interactions = interactions.merge(
        interaction_counts,
        on="customer_tweet_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 2. Create message-length buckets
    # ---------------------------------------------------------

    interactions["text_length"] = (
        interactions["customer_text"]
        .str.len()
    )

    interactions["length_bucket"] = pd.cut(
        interactions["text_length"],
        bins=[0, 40, 100, 180, float("inf")],
        labels=[
            "short",
            "medium",
            "long",
            "very_long"
        ],
        include_lowest=True
    )

    # ---------------------------------------------------------
    # 3. Create response-count buckets
    # ---------------------------------------------------------

    interactions["response_bucket"] = interactions[
        "response_count"
    ].map(
        lambda x: (
            "one_response"
            if x == 1
            else "multiple_responses"
        )
    )

    # ---------------------------------------------------------
    # 4. Create sampling strata
    # ---------------------------------------------------------

    interactions["stratum"] = (
        interactions["length_bucket"].astype(str)
        + "_"
        + interactions["response_bucket"]
    )

    # ---------------------------------------------------------
    # 5. Sample proportionally from each stratum
    # ---------------------------------------------------------

    target_size = 100

    # First allocate approximately proportional sample counts.
    stratum_counts = (
        interactions["stratum"]
        .value_counts()
    )

    allocations = (
        stratum_counts / len(interactions) * target_size
    ).round().astype(int)

    # Make sure we don't request more rows than exist.
    allocations = allocations.clip(
        upper=stratum_counts
    )

    # Adjust total to exactly 100.
    difference = target_size - allocations.sum()

    if difference > 0:
        for stratum in stratum_counts.index:
            if difference == 0:
                break

            available = stratum_counts[stratum]
            current = allocations[stratum]

            if current < available:
                allocations[stratum] += 1
                difference -= 1

    elif difference < 0:
        for stratum in reversed(stratum_counts.index):
            if difference == 0:
                break

            if allocations[stratum] > 0:
                allocations[stratum] -= 1
                difference += 1

    # ---------------------------------------------------------
    # 6. Deterministic sampling
    # ---------------------------------------------------------

    sampled_parts = []

    for stratum, n in allocations.items():

        if n == 0:
            continue

        group = interactions[
            interactions["stratum"] == stratum
        ]

        sample = group.sample(
            n=n,
            random_state=42
        )

        sampled_parts.append(sample)

    pilot = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # 7. Shuffle final pilot set
    # ---------------------------------------------------------

    pilot = pilot.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Add a human labelling column.
    pilot.insert(
        0,
        "pilot_id",
        range(1, len(pilot) + 1)
    )

    pilot["intent_label"] = ""

    pilot["label_notes"] = ""

    # ---------------------------------------------------------
    # 8. Select useful columns for manual labelling
    # ---------------------------------------------------------

    pilot = pilot[
        [
            "pilot_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "response_count",
            "brand_response",
            "intent_label",
            "label_notes",
        ]
    ]

    # ---------------------------------------------------------
    # 9. Save
    # ---------------------------------------------------------

    pilot.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 10. Print summary
    # ---------------------------------------------------------

    print("Pilot sample created.")
    print("Rows:", len(pilot))
    print("Output:", output_path)

    print("\nResponse count distribution:")
    print(
        pilot["response_count"]
        .value_counts()
        .sort_index()
    )

    print("\nMessage length summary:")
    print(
        pilot["customer_text"]
        .str.len()
        .describe()
    )

    print("\nFirst 10 pilot examples:")
    print(
        pilot[
            [
                "pilot_id",
                "customer_tweet_id",
                "customer_text",
                "response_count"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()