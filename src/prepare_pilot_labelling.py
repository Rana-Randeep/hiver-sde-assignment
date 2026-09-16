import pandas as pd


def main():
    source_path = "data/glocare_pairs.csv"
    pilot_path = "results/pilot_sample.csv"
    output_path = "results/pilot_labelling.csv"

    # ---------------------------------------------------------
    # 1. Load original dataset and pilot sample
    # ---------------------------------------------------------

    df = pd.read_csv(source_path)
    pilot = pd.read_csv(pilot_path)

    # ---------------------------------------------------------
    # 2. Collect ALL brand responses for each customer tweet
    # ---------------------------------------------------------

    responses = (
        df.sort_values(
            ["customer_tweet_id", "brand_created_at"]
        )
        .groupby("customer_tweet_id")["brand_response"]
        .apply(
            lambda x: "\n\n--- NEXT BRAND RESPONSE ---\n\n"
            .join(x.astype(str))
        )
        .reset_index(name="all_brand_responses")
    )

    # ---------------------------------------------------------
    # 3. Join complete response context to pilot sample
    # ---------------------------------------------------------

    worksheet = pilot.merge(
        responses,
        on="customer_tweet_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 4. Keep the columns useful for manual labelling
    # ---------------------------------------------------------

    worksheet = worksheet[
        [
            "pilot_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "response_count",
            "all_brand_responses",
            "intent_label",
            "label_notes",
        ]
    ]

    # ---------------------------------------------------------
    # 5. Ensure label columns are empty
    # ---------------------------------------------------------

    worksheet["intent_label"] = ""
    worksheet["label_notes"] = ""

    # ---------------------------------------------------------
    # 6. Save labelling worksheet
    # ---------------------------------------------------------

    worksheet.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Print verification information
    # ---------------------------------------------------------

    print("Pilot labelling worksheet created.")
    print("Rows:", len(worksheet))
    print("Output:", output_path)

    print("\nColumns:")
    print(worksheet.columns.tolist())

    print("\nRows with multiple brand responses:")
    print(
        (worksheet["response_count"] > 1).sum()
    )

    print("\nFirst 5 examples:")
    print(
        worksheet[
            [
                "pilot_id",
                "customer_tweet_id",
                "customer_text",
                "response_count"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()