import pandas as pd


INPUT_PATH = "data/glocare_pairs.csv"


def show_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Make sure timestamps are actual datetime values
    df["customer_created_at"] = pd.to_datetime(
        df["customer_created_at"],
        errors="coerce"
    )

    df["brand_created_at"] = pd.to_datetime(
        df["brand_created_at"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 1. How many responses does each customer tweet have?
    # ---------------------------------------------------------
    response_counts = (
        df.groupby("customer_tweet_id")
        .size()
    )

    show_section("1. RESPONSES PER CUSTOMER TWEET")

    print("Unique customer tweets:", len(response_counts))

    print("\nResponse-count distribution:")
    print(response_counts.value_counts().sort_index())

    print("\nCustomer tweets with multiple responses:")
    print((response_counts > 1).sum())

    # ---------------------------------------------------------
    # 2. Distribution of 2-response vs 3-response interactions
    # ---------------------------------------------------------
    show_section("2. MULTI-RESPONSE INTERACTIONS")

    multi_response_counts = response_counts[
        response_counts > 1
    ]

    print(
        multi_response_counts
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 3. Time gap between first customer tweet and
    #    first/last GloCare response
    # ---------------------------------------------------------
    first_response = (
        df.groupby("customer_tweet_id")["brand_created_at"]
        .min()
    )

    last_response = (
        df.groupby("customer_tweet_id")["brand_created_at"]
        .max()
    )

    customer_time = (
        df.groupby("customer_tweet_id")["customer_created_at"]
        .first()
    )

    first_response_delay = (
        first_response - customer_time
    ).dt.total_seconds() / 60

    total_response_span = (
        last_response - first_response
    ).dt.total_seconds() / 60

    show_section("3. RESPONSE TIMING")

    print("First-response delay (minutes):")
    print(first_response_delay.describe())

    print("\nTime span from first to last GloCare response:")
    print(total_response_span.describe())

    # ---------------------------------------------------------
    # 4. Are customer texts identical within same tweet ID?
    # ---------------------------------------------------------
    text_counts_per_tweet = (
        df.groupby("customer_tweet_id")["customer_text"]
        .nunique()
    )

    show_section("4. CUSTOMER TEXT CONSISTENCY")

    print(
        "Customer tweets where all rows have the same text:",
        (text_counts_per_tweet == 1).sum()
    )

    print(
        "Customer tweets where text differs across rows:",
        (text_counts_per_tweet > 1).sum()
    )

    # ---------------------------------------------------------
    # 5. Show ALL multi-response interactions
    #    but only concise metadata
    # ---------------------------------------------------------
    show_section(
        "5. MULTI-RESPONSE INTERACTION SAMPLES"
    )

    multi_ids = (
        response_counts[
            response_counts > 1
        ]
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    for i, tweet_id in enumerate(multi_ids, start=1):

        group = (
            df[df["customer_tweet_id"] == tweet_id]
            .sort_values("brand_created_at")
        )

        print(f"\n--- Interaction {i} ---")
        print("Customer tweet ID:", tweet_id)
        print("Number of responses:", len(group))

        print(
            "Customer:",
            group.iloc[0]["customer_text"]
        )

        print(
            "Response times:"
        )

        for _, row in group.iterrows():
            print(
                " ",
                row["brand_created_at"],
                "|",
                row["brand_response"][:150]
            )

    # ---------------------------------------------------------
    # 6. Check whether repeated customer tweet IDs are
    #    concentrated in particular message lengths
    # ---------------------------------------------------------
    df["message_length"] = (
        df["customer_text"]
        .astype(str)
        .str.len()
    )

    repeated_id_set = set(
        response_counts[
            response_counts > 1
        ].index
    )

    df["has_multiple_responses"] = (
        df["customer_tweet_id"]
        .isin(repeated_id_set)
    )

    show_section(
        "6. MESSAGE LENGTH: SINGLE VS MULTI-RESPONSE"
    )

    print(
        df.groupby("has_multiple_responses")[
            "message_length"
        ].describe()
    )


if __name__ == "__main__":
    main()