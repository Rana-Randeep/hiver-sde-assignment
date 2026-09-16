import pandas as pd


INPUT_PATH = "data/glocare_pairs.csv"


def show_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape:", df.shape)

    # ---------------------------------------------------------
    # 1. Normal support examples
    # ---------------------------------------------------------
    show_section("1. RANDOM CUSTOMER -> GLOCARE RESPONSE PAIRS")

    normal_examples = df.sample(
        n=10,
        random_state=42
    )

    for i, (_, row) in enumerate(normal_examples.iterrows(), start=1):
        print(f"\n--- Example {i} ---")
        print("Customer:")
        print(row["customer_text"])
        print("\nGloCare:")
        print(row["brand_response"])

    # ---------------------------------------------------------
    # 2. Very short customer messages
    # ---------------------------------------------------------
    show_section("2. VERY SHORT CUSTOMER MESSAGES")

    short_df = df.copy()

    short_df["message_length"] = (
        short_df["customer_text"]
        .astype(str)
        .str.len()
    )

    short_examples = (
        short_df
        .sort_values("message_length")
        .drop_duplicates(subset=["customer_text"])
        .head(20)
    )

    for i, (_, row) in enumerate(short_examples.iterrows(), start=1):
        print(f"\n--- Short Example {i} ---")
        print("Length:", row["message_length"])
        print("Customer:")
        print(row["customer_text"])
        print("\nGloCare:")
        print(row["brand_response"])

    # ---------------------------------------------------------
    # 3. Duplicate customer texts
    # ---------------------------------------------------------
    show_section("3. DUPLICATE CUSTOMER TEXTS")

    duplicate_text_counts = (
        df["customer_text"]
        .value_counts()
    )

    duplicate_texts = duplicate_text_counts[
        duplicate_text_counts > 1
    ].head(15)

    for i, (text, count) in enumerate(
        duplicate_texts.items(),
        start=1
    ):
        print(f"\n--- Duplicate Text {i} ---")
        print("Occurrences:", count)
        print("Customer:")
        print(text)

        matching_rows = df[
            df["customer_text"] == text
        ]

        for j, (_, row) in enumerate(
            matching_rows.iterrows(),
            start=1
        ):
            print(f"\nResponse {j}:")
            print(row["brand_response"])

    # ---------------------------------------------------------
    # 4. Customer tweet IDs with multiple responses
    # ---------------------------------------------------------
    show_section(
        "4. CUSTOMER TWEET IDs WITH MULTIPLE GLOCARE RESPONSES"
    )

    tweet_id_counts = (
        df["customer_tweet_id"]
        .value_counts()
    )

    repeated_ids = tweet_id_counts[
        tweet_id_counts > 1
    ].head(15)

    for i, (tweet_id, count) in enumerate(
        repeated_ids.items(),
        start=1
    ):
        print(f"\n--- Conversation {i} ---")
        print("Customer tweet ID:", tweet_id)
        print("Number of GloCare responses:", count)

        matching_rows = df[
            df["customer_tweet_id"] == tweet_id
        ].sort_values("brand_created_at")

        customer_text = matching_rows.iloc[0]["customer_text"]

        print("\nCustomer:")
        print(customer_text)

        for j, (_, row) in enumerate(
            matching_rows.iterrows(),
            start=1
        ):
            print(f"\nGloCare response {j}:")
            print(row["brand_response"])

            print(
                "Response time:",
                row["brand_created_at"]
            )

    # ---------------------------------------------------------
    # 5. Basic message-length summary
    # ---------------------------------------------------------
    show_section("5. MESSAGE LENGTH SUMMARY")

    print(
        short_df["message_length"].describe()
    )


if __name__ == "__main__":
    main()