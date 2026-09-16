import pandas as pd


def main():
    file_path = "data/glocare_pairs.csv"

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # 1. How many unique customer tweet IDs?
    # ---------------------------------------------------------

    unique_customer_ids = df["customer_tweet_id"].nunique()

    print("\nUnique customer tweet IDs:", unique_customer_ids)

    # ---------------------------------------------------------
    # 2. How many unique brand tweet IDs?
    # ---------------------------------------------------------

    unique_brand_ids = df["brand_tweet_id"].nunique()

    print("Unique brand tweet IDs:", unique_brand_ids)

    # ---------------------------------------------------------
    # 3. Can one brand tweet ID appear multiple times?
    # ---------------------------------------------------------

    brand_counts = df["brand_tweet_id"].value_counts()

    repeated_brand_ids = brand_counts[brand_counts > 1]

    print(
        "\nBrand tweet IDs appearing more than once:",
        len(repeated_brand_ids)
    )

    if len(repeated_brand_ids) > 0:
        print("\nRepeated brand tweet IDs:")
        print(repeated_brand_ids.head(20))

    # ---------------------------------------------------------
    # 4. Can one customer tweet ID map to multiple brand tweets?
    # ---------------------------------------------------------

    customer_response_counts = (
        df.groupby("customer_tweet_id")["brand_tweet_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(
        "\nMaximum unique brand tweets for one customer tweet:",
        customer_response_counts.max()
    )

    print("\nDistribution of brand responses per customer tweet:")
    print(
        customer_response_counts.value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 5. Can one customer author have many different tweets?
    # ---------------------------------------------------------

    author_customer_counts = (
        df.groupby("customer_author_id")["customer_tweet_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(
        "\nUnique customer tweets per customer author:"
    )
    print(
        author_customer_counts.describe()
    )

    # ---------------------------------------------------------
    # 6. Show authors with many customer tweets
    # ---------------------------------------------------------

    print("\nTop 10 customer authors by number of tweets:")

    top_authors = author_customer_counts.head(10)

    print(top_authors)

    # ---------------------------------------------------------
    # 7. Inspect whether same customer text appears under
    #    multiple customer tweet IDs
    # ---------------------------------------------------------

    text_id_counts = (
        df.groupby("customer_text")["customer_tweet_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    repeated_text_across_ids = text_id_counts[
        text_id_counts > 1
    ]

    print(
        "\nCustomer texts appearing under multiple tweet IDs:",
        len(repeated_text_across_ids)
    )

    if len(repeated_text_across_ids) > 0:
        print(
            "\nTop repeated customer texts across different IDs:"
        )

        for text, count in repeated_text_across_ids.head(10).items():
            print("\nCount:", count)
            print("Text:", text)

    # ---------------------------------------------------------
    # 8. Show a few repeated customer IDs with all brand IDs
    # ---------------------------------------------------------

    repeated_customer_ids = (
        df["customer_tweet_id"]
        .value_counts()
    )

    repeated_customer_ids = repeated_customer_ids[
        repeated_customer_ids > 1
    ].head(10).index

    print(
        "\nExamples of customer tweet IDs with multiple "
        "brand responses:"
    )

    for customer_id in repeated_customer_ids:

        group = df[
            df["customer_tweet_id"] == customer_id
        ]

        print("\nCustomer tweet ID:", customer_id)
        print("Customer text:", group.iloc[0]["customer_text"])

        print(
            "Brand tweet IDs:",
            group["brand_tweet_id"].tolist()
        )


if __name__ == "__main__":
    main()