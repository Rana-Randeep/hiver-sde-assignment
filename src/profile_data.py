import pandas as pd


def main():
    file_path = "data/glocare_pairs.csv"

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("BASIC INFORMATION")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate full rows:")
    print(df.duplicated().sum())

    print("\nDuplicate customer tweets:")
    print(df["customer_tweet_id"].duplicated().sum())

    print("\nDuplicate customer texts:")
    print(df["customer_text"].duplicated().sum())


    print("\n" + "=" * 60)
    print("CUSTOMER MESSAGE LENGTH")
    print("=" * 60)

    df["customer_text_length"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(df["customer_text_length"].describe())


    print("\n" + "=" * 60)
    print("SHORTEST CUSTOMER MESSAGES")
    print("=" * 60)

    shortest = df.sort_values("customer_text_length")[
        ["customer_text", "brand_response"]
    ].head(20)

    for _, row in shortest.iterrows():
        print("\nCustomer:", repr(row["customer_text"]))
        print("Brand   :", repr(row["brand_response"]))


    print("\n" + "=" * 60)
    print("RANDOM SAMPLE")
    print("=" * 60)

    sample = df.sample(10, random_state=42)[
        ["customer_text", "brand_response"]
    ]

    for _, row in sample.iterrows():
        print("\nCustomer:", row["customer_text"])
        print("Brand   :", row["brand_response"])


if __name__ == "__main__":
    main()