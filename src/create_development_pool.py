import pandas as pd


INPUT_PATH = "data/glocare_pairs.csv"
GOLDEN_PATH = "results/golden_set.csv"
OUTPUT_PATH = "results/development_pool.csv"


def main():
    # ---------------------------------------------------------
    # 1. Load working dataset and locked golden set
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    # ---------------------------------------------------------
    # 2. Get the customer-tweet interaction IDs used by
    #    the golden evaluation set
    # ---------------------------------------------------------
    golden_ids = set(
        golden["customer_tweet_id"]
    )

    # ---------------------------------------------------------
    # 3. Remove ALL rows belonging to those interactions
    # ---------------------------------------------------------
    development_df = df[
        ~df["customer_tweet_id"].isin(golden_ids)
    ].copy()

    # ---------------------------------------------------------
    # 4. Safety checks
    # ---------------------------------------------------------
    assert not development_df["customer_tweet_id"].isin(
        golden_ids
    ).any()

    assert len(development_df) == (
        len(df)
        - df["customer_tweet_id"].isin(golden_ids).sum()
    )

    # ---------------------------------------------------------
    # 5. Save development pool
    # ---------------------------------------------------------
    development_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 6. Report
    # ---------------------------------------------------------
    print("=" * 70)
    print("DEVELOPMENT POOL CREATED")
    print("=" * 70)

    print("Original rows:", len(df))
    print("Golden examples:", len(golden))
    print("Unique golden interaction IDs:", len(golden_ids))

    print(
        "Rows locked out by golden interaction IDs:",
        df["customer_tweet_id"].isin(golden_ids).sum()
    )

    print(
        "Development rows:",
        len(development_df)
    )

    print(
        "Development interaction units:",
        development_df["customer_tweet_id"].nunique()
    )

    print(
        "Development pool saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()