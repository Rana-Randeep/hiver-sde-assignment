import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "results/development_pool.csv"

TRAIN_OUTPUT_PATH = "results/train.csv"
VALIDATION_OUTPUT_PATH = "results/validation.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    # ---------------------------------------------------------
    # 1. Load development pool
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_PATH)

    # ---------------------------------------------------------
    # 2. Get unique interaction IDs
    # ---------------------------------------------------------
    interaction_ids = df["customer_tweet_id"].drop_duplicates()

    # ---------------------------------------------------------
    # 3. Split interaction IDs, NOT individual rows
    # ---------------------------------------------------------
    train_ids, validation_ids = train_test_split(
        interaction_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_ids = set(train_ids)
    validation_ids = set(validation_ids)

    # ---------------------------------------------------------
    # 4. Reconstruct row-level datasets using interaction IDs
    # ---------------------------------------------------------
    train_df = df[
        df["customer_tweet_id"].isin(train_ids)
    ].copy()

    validation_df = df[
        df["customer_tweet_id"].isin(validation_ids)
    ].copy()

    # ---------------------------------------------------------
    # 5. Critical leakage check
    # ---------------------------------------------------------
    overlap = (
        set(train_df["customer_tweet_id"])
        &
        set(validation_df["customer_tweet_id"])
    )

    assert len(overlap) == 0, (
        "LEAKAGE DETECTED: train and validation "
        "share customer_tweet_id values."
    )

    # ---------------------------------------------------------
    # 6. Check that every development row was assigned
    # ---------------------------------------------------------
    assert len(train_df) + len(validation_df) == len(df)

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------
    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Report
    # ---------------------------------------------------------
    print("=" * 70)
    print("TRAIN / VALIDATION SPLIT CREATED")
    print("=" * 70)

    print("Development rows:", len(df))
    print(
        "Development interaction units:",
        df["customer_tweet_id"].nunique()
    )

    print("\nTrain:")
    print("  Rows:", len(train_df))
    print(
        "  Interaction units:",
        train_df["customer_tweet_id"].nunique()
    )

    print("\nValidation:")
    print("  Rows:", len(validation_df))
    print(
        "  Interaction units:",
        validation_df["customer_tweet_id"].nunique()
    )

    print("\nInteraction-ID overlap:", len(overlap))

    print("\nTrain proportion of interaction units:",
          round(
              train_df["customer_tweet_id"].nunique()
              / df["customer_tweet_id"].nunique(),
              4
          ))

    print("Validation proportion of interaction units:",
          round(
              validation_df["customer_tweet_id"].nunique()
              / df["customer_tweet_id"].nunique(),
              4
          ))

    print("\nRandom state:", RANDOM_STATE)

    print("\nSaved:")
    print(" ", TRAIN_OUTPUT_PATH)
    print(" ", VALIDATION_OUTPUT_PATH)


if __name__ == "__main__":
    main()