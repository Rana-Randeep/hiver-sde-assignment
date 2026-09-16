import pandas as pd
import re


INPUT_PATH = "data/glocare_pairs.csv"
OUTPUT_PATH = "results/taxonomy_boundary_review.csv"


def contains_any(text, keywords):
    """
    Return True if any keyword/phrase appears in the text.
    """
    text = str(text).lower()

    for keyword in keywords:
        if keyword in text:
            return True

    return False


def assign_review_bucket(text):
    """
    This is NOT an intent classifier.

    It only creates broad review buckets so that we can
    manually inspect potentially ambiguous taxonomy areas.
    """

    text = str(text).lower()

    buckets = []

    keyword_groups = {
        "account_general_candidate": [
            "account",
            "number",
            "my number",
            "port",
            "porting",
            "office",
            "shop",
            "customer care",
            "customer service",
            "agent",
            "help me",
            "how do i",
            "how can i",
        ],

        "data_plan_candidate": [
            "data plan",
            "data bundle",
            "bundle",
            "subscribe",
            "subscription",
            "renew",
            "renewal",
            "mb",
            "gb",
            "hotspot",
            "4g lte",
        ],

        "sim_device_candidate": [
            "sim",
            "sim card",
            "sim swap",
            "lost sim",
            "new sim",
            "phone",
            "handset",
            "device",
            "mobile phone",
        ],

        "voice_line_candidate": [
            "call",
            "calling",
            "voice",
            "dial",
            "dialing",
            "line",
            "cannot call",
            "can't call",
            "calls",
        ],
    }

    for bucket, keywords in keyword_groups.items():
        if contains_any(text, keywords):
            buckets.append(bucket)

    if not buckets:
        return "other"

    return "|".join(buckets)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape:", df.shape)
    print()

    # Create a review bucket from customer text only.
    df["review_bucket"] = df["customer_text"].apply(assign_review_bucket)

    print("Review bucket counts:")
    print(df["review_bucket"].value_counts())
    print()

    # Show examples from each bucket.
    bucket_names = [
        "account_general_candidate",
        "data_plan_candidate",
        "sim_device_candidate",
        "voice_line_candidate",
    ]

    all_examples = []

    for bucket in bucket_names:
        subset = df[df["review_bucket"].str.contains(bucket, regex=False)]

        # Maximum 15 examples per bucket.
        sample = subset[
            [
                "customer_tweet_id",
                "customer_text",
                "brand_response",
                "review_bucket",
            ]
        ].head(15)

        all_examples.append(sample)

        print("=" * 80)
        print(bucket)
        print("=" * 80)

        if len(sample) == 0:
            print("No examples found.")
        else:
            for _, row in sample.iterrows():
                print()
                print("ID:", row["customer_tweet_id"])
                print("CUSTOMER:", row["customer_text"])
                print("BRAND:", row["brand_response"])

    review_df = pd.concat(all_examples, ignore_index=True)

    review_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 80)
    print("Saved review file:")
    print(OUTPUT_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()