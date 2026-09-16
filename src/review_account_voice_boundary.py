import pandas as pd
from pathlib import Path


INPUT_PATH = Path("results/taxonomy_boundary_unique_messages.csv")
OUTPUT_PATH = Path("results/review_account_voice_boundary.csv")


df = pd.read_csv(INPUT_PATH)

TARGET_BUCKETS = {
    "account_general_candidate",
    "voice_line_candidate",
}


# Keep messages that contain BOTH boundary buckets.
review_df = df[
    df["review_bucket"].apply(
        lambda x: TARGET_BUCKETS.issubset(
            {bucket.strip() for bucket in str(x).split("|")}
        )
    )
].copy()


# Keep only the columns useful for manual review.
review_df = review_df[
    [
        "customer_tweet_id",
        "customer_text",
        "brand_responses",
        "review_bucket",
    ]
]


review_df.to_csv(OUTPUT_PATH, index=False)


print("=" * 80)
print("ACCOUNT_GENERAL <-> VOICE_LINE BOUNDARY REVIEW")
print("=" * 80)

print(f"\nCases to review: {len(review_df)}")

print("\n" + "=" * 80)
print("CASES")
print("=" * 80)

for _, row in review_df.iterrows():
    print("\n" + "-" * 80)
    print(f"customer_tweet_id: {row['customer_tweet_id']}")
    print(f"customer_text: {row['customer_text']}")
    print(f"brand_responses: {row['brand_responses']}")
    print(f"review_bucket: {row['review_bucket']}")

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(f"\nSaved review file:")
print(OUTPUT_PATH.resolve())

print("\nEND")
print("=" * 80)