from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "taxonomy_boundary_unique_messages.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_data_plan_sim_boundary.csv"
)


df = pd.read_csv(INPUT_FILE)


def split_buckets(value):
    if pd.isna(value):
        return []

    return [
        bucket.strip()
        for bucket in str(value).split("|")
        if bucket.strip()
    ]


df["bucket_list"] = df["review_bucket"].apply(split_buckets)


mask = df["bucket_list"].apply(
    lambda buckets:
    "data_plan_candidate" in buckets
    and
    "sim_device_candidate" in buckets
)


boundary_df = df.loc[
    mask,
    [
        "customer_tweet_id",
        "customer_text",
        "brand_responses",
        "review_bucket",
        "response_count",
    ]
].copy()


boundary_df["final_candidate_intent"] = ""
boundary_df["reason"] = ""


boundary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 80)
print("DATA_PLAN <-> SIM_DEVICE BOUNDARY REVIEW")
print("=" * 80)

print(f"\nBoundary cases found: {len(boundary_df)}")


for i, (_, row) in enumerate(
    boundary_df.iterrows(),
    start=1
):

    print("\n" + "-" * 80)
    print(f"CASE {i}")

    print(
        f"customer_tweet_id: "
        f"{row['customer_tweet_id']}"
    )

    print(
        f"review_bucket: "
        f"{row['review_bucket']}"
    )

    print(
        f"\nCustomer:\n"
        f"{row['customer_text']}"
    )

    print(
        f"\nHistorical response(s):\n"
        f"{row['brand_responses']}"
    )


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("\nSaved review file:")
print(OUTPUT_FILE)

print("\nEND")
print("=" * 80)