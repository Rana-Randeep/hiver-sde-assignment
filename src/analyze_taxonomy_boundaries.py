from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "taxonomy_boundary_unique_messages.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "taxonomy_boundary_matrix.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("=" * 80)
print("TAXONOMY BOUNDARY MATRIX")
print("=" * 80)

print(f"\nInput rows: {len(df)}")


# ---------------------------------------------------------
# Split review buckets
# ---------------------------------------------------------

def split_buckets(value):
    if pd.isna(value) or not str(value).strip():
        return []

    return [
        bucket.strip()
        for bucket in str(value).split("|")
        if bucket.strip()
    ]


df["bucket_list"] = df["review_bucket"].apply(split_buckets)


# ---------------------------------------------------------
# Candidate bucket counts
# ---------------------------------------------------------

candidate_buckets = sorted(
    {
        bucket
        for buckets in df["bucket_list"]
        for bucket in buckets
    }
)


print("\nCandidate buckets:")
for bucket in candidate_buckets:
    print(f"  - {bucket}")


print("\n" + "=" * 80)
print("BUCKET FREQUENCY")
print("=" * 80)

bucket_counts = {
    bucket: int(
        df["bucket_list"]
        .apply(lambda buckets: bucket in buckets)
        .sum()
    )
    for bucket in candidate_buckets
}

for bucket, count in bucket_counts.items():
    print(f"{bucket:35s} : {count}")


# ---------------------------------------------------------
# Boundary pair counts
# ---------------------------------------------------------

boundary_pairs = [
    (
        "account_general_candidate",
        "data_plan_candidate",
    ),
    (
        "account_general_candidate",
        "voice_line_candidate",
    ),
    (
        "account_general_candidate",
        "sim_device_candidate",
    ),
    (
        "data_plan_candidate",
        "voice_line_candidate",
    ),
    (
        "data_plan_candidate",
        "sim_device_candidate",
    ),
    (
        "voice_line_candidate",
        "sim_device_candidate",
    ),
]


print("\n" + "=" * 80)
print("BOUNDARY PAIR FREQUENCY")
print("=" * 80)

matrix_rows = []

for bucket_a, bucket_b in boundary_pairs:

    mask = df["bucket_list"].apply(
        lambda buckets:
        bucket_a in buckets and bucket_b in buckets
    )

    count = int(mask.sum())

    print(
        f"{bucket_a} <-> {bucket_b}: {count}"
    )

    matrix_rows.append(
        {
            "boundary_a": bucket_a,
            "boundary_b": bucket_b,
            "overlap_count": count,
        }
    )


# ---------------------------------------------------------
# Multi-boundary cases
# ---------------------------------------------------------

df["bucket_count"] = df["bucket_list"].apply(len)

multi_boundary = df[df["bucket_count"] >= 3]


print("\n" + "=" * 80)
print("MULTI-BOUNDARY CASES")
print("=" * 80)

print(
    f"\nMessages belonging to 3+ candidate buckets: "
    f"{len(multi_boundary)}"
)


for _, row in multi_boundary.iterrows():

    print("\n" + "-" * 80)

    print(
        f"customer_tweet_id: "
        f"{row['customer_tweet_id']}"
    )

    print(
        f"review_bucket: "
        f"{row['review_bucket']}"
    )

    print(
        f"customer_text: "
        f"{row['customer_text']}"
    )


# ---------------------------------------------------------
# Save pair matrix
# ---------------------------------------------------------

matrix_df = pd.DataFrame(matrix_rows)

matrix_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(f"\nSaved boundary matrix to:")
print(OUTPUT_FILE)

print("\nEND")
print("=" * 80)