# from pathlib import Path

# import pandas as pd


# # ---------------------------------------------------------
# # Paths
# # ---------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# INPUT_FILE = PROJECT_ROOT / "results" / "taxonomy_boundary_review.csv"


# # ---------------------------------------------------------
# # Load data
# # ---------------------------------------------------------

# df = pd.read_csv(INPUT_FILE)


# print("=" * 80)
# print("TAXONOMY BOUNDARY REVIEW")
# print("=" * 80)

# print(f"\nInput file : {INPUT_FILE}")
# print(f"Rows       : {len(df)}")
# print(f"Columns    : {list(df.columns)}")


# # ---------------------------------------------------------
# # Basic validation
# # ---------------------------------------------------------

# required_columns = [
#     "customer_text"
# ]

# missing_columns = [
#     col for col in required_columns
#     if col not in df.columns
# ]

# if missing_columns:
#     raise ValueError(
#         f"Missing required columns: {missing_columns}"
#     )


# # ---------------------------------------------------------
# # Display available label columns
# # ---------------------------------------------------------

# print("\nAvailable columns:")
# for col in df.columns:
#     print(f"  - {col}")


# # ---------------------------------------------------------
# # Print examples
# # ---------------------------------------------------------

# print("\n" + "=" * 80)
# print("EXAMPLES")
# print("=" * 80)

# for i, row in df.iterrows():

#     print(f"\n[{i + 1}]")

#     for col in df.columns:
#         value = row[col]

#         if pd.isna(value):
#             value = ""

#         print(f"{col}: {value}")

# print("\n" + "=" * 80)
# print("END OF REVIEW")
# print("=" * 80)

#----------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "taxonomy_boundary_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "taxonomy_boundary_unique_messages.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("=" * 80)
print("TAXONOMY BOUNDARY REVIEW - UNIQUE CUSTOMER MESSAGES")
print("=" * 80)

print(f"\nInput file        : {INPUT_FILE}")
print(f"Original rows     : {len(df)}")


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = [
    "customer_tweet_id",
    "customer_text",
    "brand_response",
    "review_bucket",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Identify repeated customer tweet IDs
# ---------------------------------------------------------

tweet_counts = (
    df["customer_tweet_id"]
    .value_counts()
)

repeated_ids = tweet_counts[tweet_counts > 1]


print(f"Unique customer tweet IDs : {df['customer_tweet_id'].nunique()}")
print(f"Repeated tweet IDs        : {len(repeated_ids)}")


if len(repeated_ids) > 0:
    print("\nRepeated customer tweet IDs:")
    print(repeated_ids.to_string())


# ---------------------------------------------------------
# Collapse to one row per customer tweet
# ---------------------------------------------------------
#
# For taxonomy review, customer_text is the unit of analysis.
#
# We keep:
#   - customer_tweet_id
#   - customer_text
#
# We also preserve all review buckets and brand responses
# so that no information is silently lost.
# ---------------------------------------------------------

unique_rows = []

for tweet_id, group in df.groupby(
    "customer_tweet_id",
    sort=False
):

    first_row = group.iloc[0]

    # Preserve unique review buckets
    buckets = sorted(
        set(
            bucket
            for bucket in group["review_bucket"].dropna()
        )
    )

    # Preserve all historical brand responses
    responses = [
        response
        for response in group["brand_response"].dropna()
    ]

    unique_rows.append(
        {
            "customer_tweet_id": tweet_id,
            "customer_text": first_row["customer_text"],
            "review_bucket": " || ".join(buckets),
            "brand_responses": " || ".join(responses),
            "response_count": len(responses),
        }
    )


unique_df = pd.DataFrame(unique_rows)


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

unique_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nOriginal review rows       : {len(df)}")
print(f"Unique customer messages  : {len(unique_df)}")
print(f"Rows removed by collapsing : {len(df) - len(unique_df)}")

print(f"\nOutput file:")
print(OUTPUT_FILE)


# ---------------------------------------------------------
# Show collapsed examples
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("COLLAPSED REPEATED CASES")
print("=" * 80)

repeated_unique = unique_df[
    unique_df["response_count"] > 1
]

if repeated_unique.empty:
    print("\nNo repeated customer messages found.")

else:
    for _, row in repeated_unique.iterrows():

        print("\n" + "-" * 80)

        print(f"customer_tweet_id : {row['customer_tweet_id']}")
        print(f"customer_text     : {row['customer_text']}")
        print(f"review_bucket     : {row['review_bucket']}")
        print(f"response_count    : {row['response_count']}")

        print("\nBrand responses:")

        for i, response in enumerate(
            row["brand_responses"].split(" || "),
            start=1
        ):
            print(f"  Response {i}: {response}")


print("\n" + "=" * 80)
print("END")
print("=" * 80)