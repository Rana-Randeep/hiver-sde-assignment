import pandas as pd


INPUT_PATH = "results/pilot_candidates_for_labeling.csv"


# ============================================================
# Load pilot sample
# ============================================================

df = pd.read_csv(INPUT_PATH)


print("=" * 80)
print("HIVER GLOCARE — PILOT LABELLING DATA")
print("=" * 80)

print(f"Total pilot examples: {len(df)}")
print()


# ============================================================
# Display examples
# ============================================================

for _, row in df.iterrows():

    print("-" * 80)

    print(f"Pilot ID       : {row['pilot_id']}")
    print(f"Customer ID    : {row['customer_tweet_id']}")

    print()
    print("CUSTOMER:")
    print(row["customer_text"])

    print()
    print("GLOCARE RESPONSE:")
    print(row["brand_response"])

    print()
    print("SAMPLING INFO:")
    print(
        f"Cluster={row['cluster_id']} | "
        f"Length={row['length_bucket']} | "
        f"MultipleResponses={row['has_multiple_responses']} | "
        f"DuplicateText={row['has_duplicate_text']}"
    )

    print()


print("=" * 80)
print("END OF PILOT DATA")
print("=" * 80)