import pandas as pd

INPUT_FILE = "results/vector_failure_analysis.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 90)
print("VECTOR RETRIEVAL FAILURE — INTENT INSPECTION")
print("=" * 90)

print(f"\nTotal failures: {len(df)}")

for _, row in df.iterrows():
    print("\n" + "-" * 90)

    print(f"Customer tweet ID : {row['customer_tweet_id']}")
    print(f"Failure mechanism : {row['failure_mechanism']}")
    print(f"Golden intent     : {row['golden_intent']}")

    print("\nGolden customer message:")
    print(row["golden_customer_text"])

    print("\nRetrieved customer message:")
    print(row["vector_top1_customer_text"])

    print("\nRetrieved historical response:")
    print(row["vector_top1_response"])

print("\n" + "=" * 90)
print("END")
print("=" * 90)