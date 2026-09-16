import pandas as pd

INPUT_FILE = "results/vector_retrieval_review.csv"
OUTPUT_FILE = "results/vector_failure_analysis.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only genuine retrieval failures.
failures = df[df["vector_relevance"] == 0].copy()

# Manual failure-mechanism assignment based on the reviewed examples.
failure_map = {
    2267327: "cross_intent_overlap",
    745771: "cross_intent_overlap",
    317832: "fine_grained_intent_distinction",
    2910699: "cross_intent_overlap",
    2218206: "cross_intent_overlap",
    1000324: "operational_mismatch",
    1043403: "operational_mismatch",
    2800841: "cross_intent_overlap",
    658682: "fine_grained_intent_distinction",
    1931834: "fine_grained_intent_distinction",
    49737: "operational_mismatch",
    598439: "context_poor_query",
    1371044: "context_poor_query",
}

failures["failure_mechanism"] = failures["customer_tweet_id"].map(failure_map)

# Safety check.
missing = failures[failures["failure_mechanism"].isna()]

if len(missing) > 0:
    print("ERROR: Some failures have no assigned mechanism:")
    print(missing[["customer_tweet_id", "golden_customer_text"]])
    raise SystemExit(1)

columns = [
    "customer_tweet_id",
    "golden_customer_text",
    "golden_intent",
    "vector_top1_customer_text",
    "vector_top1_response",
    "vector_relevance",
    "failure_mechanism",
    "review_notes",
]

failures[columns].to_csv(OUTPUT_FILE, index=False)

print("Vector retrieval failures:", len(failures))
print()
print("Failure mechanisms:")
print(
    failures["failure_mechanism"]
    .value_counts()
    .sort_values(ascending=False)
)

print()
print("Percent of failures:")
print(
    failures["failure_mechanism"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()
print(f"Saved to: {OUTPUT_FILE}")