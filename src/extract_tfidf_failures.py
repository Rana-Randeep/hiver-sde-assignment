import pandas as pd
from pathlib import Path


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "tfidf_retrieval_golden_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "tfidf_retrieval_failures.csv"


# -------------------------------------------------------------------
# Load retrieval evaluation
# -------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Keep only irrelevant retrievals
failures = df[df["relevance"] == 0].copy()

# Reset row numbering
failures = failures.reset_index(drop=True)


# -------------------------------------------------------------------
# Save failure cases
# -------------------------------------------------------------------

failures.to_csv(OUTPUT_FILE, index=False)


# -------------------------------------------------------------------
# Print summary
# -------------------------------------------------------------------

print("=" * 70)
print("TF-IDF RETRIEVAL FAILURE CASES")
print("=" * 70)

print(f"\nTotal golden examples: {len(df)}")
print(f"Irrelevant retrievals: {len(failures)}")

print("\nFailure rate:")
print(f"  {len(failures) / len(df):.2%}")

print("\nFailure file created:")
print(OUTPUT_FILE)

print("\nColumns:")
for column in failures.columns:
    print(f"  - {column}")

print("\nFirst 5 failure cases:")

for i, row in failures.head(5).iterrows():
    print("\n" + "-" * 70)
    print(f"Failure #{i + 1}")
    print(f"Golden intent:       {row['golden_intent']}")
    print(f"Golden message:      {row['golden_customer_text']}")
    print(f"Retrieved message:   {row['retrieved_customer_text']}")
    print(f"Similarity:          {row['similarity']:.4f}")
    print(f"Retrieved response:  {row['retrieved_gloCare_response']}")