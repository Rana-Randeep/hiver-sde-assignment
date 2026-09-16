import pandas as pd
from pathlib import Path


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "tfidf_retrieval_failures.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "tfidf_failure_analysis.csv"


# -------------------------------------------------------------------
# Load failure cases
# -------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

if len(df) != 32:
    raise ValueError(
        f"Expected 32 failure cases, but found {len(df)}."
    )


# -------------------------------------------------------------------
# Add failure-analysis columns
# -------------------------------------------------------------------

df["failure_category"] = ""

df["failure_notes"] = ""


# -------------------------------------------------------------------
# Save review sheet
# -------------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# -------------------------------------------------------------------
# Print summary
# -------------------------------------------------------------------

print("=" * 70)
print("TF-IDF FAILURE ANALYSIS SHEET")
print("=" * 70)

print(f"\nFailure cases: {len(df)}")

print("\nFailure categories to use:")
print("  - wrong_topic_match")
print("  - keyword_overlap")
print("  - insufficient_context")
print("  - multiple_issues")
print("  - other")

print("\nOutput file created:")
print(OUTPUT_FILE)

print("\nColumns added:")
print("  - failure_category")
print("  - failure_notes")

print("\nFirst 5 cases:")
print(
    df[
        [
            "golden_intent",
            "golden_customer_text",
            "retrieved_customer_text",
            "similarity",
            "failure_category",
            "failure_notes",
        ]
    ].head().to_string(index=False)
)