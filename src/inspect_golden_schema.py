from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "results"
    / "tfidf_retrieval_golden_review.csv"
)


print("Loading golden review file...")

df = pd.read_csv(GOLDEN_PATH)

print()
print("=" * 70)
print("GOLDEN SET SCHEMA")
print("=" * 70)

print()
print("Shape:")
print(df.shape)

print()
print("Columns:")
for column in df.columns:
    print(f"- {column}")

print()
print("Data types:")
print(df.dtypes)

print()
print("=" * 70)
print("FIRST TWO ROWS")
print("=" * 70)

print(df.head(2).to_string())