import pandas as pd
from pathlib import Path


INPUT_FILES = [
    Path("results/review_account_data_plan_boundary_decided.csv"),
    Path("results/review_data_plan_voice_boundary_decided.csv"),
    Path("results/review_data_plan_sim_boundary_decided.csv"),
    Path("results/review_account_voice_boundary_decided.csv"),
]

OUTPUT_PATH = Path("results/taxonomy_boundary_decisions.csv")


frames = []

for path in INPUT_FILES:
    print(f"Reading: {path}")

    df = pd.read_csv(path)

    # Keep track of which boundary produced the review case.
    df["boundary_source"] = path.stem

    frames.append(df)


combined_df = pd.concat(
    frames,
    ignore_index=True
)


# Sort for easier inspection.
combined_df = combined_df.sort_values(
    by="customer_tweet_id"
).reset_index(drop=True)


combined_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("CONSOLIDATED TAXONOMY BOUNDARY DECISIONS")
print("=" * 80)

print(f"\nTotal reviewed rows: {len(combined_df)}")

print(f"Unique customer tweet IDs: {combined_df['customer_tweet_id'].nunique()}")

print("\nFinal candidate intent counts:")
print(
    combined_df["final_candidate_intent"]
    .value_counts()
    .to_string()
)

print("\nBoundary source counts:")
print(
    combined_df["boundary_source"]
    .value_counts()
    .to_string()
)

print("\nDuplicate customer IDs across boundary reviews:")

duplicate_ids = (
    combined_df["customer_tweet_id"]
    .value_counts()
)

duplicate_ids = duplicate_ids[duplicate_ids > 1]

if duplicate_ids.empty:
    print("None")
else:
    print(duplicate_ids.to_string())

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("\nSaved consolidated review:")
print(OUTPUT_PATH.resolve())

print("\nEND")
print("=" * 80)