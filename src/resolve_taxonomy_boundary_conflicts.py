import pandas as pd
from pathlib import Path


INPUT_PATH = Path("results/taxonomy_boundary_decisions.csv")
OUTPUT_PATH = Path("results/taxonomy_boundary_decisions_resolved.csv")


df = pd.read_csv(INPUT_PATH)


# Canonical decision for the only conflicting message.
canonical_decisions = {
    49720: (
        "non_actionable_or_context_required",
        "The customer provides SIM serial, PUK and alternate-number information but does not state the underlying problem clearly enough to determine a specific intent from this message alone."
    )
}


for tweet_id, (intent, reason) in canonical_decisions.items():
    mask = df["customer_tweet_id"] == tweet_id

    df.loc[mask, "final_candidate_intent"] = intent
    df.loc[mask, "decision_reason"] = reason


df.to_csv(OUTPUT_PATH, index=False)


print("=" * 80)
print("RESOLVED TAXONOMY BOUNDARY CONFLICTS")
print("=" * 80)

print(f"\nTotal review rows: {len(df)}")
print(f"Unique customer tweet IDs: {df['customer_tweet_id'].nunique()}")

print("\nCanonical decision applied:")
for tweet_id, (intent, _) in canonical_decisions.items():
    print(f"- {tweet_id}: {intent}")


# Validate again after resolution.
decision_summary = (
    df.groupby("customer_tweet_id")["final_candidate_intent"]
    .nunique()
)

conflict_count = (decision_summary > 1).sum()

print("\nRemaining conflicting messages:", conflict_count)

if conflict_count == 0:
    print("All reviewed messages now have one consistent final intent.")
else:
    print("WARNING: conflicts still remain.")


print("\nFinal candidate intent counts:")
print(
    df.drop_duplicates("customer_tweet_id")["final_candidate_intent"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("\nSaved resolved review:")
print(OUTPUT_PATH.resolve())

print("\nEND")
print("=" * 80)