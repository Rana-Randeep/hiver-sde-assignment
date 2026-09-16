import pandas as pd
from pathlib import Path


INPUT_PATH = Path("results/taxonomy_boundary_decisions.csv")


df = pd.read_csv(INPUT_PATH)


print("=" * 80)
print("TAXONOMY BOUNDARY DECISION VALIDATION")
print("=" * 80)

print(f"\nTotal review rows: {len(df)}")
print(f"Unique customer tweet IDs: {df['customer_tweet_id'].nunique()}")


# Count how many distinct final intents each customer message received.
decision_summary = (
    df.groupby("customer_tweet_id")["final_candidate_intent"]
    .agg(
        unique_intents="nunique",
        intents=lambda x: sorted(set(x))
    )
    .reset_index()
)


conflicts = decision_summary[
    decision_summary["unique_intents"] > 1
].copy()


print("\n" + "=" * 80)
print("DECISION CONSISTENCY")
print("=" * 80)

print(f"\nMessages with one consistent final intent: "
      f"{(decision_summary['unique_intents'] == 1).sum()}")

print(f"Messages with conflicting final intents: "
      f"{len(conflicts)}")


if conflicts.empty:
    print("\nNo conflicting decisions found.")
else:
    print("\nConflicting customer IDs:")
    print(conflicts.to_string(index=False))


    print("\n" + "=" * 80)
    print("CONFLICT DETAILS")
    print("=" * 80)

    for tweet_id in conflicts["customer_tweet_id"]:
        print("\n" + "-" * 80)
        print(f"customer_tweet_id: {tweet_id}")

        rows = df[df["customer_tweet_id"] == tweet_id]

        print(f"customer_text: {rows.iloc[0]['customer_text']}")

        print("\nReviews:")

        for _, row in rows.iterrows():
            print(
                f"- boundary_source: {row['boundary_source']}\n"
                f"  final_candidate_intent: "
                f"{row['final_candidate_intent']}\n"
                f"  reason: {row['decision_reason']}"
            )


print("\n" + "=" * 80)
print("DECISION DISTRIBUTION BY UNIQUE MESSAGE")
print("=" * 80)

unique_message_decisions = (
    df.sort_values("customer_tweet_id")
    .drop_duplicates("customer_tweet_id")
)

print(
    unique_message_decisions["final_candidate_intent"]
    .value_counts()
    .to_string()
)


print("\nEND")
print("=" * 80)