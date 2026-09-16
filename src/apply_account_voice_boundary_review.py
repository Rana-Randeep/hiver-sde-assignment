import pandas as pd
from pathlib import Path


INPUT_PATH = Path("results/review_account_voice_boundary.csv")
OUTPUT_PATH = Path("results/review_account_voice_boundary_decided.csv")


decisions = {
    9089: (
        "non_actionable_or_context_required",
        "The customer says they have repeatedly provided their number, but the actual support problem or requested action is not clear from the message alone."
    ),
    21374: (
        "unauthorized_charge_or_vas",
        "The customer reports an unauthorized deduction for caller tunes they did not subscribe to. The core support action is investigation of an unauthorized VAS charge."
    ),
    28324: (
        "unauthorized_charge_or_vas",
        "The customer reports an unexplained account deduction, and the historical response asks about the subscription date, indicating a subscription or VAS-related charge."
    ),
    28340: (
        "voice_or_line_issue",
        "The historical response indicates that the support interaction concerns whether the customer's line is active, pointing to a line-related issue."
    ),
}


df = pd.read_csv(INPUT_PATH)


df["final_candidate_intent"] = df["customer_tweet_id"].map(
    lambda tweet_id: decisions[int(tweet_id)][0]
)

df["decision_reason"] = df["customer_tweet_id"].map(
    lambda tweet_id: decisions[int(tweet_id)][1]
)


df.to_csv(OUTPUT_PATH, index=False)


print("=" * 80)
print("ACCOUNT_GENERAL <-> VOICE_LINE BOUNDARY DECISIONS")
print("=" * 80)

print(f"\nCases reviewed: {len(df)}")

print("\nFinal candidate intent counts:")
print(df["final_candidate_intent"].value_counts().to_string())

print("\n" + "=" * 80)
print("DECISIONS")
print("=" * 80)

for _, row in df.iterrows():
    print("\n" + "-" * 80)
    print(f"customer_tweet_id: {row['customer_tweet_id']}")
    print(f"final_candidate_intent: {row['final_candidate_intent']}")
    print(f"reason: {row['decision_reason']}")

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("\nSaved decided review:")
print(OUTPUT_PATH.resolve())

print("\nEND")
print("=" * 80)