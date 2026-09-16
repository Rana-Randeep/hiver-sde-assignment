from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_data_plan_voice_boundary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_data_plan_voice_boundary_decided.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Manual decisions
# ---------------------------------------------------------

decisions = {

    9089: (
        "non_actionable_or_context_required",
        "The customer refers to repeated prior interactions but does not state the underlying issue clearly enough to determine a specific service intent."
    ),

    21374: (
        "unauthorized_charge_or_vas",
        "The customer reports an unauthorized charge for caller tunes. This is a VAS/unauthorized-charge issue rather than a general voice-line problem."
    ),

    15649: (
        "data_plan_issue",
        "The customer is primarily discussing a 50MB bonus/data benefit, inability to access the service, and the data not lasting. The core issue is the data/bonus service."
    ),

    15658: (
        "data_plan_issue",
        "The customer reports missing free data and being charged per MB. The primary support problem concerns a data benefit/data service."
    ),
}


# ---------------------------------------------------------
# Apply decisions
# ---------------------------------------------------------

df["final_candidate_intent"] = ""
df["reason"] = ""


for index, row in df.iterrows():

    tweet_id = int(row["customer_tweet_id"])

    if tweet_id not in decisions:
        raise ValueError(
            f"No manual decision found for tweet ID {tweet_id}"
        )

    intent, reason = decisions[tweet_id]

    df.at[index, "final_candidate_intent"] = intent
    df.at[index, "reason"] = reason


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("=" * 80)
print("DATA_PLAN <-> VOICE_LINE BOUNDARY DECISIONS")
print("=" * 80)

print(f"\nCases reviewed: {len(df)}")

print("\nFinal candidate intent counts:")

print(
    df["final_candidate_intent"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# Show decisions
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DECISIONS")
print("=" * 80)


for _, row in df.iterrows():

    print("\n" + "-" * 80)

    print(
        f"customer_tweet_id: "
        f"{row['customer_tweet_id']}"
    )

    print(
        f"final_candidate_intent: "
        f"{row['final_candidate_intent']}"
    )

    print(
        f"reason: "
        f"{row['reason']}"
    )


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("\nSaved decided review:")
print(OUTPUT_FILE)

print("\nEND")
print("=" * 80)