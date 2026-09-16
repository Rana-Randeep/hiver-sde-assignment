from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_account_data_plan_boundary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_account_data_plan_boundary_decided.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Manual taxonomy decisions
# ---------------------------------------------------------
#
# These decisions are based on the customer's actual
# support need, not merely the candidate review buckets.
# ---------------------------------------------------------

decisions = {

    9089: (
        "non_actionable_or_context_required",
        "The message refers to repeated prior interactions but does not clearly state the underlying issue. Historical context is required."
    ),

    9099: (
        "account_or_general_support",
        "The customer requests a refund and describes an inaccessible line/account-specific situation. Resolution requires account-specific investigation."
    ),

    19085: (
        "data_plan_issue",
        "The customer explicitly asks how much data is included with a specific N5000 data plan."
    ),

    19090: (
        "data_plan_issue",
        "The customer asks how to unsubscribe from data auto-renewal after a data subscription."
    ),

    19092: (
        "data_plan_issue",
        "The customer explicitly asks how to unsubscribe from a data service."
    ),

    20364: (
        "non_actionable_or_context_required",
        "The message only provides a phone number and does not contain a sufficiently clear support request."
    ),

    21374: (
        "unauthorized_charge_or_vas",
        "The customer reports a charge for caller tunes that they never subscribed to."
    ),

    49720: (
        "sim_or_device_issue",
        "The message provides SIM serial, PUK and alternate-number information in the context of a SIM-related support case."
    ),

    49723: (
        "sim_or_device_issue",
        "The customer is dealing with SIM replacement and provides the information needed for SIM-related support."
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
print("ACCOUNT_GENERAL <-> DATA_PLAN BOUNDARY DECISIONS")
print("=" * 80)

print(f"\nCases reviewed: {len(df)}")

print("\nFinal candidate intent counts:")

print(
    df["final_candidate_intent"]
    .value_counts()
    .to_string()
)


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