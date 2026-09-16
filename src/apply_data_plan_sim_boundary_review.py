from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_data_plan_sim_boundary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "review_data_plan_sim_boundary_decided.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Manual decisions
# ---------------------------------------------------------

decisions = {

    46215: (
        "sim_or_device_issue",
        "The customer asks whether a SIM swap is required for 4G LTE on a 4G phone. The core support action concerns SIM/device compatibility."
    ),

    49720: (
        "non_actionable_or_context_required",
        "The customer provides SIM serial, PUK and alternate-number information but does not state the underlying problem clearly enough to determine a specific intent from this message alone."
    ),

    49723: (
        "sim_or_device_issue",
        "The customer is discussing SIM-related support and provides the mobile number; the surrounding historical response explicitly requests new SIM serial and PUK information."
    ),

    49734: (
        "data_plan_issue",
        "The customer says the SIM was given 3.2GB, indicating the resolved issue concerned a data allocation/benefit."
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
print("DATA_PLAN <-> SIM_DEVICE BOUNDARY DECISIONS")
print("=" * 80)

print(f"\nCases reviewed: {len(df)}")

print("\nFinal candidate intent counts:")

print(
    df["final_candidate_intent"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# Decisions
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