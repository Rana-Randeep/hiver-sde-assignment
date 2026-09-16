import pandas as pd
from pathlib import Path


INPUT = Path("results/pilot_candidates_taxonomy_v2.csv")
OUTPUT = Path("results/pilot_training_labels.csv")


FINAL_INTENTS = {
    "network_issue",
    "unauthorized_charge_or_vas",
    "data_plan_issue",
    "data_issue",
    "recharge_or_airtime_issue",
    "bonus_or_promotion_issue",
    "sim_or_device_issue",
    "voice_or_line_issue",
    "non_actionable_or_context_required",
    "account_or_general_support",
}


def main():
    df = pd.read_csv(INPUT)

    # Normalize the one intermediate label to the final taxonomy.
    df["intent"] = df["taxonomy_v2_intent"].replace(
        {"data_service_issue": "data_issue"}
    )

    # Keep only the fields required for supervised classification.
    training = df[
        ["pilot_id", "customer_tweet_id", "customer_text", "intent"]
    ].copy()

    # Validate labels.
    invalid = sorted(set(training["intent"]) - FINAL_INTENTS)

    if invalid:
        raise ValueError(f"Invalid intents found: {invalid}")

    if training["intent"].isna().any():
        raise ValueError("Some training examples have missing intent labels.")

    # Validate one row per pilot example.
    if training["pilot_id"].duplicated().any():
        raise ValueError("Duplicate pilot_id values found.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT, index=False)

    print("Created:", OUTPUT)
    print("Rows:", len(training))
    print("Columns:", list(training.columns))
    print()
    print("Intent distribution:")
    print(training["intent"].value_counts().to_string())
    print()
    print("Unique intents:", training["intent"].nunique())
    print("Missing intents:", training["intent"].isna().sum())
    print("Invalid intents:", invalid)


if __name__ == "__main__":
    main()
