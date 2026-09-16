import pandas as pd

INPUT_FILE = "results/vector_failure_analysis.csv"
OUTPUT_FILE = "results/vector_failure_intent_comparison.csv"

df = pd.read_csv(INPUT_FILE)

# Manually identified intent of the retrieved historical example.
# This is analysis only, not model training.
retrieved_intent_map = {
    2267327: "voice_or_line_issue",
    745771: "network_issue",
    317832: "bonus_or_promotion_issue",
    2910699: "recharge_or_airtime_issue",
    2218206: "unauthorized_charge_or_vas",
    1000324: "recharge_or_airtime_issue",
    1043403: "recharge_or_airtime_issue",
    2800841: "unauthorized_charge_or_vas",
    658682: "data_plan_issue",
    1931834: "unauthorized_charge_or_vas",
    49737: "data_plan_issue",
    598439: "account_or_general_support",
    1371044: "bonus_or_promotion_issue",
}

df["retrieved_intent"] = df["customer_tweet_id"].map(retrieved_intent_map)

# Check that every failure received an intent.
missing = df[df["retrieved_intent"].isna()]

if len(missing) > 0:
    print("ERROR: Missing retrieved intent labels:")
    print(missing[["customer_tweet_id", "golden_intent"]])
    raise SystemExit(1)

df["intent_match"] = (
    df["golden_intent"] == df["retrieved_intent"]
)

print("=" * 80)
print("VECTOR FAILURE — GOLDEN VS RETRIEVED INTENT")
print("=" * 80)

print(f"\nTotal failures: {len(df)}")

print("\nIntent comparison:")
print(
    df[
        [
            "customer_tweet_id",
            "golden_intent",
            "retrieved_intent",
            "intent_match",
            "failure_mechanism",
        ]
    ].to_string(index=False)
)

print("\nIntent match count:")
print(df["intent_match"].value_counts())

print("\nIntent match percentage:")
print(df["intent_match"].mean() * 100)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")