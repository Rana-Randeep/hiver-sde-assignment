import pandas as pd
from pathlib import Path


INPUT = Path("results/tfidf_baseline_validation_predictions.csv")
OUTPUT = Path("results/tfidf_baseline_failure_analysis.csv")


def classify_failure(row):
    true_intent = row["true_intent"]
    predicted_intent = row["predicted_intent"]
    text = str(row["customer_text"]).lower()

    # Clear intent-boundary confusion.
    if {
        true_intent,
        predicted_intent,
    } == {
        "data_issue",
        "data_plan_issue",
    }:
        return "intent_boundary_data_vs_data_plan"

    # Very short / low-context customer messages.
    if len(text.split()) <= 8:
        return "low_information_message"

    # Vague message where a keyword strongly triggered another class.
    if (
        true_intent == "non_actionable_or_context_required"
        and predicted_intent == "network_issue"
    ):
        return "lexical_trigger_network"

    # Remaining semantically similar classes.
    if (
        true_intent == "recharge_or_airtime_issue"
        and predicted_intent == "sim_or_device_issue"
    ):
        return "intent_boundary_recharge_vs_sim_device"

    if (
        true_intent == "account_or_general_support"
        and predicted_intent == "data_plan_issue"
    ):
        return "intent_boundary_account_vs_data_plan"

    if (
        true_intent == "data_plan_issue"
        and predicted_intent == "account_or_general_support"
    ):
        return "intent_boundary_data_plan_vs_account"

    if (
        true_intent == "bonus_or_promotion_issue"
        and predicted_intent == "sim_or_device_issue"
    ):
        return "lexical_or_sparse_training_data"

    if (
        true_intent == "voice_or_line_issue"
        and predicted_intent == "non_actionable_or_context_required"
    ):
        return "low_information_message"

    return "other"


def main():
    df = pd.read_csv(INPUT)

    failures = df[
        df["true_intent"] != df["predicted_intent"]
    ].copy()

    failures["failure_mode"] = failures.apply(
        classify_failure,
        axis=1,
    )

    failures["review_status"] = "needs_manual_review"

    failures.to_csv(
        OUTPUT,
        index=False,
    )

    print("=== TF-IDF Baseline Failure Analysis ===")
    print()
    print("Total validation examples:", len(df))
    print("Misclassified examples:", len(failures))
    print("Error rate:", round(len(failures) / len(df), 4))
    print()
    print("Failure modes:")
    print(
        failures["failure_mode"]
        .value_counts()
        .to_string()
    )
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()