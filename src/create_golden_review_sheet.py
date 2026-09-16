from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_candidate_pool_review.csv")
OUTPUT_FILE = Path("results/golden_review_sheet.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    # Human annotation columns.
    df["golden_selected"] = ""
    df["golden_intent"] = ""
    df["confidence"] = ""
    df["case_type"] = ""
    df["escalation_expected"] = ""
    df["annotation_notes"] = ""

    # Put the human-review columns first.
    output_columns = [
        "golden_selected",
        "golden_intent",
        "confidence",
        "case_type",
        "escalation_expected",
        "annotation_notes",
        "customer_tweet_id",
        "customer_text",
        "brand_response",
        "text_length",
        "has_question_mark",
        "has_number",
        "signal_network",
        "signal_data",
        "signal_recharge",
        "signal_sim_device",
        "signal_vas_charge",
        "signal_bonus_promo",
        "signal_account_general",
        "signal_low_information",
        "signal_possible_escalation",
        "signal_short_message",
        "topic_signal_count",
        "review_priority",
    ]

    review_df = df[output_columns].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(OUTPUT_FILE, index=False)

    print("Golden review sheet created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows available for review: {len(review_df)}")
    print()

    print("Human annotation columns:")
    print("  golden_selected")
    print("  golden_intent")
    print("  confidence")
    print("  case_type")
    print("  escalation_expected")
    print("  annotation_notes")
    print()

    print("Allowed golden_intent values:")
    intents = [
        "network_issue",
        "data_service_issue",
        "data_plan_issue",
        "unauthorized_charge_or_vas",
        "recharge_or_airtime_issue",
        "sim_or_device_issue",
        "voice_or_line_issue",
        "bonus_or_promotion_issue",
        "account_or_general_support",
        "non_actionable_or_context_required",
    ]

    for intent in intents:
        print(f"  {intent}")

    print()

    print("Suggested case_type values:")
    print("  routine")
    print("  rare")
    print("  ambiguous")
    print("  low_information")
    print("  escalation_relevant")

    print()

    print("Suggested confidence values:")
    print("  high")
    print("  medium")
    print("  low")

    print()

    print("Suggested escalation_expected values:")
    print("  yes")
    print("  no")
    print("  uncertain")


if __name__ == "__main__":
    main()