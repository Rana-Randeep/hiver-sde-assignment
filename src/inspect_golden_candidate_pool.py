from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_candidate_pool.csv")
OUTPUT_FILE = Path("results/golden_candidate_pool_review.csv")


def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def contains_any(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["customer_text_clean"] = clean_text(df["customer_text"])

    text = df["customer_text_clean"].str.lower()

    # Basic observable signals.
    df["text_length"] = df["customer_text_clean"].str.len()
    df["has_question_mark"] = df["customer_text_clean"].str.contains(
        r"\?", regex=True, na=False
    )
    df["has_number"] = df["customer_text_clean"].str.contains(
        r"\d", regex=True, na=False
    )

    # Topic-oriented lexical signals.
    network_keywords = [
        "network",
        "signal",
        "coverage",
        "mast",
        "4g",
        "3g",
        "lte",
        "no service",
        "network fluctuating",
    ]

    data_keywords = [
        "data",
        "browse",
        "browsing",
        "internet",
        "megabyte",
        "mb",
        "gb",
        "bundle",
        "subscription",
    ]

    recharge_keywords = [
        "recharge",
        "airtime",
        "credit",
        "top up",
        "topup",
        "refill",
    ]

    sim_device_keywords = [
        "sim",
        "puk",
        "pin",
        "phone",
        "device",
        "handset",
        "4g sim",
    ]

    vas_charge_keywords = [
        "deduct",
        "deducted",
        "charged",
        "charge",
        "billing",
        "subscription",
        "unsolicited",
        "unauthorized",
        "service",
        "vas",
    ]

    bonus_promo_keywords = [
        "bonus",
        "promo",
        "promotion",
        "offer",
        "bumpa",
        "bonus data",
    ]

    account_general_keywords = [
        "account",
        "customer care",
        "customer service",
        "help",
        "how do i",
        "please",
        "requirements",
        "activate",
        "transfer",
        "migrate",
    ]

    low_information_keywords = [
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "pls",
        "please",
        "help",
    ]

    df["signal_network"] = text.apply(
        lambda x: contains_any(x, network_keywords)
    )

    df["signal_data"] = text.apply(
        lambda x: contains_any(x, data_keywords)
    )

    df["signal_recharge"] = text.apply(
        lambda x: contains_any(x, recharge_keywords)
    )

    df["signal_sim_device"] = text.apply(
        lambda x: contains_any(x, sim_device_keywords)
    )

    df["signal_vas_charge"] = text.apply(
        lambda x: contains_any(x, vas_charge_keywords)
    )

    df["signal_bonus_promo"] = text.apply(
        lambda x: contains_any(x, bonus_promo_keywords)
    )

    df["signal_account_general"] = text.apply(
        lambda x: contains_any(x, account_general_keywords)
    )

    df["signal_low_information"] = text.apply(
        lambda x: contains_any(x, low_information_keywords)
    )

    # Useful for identifying possible escalation/context cases.
    escalation_keywords = [
        "number",
        "phone number",
        "account",
        "personal",
        "my money",
        "deducted",
        "charged",
        "refund",
        "complaint",
        "fraud",
        "unauthorized",
        "stolen",
        "blocked",
        "cannot",
        "can't",
        "unable",
    ]

    df["signal_possible_escalation"] = text.apply(
        lambda x: contains_any(x, escalation_keywords)
    )

    # Very short messages are useful candidates for
    # non-actionable/context-dependent evaluation.
    df["signal_short_message"] = df["text_length"] <= 45

    # Count how many broad topic signals fire.
    signal_columns = [
        "signal_network",
        "signal_data",
        "signal_recharge",
        "signal_sim_device",
        "signal_vas_charge",
        "signal_bonus_promo",
        "signal_account_general",
    ]

    df["topic_signal_count"] = df[signal_columns].sum(axis=1)

    # Candidate review priority.
    #
    # This is NOT a model score and NOT a label.
    # It simply helps us find potentially useful difficult cases.
    df["review_priority"] = (
        df["signal_short_message"].astype(int)
        + df["signal_possible_escalation"].astype(int)
        + (df["topic_signal_count"] >= 2).astype(int)
        + df["signal_low_information"].astype(int)
    )

    # Keep useful review columns together.
    review_columns = [
        "customer_tweet_id",
        "customer_text",
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
        "brand_response",
    ]

    review_df = df[review_columns].copy()

    # Sort potentially interesting cases first for manual review.
    review_df = review_df.sort_values(
        ["review_priority", "topic_signal_count", "text_length"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(OUTPUT_FILE, index=False)

    print("Golden candidate pool inspection created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows inspected: {len(review_df)}")
    print()

    print("Message length statistics:")
    print(
        review_df["text_length"]
        .describe()
        .round(2)
        .to_string()
    )
    print()

    print("Observable signal counts:")
    signal_summary = {
        "network": review_df["signal_network"].sum(),
        "data": review_df["signal_data"].sum(),
        "recharge": review_df["signal_recharge"].sum(),
        "sim/device": review_df["signal_sim_device"].sum(),
        "VAS/charge": review_df["signal_vas_charge"].sum(),
        "bonus/promotion": review_df["signal_bonus_promo"].sum(),
        "account/general": review_df["signal_account_general"].sum(),
        "low-information": review_df["signal_low_information"].sum(),
        "possible-escalation": review_df["signal_possible_escalation"].sum(),
        "short-message": review_df["signal_short_message"].sum(),
    }

    for name, count in signal_summary.items():
        percentage = (count / len(review_df)) * 100
        print(f"{name:20s}: {count:3d} ({percentage:5.1f}%)")

    print()

    print("Topic signal count distribution:")
    print(
        review_df["topic_signal_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("Review priority distribution:")
    print(
        review_df["review_priority"]
        .value_counts()
        .sort_index(ascending=False)
        .to_string()
    )
    print()

    print("Top 20 high-priority candidates:")
    print(
        review_df[
            [
                "customer_tweet_id",
                "customer_text",
                "topic_signal_count",
                "signal_possible_escalation",
                "signal_short_message",
                "review_priority",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()