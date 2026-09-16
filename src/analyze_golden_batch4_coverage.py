import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_FILE = PROJECT_ROOT / "results" / "golden_review_sheet.csv"
GOLDEN_FILE = PROJECT_ROOT / "results" / "golden_set_working.csv"


# ============================================================
# LOAD DATA
# ============================================================

review = pd.read_csv(REVIEW_FILE)
golden = pd.read_csv(GOLDEN_FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("GOLDEN SET - BATCH 4 COVERAGE ANALYSIS")
print("=" * 70)

print(f"\nReview pool rows: {len(review)}")
print(f"Current golden rows: {len(golden)}")


# ============================================================
# IDENTIFY REMAINING CANDIDATES
# ============================================================

# A candidate is considered reviewed/used if its customer_tweet_id
# already exists in the current golden set.

golden_ids = set(golden["customer_tweet_id"].astype(str))

review["customer_tweet_id_str"] = (
    review["customer_tweet_id"].astype(str)
)

remaining = review[
    ~review["customer_tweet_id_str"].isin(golden_ids)
].copy()

print(f"Remaining candidates: {len(remaining)}")


# ============================================================
# CURRENT GOLDEN INTENT COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("1. CURRENT GOLDEN INTENT COVERAGE")
print("=" * 70)

intent_counts = (
    golden["golden_intent"]
    .value_counts()
    .sort_index()
)

print(intent_counts.to_string())


# ============================================================
# CURRENT GOLDEN CASE-TYPE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("2. CURRENT GOLDEN CASE-TYPE COVERAGE")
print("=" * 70)

case_counts = (
    golden["case_type"]
    .value_counts(dropna=False)
    .sort_index()
)

print(case_counts.to_string())


# ============================================================
# CURRENT GOLDEN ESCALATION COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("3. CURRENT GOLDEN ESCALATION COVERAGE")
print("=" * 70)

escalation_counts = (
    golden["escalation_expected"]
    .value_counts(dropna=False)
    .sort_index()
)

print(escalation_counts.to_string())


# ============================================================
# REMAINING CANDIDATE SAMPLING SIGNALS
# ============================================================

signal_columns = [
    "short_message",
    "low_information_signal",
    "possible_escalation",
]

available_signal_columns = [
    col for col in signal_columns
    if col in remaining.columns
]

if available_signal_columns:
    print("\n" + "=" * 70)
    print("4. REMAINING CANDIDATE SIGNAL COVERAGE")
    print("=" * 70)

    for col in available_signal_columns:
        print(f"\n{col}:")
        print(
            remaining[col]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# KEYWORD-BASED CANDIDATE GROUPS
# ============================================================
#
# These are NOT labels.
# They are only screening signals to help us inspect
# potentially underrepresented areas.
#
# Human annotation remains the ground truth.
# ============================================================

text = (
    remaining["customer_text"]
    .fillna("")
    .astype(str)
    .str.lower()
)

candidate_groups = {
    "account_or_general_candidate": [
        "account",
        "profile",
        "registered",
        "registration",
        "verify",
        "verification",
        "customer care",
        "customer service",
        "help me",
        "my number",
    ],

    "voice_line_candidate": [
        "call",
        "calling",
        "voice",
        "dial",
        "can't call",
        "cannot call",
        "unable to call",
        "make calls",
        "receive calls",
        "line",
        "emergency call",
    ],

    "sim_device_candidate": [
        "sim",
        "puk",
        "swap",
        "replace sim",
        "new sim",
        "handset",
        "phone",
        "device",
        "imei",
    ],

    "bonus_promotion_candidate": [
        "bonus",
        "promotion",
        "promo",
        "offer",
        "reward",
        "bumba",
        "bounce",
        "free airtime",
    ],

    "network_candidate": [
        "network",
        "signal",
        "coverage",
        "no network",
        "poor network",
        "no signal",
    ],

    "data_service_candidate": [
        "browse",
        "browsing",
        "internet",
        "data not working",
        "can't browse",
        "cannot browse",
        "slow",
        "download",
    ],

    "data_plan_candidate": [
        "data plan",
        "data bundle",
        "bundle",
        "subscribe",
        "subscription",
        "renew",
        "renewal",
        "mb",
        "gb",
    ],

    "recharge_airtime_candidate": [
        "recharge",
        "recharged",
        "airtime",
        "credit",
        "loaded",
        "load",
    ],

    "unauthorized_charge_candidate": [
        "deduct",
        "deducted",
        "deduction",
        "charged",
        "charge",
        "vas",
        "subscription i didn't",
        "didn't subscribe",
        "unauthorized",
    ],
}


print("\n" + "=" * 70)
print("5. REMAINING CANDIDATE SCREENING GROUPS")
print("=" * 70)

group_results = {}

for group_name, keywords in candidate_groups.items():

    pattern = "|".join(
        keyword.replace(" ", r"\s+")
        for keyword in keywords
    )

    mask = text.str.contains(
        pattern,
        regex=True,
        na=False
    )

    matched = remaining[mask].copy()

    group_results[group_name] = matched

    print(f"\n{group_name}: {len(matched)} candidates")

    if len(matched) > 0:
        display_columns = [
            "customer_tweet_id",
            "customer_text"
        ]

        if "review_priority" in matched.columns:
            display_columns.append("review_priority")

        print(
            matched[display_columns]
            .head(8)
            .to_string(index=False)
        )


# ============================================================
# POTENTIALLY HIGH-VALUE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("6. POTENTIALLY HIGH-VALUE REMAINING CANDIDATES")
print("=" * 70)

high_value = remaining.copy()

# Prefer explicit sampling signals where available.
if "possible_escalation" in high_value.columns:
    high_value["possible_escalation_flag"] = (
        high_value["possible_escalation"]
        .fillna(False)
        .astype(bool)
    )
else:
    high_value["possible_escalation_flag"] = False

if "short_message" in high_value.columns:
    high_value["short_message_flag"] = (
        high_value["short_message"]
        .fillna(False)
        .astype(bool)
    )
else:
    high_value["short_message_flag"] = False

if "low_information_signal" in high_value.columns:
    high_value["low_information_flag"] = (
        high_value["low_information_signal"]
        .fillna(False)
        .astype(bool)
    )
else:
    high_value["low_information_flag"] = False


# A simple screening score.
# IMPORTANT: this is NOT a model score and NOT a label.
high_value["screening_score"] = (
    high_value["possible_escalation_flag"].astype(int) * 3
    + high_value["low_information_flag"].astype(int) * 2
    + high_value["short_message_flag"].astype(int) * 1
)

high_value = high_value.sort_values(
    by=["screening_score"],
    ascending=False
)

display_columns = [
    "customer_tweet_id",
    "customer_text",
    "screening_score",
]

if "review_priority" in high_value.columns:
    display_columns.append("review_priority")

print(
    high_value[display_columns]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("7. SUMMARY")
print("=" * 70)

print(f"Current golden set : {len(golden)}")
print(f"Remaining pool     : {len(remaining)}")
print(f"Minimum Batch 4    : 13")
print(f"Recommended review : 20")

print("\nIMPORTANT:")
print("Keyword groups and screening scores are ONLY candidate")
print("selection aids. They are NOT human intent labels.")
print("No rows were selected or modified by this script.")

print("\nEND")