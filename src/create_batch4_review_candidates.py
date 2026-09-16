import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_FILE = PROJECT_ROOT / "results" / "golden_review_sheet.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "batch4_review_candidates.csv"


# ============================================================
# LOAD REVIEW POOL
# ============================================================

review = pd.read_csv(REVIEW_FILE)

# IDs already present in the current golden set
golden_file = PROJECT_ROOT / "results" / "golden_set_working.csv"
golden = pd.read_csv(golden_file)

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

review_ids = review["customer_tweet_id"].astype(str)

remaining = review[
    ~review_ids.isin(golden_ids)
].copy()


# ============================================================
# TARGETED BATCH 4 CANDIDATES
# ============================================================
#
# These IDs were selected from the coverage analysis output.
# They are REVIEW candidates, NOT final golden labels.
# ============================================================

selected_ids = {
    # --------------------------------------------------------
    # 1. Account / general support
    # --------------------------------------------------------
    2486781: "underrepresented account/registered-line case",
    598439: "minimal account/number-only context",

    # --------------------------------------------------------
    # 2. Voice / line
    # --------------------------------------------------------
    918858: "clear voice/calling failure",
    1371044: "context-dependent line/dialling case",

    # --------------------------------------------------------
    # 3. SIM / device
    # --------------------------------------------------------
    2054146: "possible SIM/network boundary case",
    2500307: "combined voice + data + device case",

    # --------------------------------------------------------
    # 4. Bonus / promotion
    # --------------------------------------------------------
    1825267: "rare bonus-related case",
    2330536: "promotion/bonus eligibility boundary",

    # --------------------------------------------------------
    # 5. Network / data boundary
    # --------------------------------------------------------
    2463673: "network outage / location-specific escalation",
    539077: "persistent location-specific network complaint",
    2965614: "generic internet/data failure",
    2005903: "connection failure with location + device context",

    # --------------------------------------------------------
    # 6. Unauthorized charge / recharge boundary
    # --------------------------------------------------------
    1462270: "recurring unauthorized airtime deduction",
    1079213: "recharge + unwanted auto-renewal + deduction",
    575889: "recharge case with incomplete context",

    # --------------------------------------------------------
    # 7. Ambiguous / context-dependent
    # --------------------------------------------------------
    2693389: "customer-service complaint without explicit issue",
    209853: "prior-thread/context-dependent complaint",
    160989: "no-response / missing-context case",

    # --------------------------------------------------------
    # 8. Escalation / multi-issue
    # --------------------------------------------------------
    9099: "repeated identity requests + refund demand",
}


# ============================================================
# FILTER CANDIDATES
# ============================================================

selected_ids_str = {
    str(tweet_id)
    for tweet_id in selected_ids.keys()
}

batch4 = remaining[
    remaining["customer_tweet_id"].astype(str).isin(
        selected_ids_str
    )
].copy()


# ============================================================
# ADD REVIEW REASON
# ============================================================

batch4["batch4_review_reason"] = (
    batch4["customer_tweet_id"]
    .astype(int)
    .map(selected_ids)
)


# ============================================================
# PRESERVE A CLEAN REVIEW ORDER
# ============================================================

ordered_ids = [
    2486781,
    598439,
    918858,
    1371044,
    2054146,
    2500307,
    1825267,
    2330536,
    2463673,
    539077,
    2965614,
    2005903,
    1462270,
    1079213,
    575889,
    2693389,
    209853,
    160989,
    9099,
]

order_map = {
    str(tweet_id): position
    for position, tweet_id in enumerate(ordered_ids)
}

batch4["batch4_order"] = (
    batch4["customer_tweet_id"]
    .astype(str)
    .map(order_map)
)

batch4 = batch4.sort_values("batch4_order")


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("BATCH 4 REVIEW CANDIDATE GENERATION")
print("=" * 70)

print(f"\nRemaining candidate pool : {len(remaining)}")
print(f"Requested review set     : {len(selected_ids)}")
print(f"Found candidates          : {len(batch4)}")

missing_ids = [
    tweet_id
    for tweet_id in selected_ids.keys()
    if str(tweet_id) not in set(
        batch4["customer_tweet_id"].astype(str)
    )
]

if missing_ids:
    print("\nWARNING - IDs not found:")
    print(missing_ids)
else:
    print("\nAll requested candidate IDs were found.")


# ============================================================
# DISPLAY REVIEW TABLE
# ============================================================

display_columns = [
    "customer_tweet_id",
    "customer_text",
    "batch4_review_reason",
]

if "review_priority" in batch4.columns:
    display_columns.append("review_priority")

if "case_type" in batch4.columns:
    display_columns.append("case_type")

if "escalation_expected" in batch4.columns:
    display_columns.append("escalation_expected")

print("\n" + "=" * 70)
print("BATCH 4 REVIEW CANDIDATES")
print("=" * 70)

print(
    batch4[display_columns]
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

batch4.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nThese are REVIEW candidates only.")
print("Do not treat batch4_review_reason as the final human label.")