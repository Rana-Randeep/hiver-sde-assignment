import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH4_FILE = (
    PROJECT_ROOT
    / "results"
    / "batch4_review_candidates_labeled.csv"
)

GOLDEN_FILE = (
    PROJECT_ROOT
    / "results"
    / "golden_set_working.csv"
)


# ============================================================
# EXPECTED TAXONOMY
# ============================================================

EXPECTED_INTENTS = {
    "network_issue",
    "data_issue",
    "data_plan_issue",
    "unauthorized_charge_or_vas",
    "recharge_or_airtime_issue",
    "sim_or_device_issue",
    "voice_or_line_issue",
    "bonus_or_promotion_issue",
    "account_or_general_support",
    "non_actionable_or_context_required",
}

EXPECTED_CASE_TYPES = {
    "ambiguous",
    "escalation_relevant",
    "low_information",
    "rare",
    "routine",
}


# ============================================================
# LOAD FILES
# ============================================================

batch4 = pd.read_csv(BATCH4_FILE)
golden = pd.read_csv(GOLDEN_FILE)


print("=" * 70)
print("BATCH 4 LABEL CONSISTENCY CHECK")
print("=" * 70)


# ============================================================
# 1. BASIC COUNTS
# ============================================================

print("\n1. BASIC COUNTS")

print(f"Current golden rows : {len(golden)}")
print(f"Batch 4 rows        : {len(batch4)}")
print(f"Expected after merge: {len(golden) + len(batch4)}")


# ============================================================
# 2. DUPLICATE BATCH 4 IDs
# ============================================================

print("\n2. BATCH 4 DUPLICATE CHECK")

duplicate_batch4 = batch4[
    batch4["customer_tweet_id"].duplicated(keep=False)
]

if len(duplicate_batch4) == 0:
    print("PASS - No duplicate IDs inside Batch 4.")
else:
    print("FAIL - Duplicate IDs found:")
    print(
        duplicate_batch4[
            ["customer_tweet_id", "customer_text"]
        ].to_string(index=False)
    )


# ============================================================
# 3. OVERLAP WITH EXISTING GOLDEN SET
# ============================================================

print("\n3. OVERLAP WITH CURRENT GOLDEN SET")

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

batch4_ids = set(
    batch4["customer_tweet_id"].astype(str)
)

overlap = golden_ids.intersection(batch4_ids)

if len(overlap) == 0:
    print("PASS - No Batch 4 IDs overlap with current golden set.")
else:
    print("FAIL - Overlapping IDs:")
    print(sorted(overlap))


# ============================================================
# 4. MISSING LABEL CHECK
# ============================================================

print("\n4. MISSING LABEL CHECK")

required_columns = [
    "golden_intent",
    "case_type",
    "escalation_expected",
]

all_complete = True

for column in required_columns:

    missing = batch4[column].isna().sum()

    if missing == 0:
        print(f"PASS - {column}: no missing values")
    else:
        print(f"FAIL - {column}: {missing} missing values")
        all_complete = False


# ============================================================
# 5. INTENT TAXONOMY CHECK
# ============================================================

print("\n5. INTENT TAXONOMY CHECK")

batch4_intents = set(
    batch4["golden_intent"].dropna().unique()
)

unknown_intents = batch4_intents - EXPECTED_INTENTS

print("Batch 4 intents:")
for intent in sorted(batch4_intents):
    print(f"  {intent}")

if not unknown_intents:
    print("\nPASS - All Batch 4 intents are in the expected taxonomy.")
else:
    print("\nFAIL - Unknown intent values:")
    for intent in sorted(unknown_intents):
        print(f"  {intent}")


# ============================================================
# 6. CASE TYPE TAXONOMY CHECK
# ============================================================

print("\n6. CASE TYPE TAXONOMY CHECK")

batch4_case_types = set(
    batch4["case_type"].dropna().unique()
)

unknown_case_types = (
    batch4_case_types - EXPECTED_CASE_TYPES
)

print("Batch 4 case types:")
for case_type in sorted(batch4_case_types):
    print(f"  {case_type}")

if not unknown_case_types:
    print("\nPASS - All Batch 4 case types are in the existing taxonomy.")
else:
    print("\nFAIL - Unknown case_type values:")
    for case_type in sorted(unknown_case_types):
        print(f"  {case_type}")


# ============================================================
# 7. ESCALATION VALUE CHECK
# ============================================================

print("\n7. ESCALATION VALUE CHECK")

allowed_escalation = {
    "yes",
    "no",
}

batch4_escalation = set(
    batch4["escalation_expected"]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
)

unknown_escalation = (
    batch4_escalation - allowed_escalation
)

print("Batch 4 escalation values:")
for value in sorted(batch4_escalation):
    print(f"  {value}")

if not unknown_escalation:
    print("\nPASS - Escalation values are valid.")
else:
    print("\nFAIL - Unknown escalation values:")
    for value in sorted(unknown_escalation):
        print(f"  {value}")


# ============================================================
# 8. DISTRIBUTION
# ============================================================

print("\n8. BATCH 4 INTENT DISTRIBUTION")

print(
    batch4["golden_intent"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\n8b. BATCH 4 CASE-TYPE DISTRIBUTION")

print(
    batch4["case_type"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\n8c. BATCH 4 ESCALATION DISTRIBUTION")

print(
    batch4["escalation_expected"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 9. FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

checks_passed = (
    len(batch4) == 19
    and len(duplicate_batch4) == 0
    and len(overlap) == 0
    and all_complete
    and not unknown_intents
    and not unknown_case_types
    and not unknown_escalation
)

if checks_passed:
    print("\nPASS")
    print("Batch 4 is structurally consistent and ready for")
    print("the next review step.")
else:
    print("\nREVIEW REQUIRED")
    print("Do NOT merge Batch 4 into the golden set yet.")


print("\nEND")