from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "end_to_end_golden_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "reply_failure_analysis.csv"
)


df = pd.read_csv(INPUT_PATH)


# Conservative diagnostic patterns.
# These identify replies that deserve manual inspection.
UNSUPPORTED_PATTERNS = [
    r"\b4g is active\b",
    r"\bthere is 4g coverage\b",
    r"\bwe are aware of the issue\b",
    r"\bwe are currently working\b",
    r"\bwork is ongoing\b",
    r"\bour engineers are\b",
    r"\bwe have (?:checked|verified|investigated|resolved|escalated)\b",
    r"\bthe issue (?:has been|is being) resolved\b",
    r"\bservice has been (?:activated|deactivated|restored)\b",
    r"\bcaller tune service has been deactivated\b",
]

GENERIC_PATTERNS = [
    r"\bprovide your (?:mobile )?number\b",
    r"\bshare your (?:mobile )?number\b",
    r"\bprovide your phone number\b",
    r"\bshare your phone number\b",
    r"\bshare your number\b",
    r"\bprovide your number\b",
    r"\bshare your .*location\b",
    r"\bprovide your .*location\b",
    r"\bshare .*device\b",
    r"\bprovide .*device\b",
    r"\bassist you better\b",
    r"\bassist you further\b",
]

HISTORICAL_CONTAMINATION_PATTERNS = [
    r"\bI bought\b",
    r"\bI recharge\b",
    r"\bI recharged\b",
    r"\bI didn't\b",
    r"\bI do not\b",
    r"\bI want to\b",
    r"\bCould you clarify\b",
]


def matches_any(text, patterns):
    text = str(text or "")
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


df["unsupported_claim_signal"] = (
    df["draft_reply"]
    .fillna("")
    .apply(
        lambda x: matches_any(
            x,
            UNSUPPORTED_PATTERNS,
        )
    )
)

df["generic_reply_signal"] = (
    df["draft_reply"]
    .fillna("")
    .apply(
        lambda x: matches_any(
            x,
            GENERIC_PATTERNS,
        )
    )
)

df["historical_contamination_signal"] = (
    df["draft_reply"]
    .fillna("")
    .apply(
        lambda x: matches_any(
            x,
            HISTORICAL_CONTAMINATION_PATTERNS,
        )
    )
)

df["intent_error"] = (
    df["golden_intent"]
    != df["predicted_intent"]
)

df["escalation_false_positive"] = (
    (df["golden_escalation_expected"].astype(str).str.lower() == "no")
    & (df["actual_decision"].astype(str).str.lower() == "human")
)


# Keep only useful analysis columns.
result = df[
    [
        "golden_id",
        "customer_text",
        "golden_intent",
        "predicted_intent",
        "confidence",
        "golden_escalation_expected",
        "actual_decision",
        "draft_reply",
        "intent_error",
        "escalation_false_positive",
        "unsupported_claim_signal",
        "generic_reply_signal",
        "historical_contamination_signal",
    ]
].copy()


result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("=" * 80)
print("REPLY FAILURE ANALYSIS")
print("=" * 80)

total = len(result)

print(f"\nGolden examples: {total}")

for column, label in [
    (
        "generic_reply_signal",
        "Generic reply signal",
    ),
    (
        "unsupported_claim_signal",
        "Unsupported claim signal",
    ),
    (
        "historical_contamination_signal",
        "Historical contamination signal",
    ),
]:
    count = int(result[column].sum())
    print(
        f"{label}: {count}/{total} "
        f"({count / total:.2%})"
    )


print("\nExamples requiring inspection:")

inspection_mask = (
    result["generic_reply_signal"]
    | result["unsupported_claim_signal"]
    | result["historical_contamination_signal"]
)

inspection = result.loc[
    inspection_mask,
    [
        "golden_id",
        "golden_intent",
        "predicted_intent",
        "generic_reply_signal",
        "unsupported_claim_signal",
        "historical_contamination_signal",
        "draft_reply",
    ],
]

print(
    inspection.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_PATH)

print("=" * 80)
print("REPLY ANALYSIS COMPLETE")
print("=" * 80)