from pathlib import Path
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "end_to_end_golden_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "top_failure_modes.csv"
)

EXAMPLES_PATH = (
    PROJECT_ROOT
    / "results"
    / "top_failure_examples.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "golden_id",
    "customer_text",
    "golden_intent",
    "predicted_intent",
    "confidence",
    "golden_escalation_expected",
    "actual_decision",
    "draft_reply",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce",
)


# ============================================================
# FAILURE SIGNALS
# ============================================================

# 1. Intent classification error
df["intent_error"] = (
    df["golden_intent"]
    != df["predicted_intent"]
)


# 2. Escalation false positive
#
# Expected:
#   yes -> human
#   no  -> auto
#
df["escalation_false_positive"] = (
    (df["golden_escalation_expected"] == "no")
    & (df["actual_decision"] == "human")
)


# 3. Unsupported capability/current-state language
#
# These are deliberately conservative patterns.
# They are diagnostic signals, not proof that every match
# is definitely wrong.
#
CAPABILITY_PATTERNS = [
    r"\bwe (?:have )?checked\b",
    r"\bwe (?:have )?verified\b",
    r"\bwe (?:have )?investigated\b",
    r"\bwe (?:have )?escalated\b",
    r"\bwe (?:have )?resolved\b",
    r"\bthe issue has been resolved\b",
    r"\bthe issue is being resolved\b",
    r"\byour service has been restored\b",
    r"\byour service has been activated\b",
    r"\byour service has been deactivated\b",
    r"\bwe (?:have )?processed your refund\b",
    r"\byour account shows\b",
    r"\byour auto renewal was cancelled\b",
    r"\bthe issue has been escalated\b",
    r"\bwe are currently working on\b",
    r"\bwe are aware of the concerns\b",
]

capability_regex = re.compile(
    "|".join(CAPABILITY_PATTERNS),
    flags=re.IGNORECASE,
)

df["unsupported_claim_signal"] = (
    df["draft_reply"]
    .fillna("")
    .apply(
        lambda text: bool(
            capability_regex.search(str(text))
        )
    )
)


# 4. Generic / weak clarification signal
#
# A diagnostic heuristic for replies that mostly ask
# for a phone number/details without addressing the issue.
#
GENERIC_PATTERNS = [
    r"\bkindly provide your (?:phone )?number\b",
    r"\bplease provide your (?:phone )?number\b",
    r"\bprovide your number\b",
    r"\bshare your number\b",
    r"\bprovide the relevant details\b",
    r"\bshare the relevant details\b",
    r"\bassist you further\b",
]

generic_regex = re.compile(
    "|".join(GENERIC_PATTERNS),
    flags=re.IGNORECASE,
)

df["generic_reply_signal"] = (
    df["draft_reply"]
    .fillna("")
    .apply(
        lambda text: bool(
            generic_regex.search(str(text))
        )
    )
)


# 5. Low-confidence prediction
#
# Use a transparent diagnostic threshold.
# This is NOT presented as the production escalation
# threshold.
LOW_CONFIDENCE_THRESHOLD = 0.50

df["low_confidence"] = (
    df["confidence"]
    < LOW_CONFIDENCE_THRESHOLD
)


# ============================================================
# FAILURE SUMMARY
# ============================================================

total = len(df)

failure_rows = [
    {
        "failure_mode": "intent_misclassification",
        "count": int(df["intent_error"].sum()),
        "rate": round(
            df["intent_error"].mean(),
            4,
        ),
        "definition": (
            "Predicted intent differs from golden intent."
        ),
    },
    {
        "failure_mode": "escalation_false_positive",
        "count": int(
            df["escalation_false_positive"].sum()
        ),
        "rate": round(
            df["escalation_false_positive"].mean(),
            4,
        ),
        "definition": (
            "System escalated an example marked as "
            "not requiring human escalation."
        ),
    },
    {
        "failure_mode": "unsupported_claim_signal",
        "count": int(
            df["unsupported_claim_signal"].sum()
        ),
        "rate": round(
            df["unsupported_claim_signal"].mean(),
            4,
        ),
        "definition": (
            "Reply contains language suggesting "
            "unsupported live/internal action or state."
        ),
    },
    {
        "failure_mode": "generic_reply_signal",
        "count": int(
            df["generic_reply_signal"].sum()
        ),
        "rate": round(
            df["generic_reply_signal"].mean(),
            4,
        ),
        "definition": (
            "Reply relies on generic information requests "
            "rather than directly addressing the issue."
        ),
    },
    {
        "failure_mode": "low_confidence_prediction",
        "count": int(
            df["low_confidence"].sum()
        ),
        "rate": round(
            df["low_confidence"].mean(),
            4,
        ),
        "definition": (
            f"Classifier confidence below "
            f"{LOW_CONFIDENCE_THRESHOLD:.2f}."
        ),
    },
]

summary_df = pd.DataFrame(failure_rows)

summary_df = summary_df.sort_values(
    "count",
    ascending=False,
).reset_index(drop=True)

summary_df.insert(
    0,
    "rank",
    range(1, len(summary_df) + 1),
)


# ============================================================
# REAL EXAMPLES
# ============================================================

example_frames = []

for _, row in summary_df.iterrows():

    mode = row["failure_mode"]

    if mode == "intent_misclassification":
        mask = df["intent_error"]

    elif mode == "escalation_false_positive":
        mask = df["escalation_false_positive"]

    elif mode == "unsupported_claim_signal":
        mask = df["unsupported_claim_signal"]

    elif mode == "generic_reply_signal":
        mask = df["generic_reply_signal"]

    elif mode == "low_confidence_prediction":
        mask = df["low_confidence"]

    else:
        continue

    examples = df.loc[
        mask,
        [
            "golden_id",
            "customer_text",
            "golden_intent",
            "predicted_intent",
            "confidence",
            "golden_escalation_expected",
            "actual_decision",
            "draft_reply",
        ],
    ].copy()

    examples.insert(
        0,
        "failure_mode",
        mode,
    )

    # Keep the most useful examples first.
    if mode == "intent_misclassification":
        examples = examples.sort_values(
            "confidence",
            ascending=False,
        )

    elif mode == "low_confidence_prediction":
        examples = examples.sort_values(
            "confidence",
            ascending=True,
        )

    examples = examples.head(10)

    example_frames.append(examples)


examples_df = pd.concat(
    example_frames,
    ignore_index=True,
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

examples_df.to_csv(
    EXAMPLES_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("TOP FAILURE MODE ANALYSIS")
print("=" * 80)

print(f"\nGolden examples analysed: {total}")

print("\nFailure-mode summary:")
print(
    summary_df[
        [
            "rank",
            "failure_mode",
            "count",
            "rate",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_PATH)
print(EXAMPLES_PATH)

print("\nTop examples:")
print(
    examples_df[
        [
            "failure_mode",
            "golden_id",
            "golden_intent",
            "predicted_intent",
            "confidence",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 80)