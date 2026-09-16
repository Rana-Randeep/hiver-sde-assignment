from pathlib import Path
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
    / "escalation_failure_examples.csv"
)


df = pd.read_csv(INPUT_PATH)

# Normalize values so the comparison is robust.
expected = (
    df["golden_escalation_expected"]
    .astype(str)
    .str.strip()
    .str.lower()
)

actual = (
    df["actual_decision"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# False positive:
# golden says escalation is NOT expected,
# but the system escalated to human.
false_positive = df[
    (expected == "no")
    & (actual == "human")
].copy()

# False negative:
# golden says human escalation is expected,
# but system did not escalate.
false_negative = df[
    (expected == "yes")
    & (actual != "human")
].copy()


print("=" * 80)
print("ESCALATION FAILURE ANALYSIS")
print("=" * 80)

print(f"\nTotal golden examples: {len(df)}")

print(
    f"Expected human escalation: "
    f"{(expected == 'yes').sum()}"
)

print(
    f"Expected non-escalation: "
    f"{(expected == 'no').sum()}"
)

print(
    f"Actual human escalation: "
    f"{(actual == 'human').sum()}"
)

print(
    f"\nFalse-positive escalations: "
    f"{len(false_positive)} "
    f"({len(false_positive) / len(df):.2%})"
)

print(
    f"False-negative escalations: "
    f"{len(false_negative)} "
    f"({len(false_negative) / len(df):.2%})"
)


# ----------------------------------------------------------------
# Breakdown of false positives by intent
# ----------------------------------------------------------------

print("\n" + "-" * 80)
print("FALSE POSITIVES BY GOLDEN INTENT")
print("-" * 80)

fp_by_intent = (
    false_positive
    .groupby("golden_intent")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(
    fp_by_intent.to_string(index=False)
)


# ----------------------------------------------------------------
# Actual examples
# ----------------------------------------------------------------

columns = [
    "golden_id",
    "customer_text",
    "golden_intent",
    "predicted_intent",
    "confidence",
    "golden_escalation_expected",
    "actual_decision",
    "draft_reply",
]

examples = false_positive[columns].copy()

examples.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "-" * 80)
print("FALSE-POSITIVE ESCALATION EXAMPLES")
print("-" * 80)

for _, row in examples.iterrows():

    print(f"\nGolden ID: {row['golden_id']}")
    print(f"Customer: {row['customer_text']}")
    print(f"Golden intent: {row['golden_intent']}")
    print(f"Predicted intent: {row['predicted_intent']}")
    print(f"Confidence: {row['confidence']:.4f}")
    print(
        f"Expected escalation: "
        f"{row['golden_escalation_expected']}"
    )
    print(
        f"Actual decision: "
        f"{row['actual_decision']}"
    )
    print(f"Reply: {row['draft_reply']}")


print("\n" + "=" * 80)
print("Saved:")
print(OUTPUT_PATH)
print("=" * 80)