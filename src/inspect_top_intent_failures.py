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
    / "top_intent_failure_examples.csv"
)


df = pd.read_csv(INPUT_PATH)


PAIRS = [
    ("data_issue", "network_issue"),
    ("data_issue", "data_plan_issue"),
    ("network_issue", "data_plan_issue"),
    ("unauthorized_charge_or_vas", "data_plan_issue"),
]


frames = []

for golden_intent, predicted_intent in PAIRS:

    subset = df[
        (df["golden_intent"] == golden_intent)
        & (df["predicted_intent"] == predicted_intent)
    ].copy()

    if subset.empty:
        continue

    subset = subset[
        [
            "golden_id",
            "customer_text",
            "golden_intent",
            "predicted_intent",
            "confidence",
            "golden_escalation_expected",
            "actual_decision",
            "draft_reply",
        ]
    ]

    frames.append(subset)


if not frames:
    raise ValueError("No matching confusion examples found.")


result = pd.concat(
    frames,
    ignore_index=True,
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("=" * 80)
print("TOP INTENT FAILURE EXAMPLES")
print("=" * 80)

for golden_intent, predicted_intent in PAIRS:

    subset = result[
        (result["golden_intent"] == golden_intent)
        & (result["predicted_intent"] == predicted_intent)
    ]

    if subset.empty:
        continue

    print(
        f"\n{'-' * 80}\n"
        f"{golden_intent} -> {predicted_intent}"
    )

    for _, row in subset.iterrows():

        print(f"\nGolden ID: {row['golden_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Confidence: {row['confidence']:.4f}")
        print(
            f"Escalation expected: "
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