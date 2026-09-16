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
    / "confidence_calibration_analysis.csv"
)


df = pd.read_csv(INPUT_PATH)

df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce",
)

df["correct"] = (
    df["golden_intent"]
    == df["predicted_intent"]
)


correct = df[df["correct"]].copy()
incorrect = df[~df["correct"]].copy()


summary = pd.DataFrame(
    [
        {
            "group": "correct_predictions",
            "count": len(correct),
            "mean_confidence": correct["confidence"].mean(),
            "median_confidence": correct["confidence"].median(),
            "min_confidence": correct["confidence"].min(),
            "max_confidence": correct["confidence"].max(),
        },
        {
            "group": "incorrect_predictions",
            "count": len(incorrect),
            "mean_confidence": incorrect["confidence"].mean(),
            "median_confidence": incorrect["confidence"].median(),
            "min_confidence": incorrect["confidence"].min(),
            "max_confidence": incorrect["confidence"].max(),
        },
        {
            "group": "all_predictions",
            "count": len(df),
            "mean_confidence": df["confidence"].mean(),
            "median_confidence": df["confidence"].median(),
            "min_confidence": df["confidence"].min(),
            "max_confidence": df["confidence"].max(),
        },
    ]
)

summary.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)

print("\nSummary:")
print(summary.to_string(index=False))

mean_correct = correct["confidence"].mean()
mean_incorrect = incorrect["confidence"].mean()

gap = mean_correct - mean_incorrect

print(
    f"\nMean confidence gap "
    f"(correct - incorrect): {gap:.4f}"
)

print(
    f"Correct prediction mean confidence: "
    f"{mean_correct:.4f}"
)

print(
    f"Incorrect prediction mean confidence: "
    f"{mean_incorrect:.4f}"
)

print("\nInterpretation:")
if abs(gap) < 0.02:
    print(
        "Confidence shows weak separation between "
        "correct and incorrect predictions."
    )
elif gap > 0:
    print(
        "Confidence is higher on correct predictions, "
        "showing some separation."
    )
else:
    print(
        "Confidence is unexpectedly higher on incorrect "
        "predictions."
    )

print("\nSaved:")
print(OUTPUT_PATH)

print("=" * 80)
print("CONFIDENCE ANALYSIS COMPLETE")
print("=" * 80)