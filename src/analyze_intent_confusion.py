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
    / "intent_confusion_pairs.csv"
)

df = pd.read_csv(INPUT_PATH)

errors = df[
    df["golden_intent"] != df["predicted_intent"]
].copy()

confusion = (
    errors
    .groupby(
        ["golden_intent", "predicted_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)

confusion["error_rate"] = (
    confusion["count"] / len(df)
).round(4)

print("=" * 80)
print("INTENT CONFUSION ANALYSIS")
print("=" * 80)

print(f"\nTotal golden examples: {len(df)}")
print(f"Total intent errors: {len(errors)}")
print(
    f"Intent error rate: "
    f"{len(errors) / len(df):.2%}"
)

print("\nTop confusion pairs:")
print(
    confusion.head(15).to_string(index=False)
)

confusion.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)