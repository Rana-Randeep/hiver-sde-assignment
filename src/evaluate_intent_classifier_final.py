from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


GOLDEN_FILE = Path("results/golden_set.csv")
MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")

METRICS_FILE = Path("results/tfidf_baseline_golden_metrics.json")
PREDICTIONS_FILE = Path("results/tfidf_baseline_golden_predictions.csv")
CONFUSION_MATRIX_FILE = Path("results/tfidf_baseline_golden_confusion_matrix.csv")


# ---------------------------------------------------------
# 1. Load the selected golden set
# ---------------------------------------------------------

df = pd.read_csv(GOLDEN_FILE)

df = df[df["golden_selected"].eq("yes")].copy()

print(f"Selected golden examples: {len(df)}")

if df.empty:
    raise ValueError("No golden examples have golden_selected == 'yes'.")


# ---------------------------------------------------------
# 2. Load the already-trained classifier
# ---------------------------------------------------------

model = joblib.load(MODEL_FILE)


# ---------------------------------------------------------
# 3. Predict intents
# ---------------------------------------------------------

X = df["customer_text"]
y_true = df["golden_intent"]

y_pred = model.predict(X)


# ---------------------------------------------------------
# 4. Calculate headline metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ---------------------------------------------------------
# 5. Per-class metrics
# ---------------------------------------------------------

labels = sorted(
    set(y_true) | set(y_pred)
)

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0,
)


# ---------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

cm_df.index.name = "actual_intent"

cm_df.to_csv(CONFUSION_MATRIX_FILE)


# ---------------------------------------------------------
# 7. Save predictions
# ---------------------------------------------------------

prediction_df = df[
    [
        "golden_id",
        "customer_tweet_id",
        "customer_text",
        "golden_intent",
    ]
].copy()

prediction_df["predicted_intent"] = y_pred
prediction_df["correct"] = (
    prediction_df["golden_intent"]
    == prediction_df["predicted_intent"]
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ---------------------------------------------------------
# 8. Save metrics
# ---------------------------------------------------------

metrics = {
    "evaluation_set": "selected_golden_set",
    "golden_examples": len(df),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "classification_report": report,
    "labels": labels,
}

import json

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


# ---------------------------------------------------------
# 9. Print results
# ---------------------------------------------------------

print()
print("=== Final Intent Classifier Evaluation ===")
print(f"Golden examples: {len(df)}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print()
print("=== Per-Class Metrics ===")

for label in labels:
    row = report[label]

    print(
        f"{label}: "
        f"precision={row['precision']:.4f}, "
        f"recall={row['recall']:.4f}, "
        f"f1={row['f1-score']:.4f}, "
        f"support={int(row['support'])}"
    )

print()
print("=== Confusion Matrix ===")
print(cm_df)

print()
print(f"Saved metrics:          {METRICS_FILE}")
print(f"Saved predictions:      {PREDICTIONS_FILE}")
print(f"Saved confusion matrix: {CONFUSION_MATRIX_FILE}")