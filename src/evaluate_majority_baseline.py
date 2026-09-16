import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_FILE = Path("results/pilot_training_labels.csv")

METRICS_FILE = Path(
    "results/majority_baseline_validation_metrics.json"
)


def main():
    df = pd.read_csv(DATA_FILE)

    X = df["customer_text"].fillna("")
    y = df["intent"]

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    majority_intent = y_train.value_counts().idxmax()

    predictions = [majority_intent] * len(y_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_validation,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "dataset": str(DATA_FILE),
        "total_examples": int(len(df)),
        "training_examples": int(len(X_train)),
        "validation_examples": int(len(X_validation)),
        "random_state": 42,
        "stratified": True,
        "baseline": "majority_class",
        "majority_intent": majority_intent,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "note": (
            "Trivial majority-class baseline on the same "
            "pilot validation split."
        ),
    }

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("=== Majority-Class Baseline ===")
    print()
    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_validation))
    print("Majority intent:", majority_intent)
    print()
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print()
    print("Saved metrics:", METRICS_FILE)


if __name__ == "__main__":
    main()