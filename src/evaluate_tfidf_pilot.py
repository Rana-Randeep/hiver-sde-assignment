import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_FILE = Path("results/pilot_training_labels.csv")
MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")

METRICS_FILE = Path("results/tfidf_baseline_validation_metrics.json")
PREDICTIONS_FILE = Path("results/tfidf_baseline_validation_predictions.csv")


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

    # Load the same model configuration used for the baseline.
    # We refit it only on the training portion.
    model = joblib.load(MODEL_FILE)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

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

    predictions_df = pd.DataFrame(
        {
            "customer_text": X_validation.values,
            "true_intent": y_validation.values,
            "predicted_intent": predictions,
        }
    )

    metrics = {
        "dataset": str(DATA_FILE),
        "total_examples": int(len(df)),
        "training_examples": int(len(X_train)),
        "validation_examples": int(len(X_validation)),
        "test_size": 0.20,
        "random_state": 42,
        "stratified": True,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "note": (
            "This is pilot-set validation performance. "
            "The locked golden set was not used."
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

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("=== TF-IDF + Logistic Regression: Held-out Pilot Validation ===")
    print()
    print("Total examples:", len(df))
    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_validation))
    print("Random state:", 42)
    print("Stratified:", True)
    print()
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print()
    print("Classification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0,
        )
    )

    print("Saved metrics:", METRICS_FILE)
    print("Saved predictions:", PREDICTIONS_FILE)


if __name__ == "__main__":
    main()