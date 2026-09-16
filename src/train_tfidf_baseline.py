import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


TRAIN_FILE = Path("results/pilot_training_labels.csv")
MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")

METRICS_FILE = Path("results/tfidf_baseline_training_metrics.json")
REPORT_FILE = Path("results/tfidf_baseline_training_report.csv")
DISTRIBUTION_FILE = Path("results/tfidf_baseline_training_distribution.csv")


def main():
    df = pd.read_csv(TRAIN_FILE)

    X = df["customer_text"].fillna("")
    y = df["intent"]

    model_config = {
        "vectorizer": {
            "type": "TfidfVectorizer",
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_features": 10000,
        },
        "classifier": {
            "type": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
    }

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=10000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    report_dict = classification_report(
        y,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    distribution_df = (
        y.value_counts()
        .rename_axis("intent")
        .reset_index(name="count")
    )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "dataset": str(TRAIN_FILE),
        "num_examples": int(len(df)),
        "num_intents": int(y.nunique()),
        "training_accuracy": float(accuracy),
        "model": "TF-IDF + Logistic Regression",
        "configuration": model_config,
        "warning": (
            "This is training-set performance only. "
            "It must not be reported as held-out evaluation performance."
        ),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_df.to_csv(REPORT_FILE)

    distribution_df.to_csv(
        DISTRIBUTION_FILE,
        index=False,
    )

    print("=== TF-IDF + Logistic Regression Baseline ===")
    print()
    print("Training examples:", len(df))
    print("Number of intents:", y.nunique())
    print("Training accuracy:", round(accuracy, 4))
    print()
    print("IMPORTANT:")
    print("This is training-set performance only.")
    print("Do NOT use it as the final baseline result.")
    print()
    print("Artifacts saved:")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("Classification report:", REPORT_FILE)
    print("Label distribution:", DISTRIBUTION_FILE)


if __name__ == "__main__":
    main()