import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_FILE = "results/vector_retrieval_review.csv"
MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")


# Load locked golden evaluation set.
df = pd.read_csv(GOLDEN_FILE)

# Load the already-trained classifier.
model = joblib.load(MODEL_FILE)

# Predict intents for all golden examples.
predicted_intents = model.predict(
    df["golden_customer_text"].tolist()
)

df["predicted_intent"] = predicted_intents

# Compare predictions against golden labels.
accuracy = accuracy_score(
    df["golden_intent"],
    df["predicted_intent"]
)

macro_f1 = f1_score(
    df["golden_intent"],
    df["predicted_intent"],
    average="macro",
    zero_division=0,
)

print("=" * 80)
print("INTENT CLASSIFIER — GOLDEN SET EVALUATION")
print("=" * 80)

print(f"\nGolden examples: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        df["golden_intent"],
        df["predicted_intent"],
        zero_division=0,
    )
)

print("\nPrediction distribution:")
print(
    df["predicted_intent"]
    .value_counts()
    .sort_index()
)

# Save predictions for later retrieval experiments.
OUTPUT_FILE = "results/golden_intent_predictions.csv"
df[
    [
        "customer_tweet_id",
        "golden_customer_text",
        "golden_intent",
        "predicted_intent",
    ]
].to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved predictions to: {OUTPUT_FILE}")
print("=" * 80)