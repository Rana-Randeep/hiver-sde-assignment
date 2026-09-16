import pandas as pd

from intent_classifier import predict_intent


DATA_PATH = "data/glocare_pairs.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # One unique customer message = one smoke-test candidate
    df = df.drop_duplicates(subset=["customer_text"]).copy()

    print(f"Unique customer messages available: {len(df)}")

    predictions = []

    print("\nRunning classifier on customer messages...")

    for _, row in df.iterrows():
        result = predict_intent(row["customer_text"])

        predictions.append(
            {
                "customer_text": row["customer_text"],
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
            }
        )

    predicted_df = pd.DataFrame(predictions)

    # Highest-confidence examples first
    predicted_df = predicted_df.sort_values(
        "confidence",
        ascending=False
    )

    # Select at most one message per predicted intent
    selected = (
        predicted_df
        .drop_duplicates(subset=["predicted_intent"])
        .head(5)
        .reset_index(drop=True)
    )

    print("\nSelected smoke-test candidates:\n")

    for i, row in selected.iterrows():
        print(f"--- Candidate {i + 1} ---")
        print(f"Customer: {row['customer_text']}")
        print(f"Predicted intent: {row['predicted_intent']}")
        print(f"Confidence: {row['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()