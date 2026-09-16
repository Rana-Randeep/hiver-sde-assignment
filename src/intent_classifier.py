from pathlib import Path

import joblib


MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")


# Load the already-trained TF-IDF + Logistic Regression pipeline.
model = joblib.load(MODEL_FILE)


def predict_intent(customer_message):
    """
    Predict the intent and confidence for one customer message.

    Args:
        customer_message: str
            Incoming customer support message.

    Returns:
        dict containing:
            intent: predicted intent
            confidence: probability of the predicted intent
    """

    predicted_intent = model.predict([customer_message])[0]

    

    probabilities = model.predict_proba([customer_message])[0]

    confidence = probabilities.max()

    return {
        "intent": predicted_intent,
        "confidence": float(confidence),
    }


if __name__ == "__main__":
    test_message = "my internet is not working"

    result = predict_intent(test_message)

    print("Customer message:", test_message)
    print("Predicted intent:", result["intent"])
    print("Confidence:", round(result["confidence"], 4))