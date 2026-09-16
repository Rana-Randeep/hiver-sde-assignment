from intent_classifier import predict_intent


TEST_MESSAGES = [
    "there is no network coverage",
    "my mobile data is not working",
    "my recharge failed",
    "my sim card is not detected",
    "where is my bonus",
]


def main():
    print("Intent classifier sanity check")
    print("=" * 70)

    for message in TEST_MESSAGES:
        result = predict_intent(message)

        print(f"Message:    {message}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 70)


if __name__ == "__main__":
    main()