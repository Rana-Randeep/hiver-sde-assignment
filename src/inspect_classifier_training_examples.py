from pathlib import Path

import pandas as pd


TRAIN_FILE = Path("results/pilot_training_labels.csv")


def show_examples(df, intent, n=10):
    examples = (
        df[df["intent"] == intent][["customer_text", "intent"]]
        .drop_duplicates()
        .head(n)
    )

    print(f"\nExamples for: {intent}")
    print("-" * 70)

    if examples.empty:
        print("No examples found.")
        return

    for i, row in examples.iterrows():
        print(f"- {row['customer_text']}")


def main():
    df = pd.read_csv(TRAIN_FILE)

    print("Training data shape:", df.shape)

    intents_to_inspect = [
        "unauthorized_charge_or_vas",
        "data_issue",
        "voice_or_line_issue",
        "bonus_or_promotion_issue",
    ]

    for intent in intents_to_inspect:
        show_examples(df, intent)


if __name__ == "__main__":
    main()