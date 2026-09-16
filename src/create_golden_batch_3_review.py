from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_batch_3.csv")
OUTPUT_FILE = Path("results/golden_batch_3_review.txt")


def main():
    df = pd.read_csv(INPUT_FILE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write("HIVER GOLDEN SET - BATCH 3 HUMAN REVIEW\n")
        f.write("=" * 80 + "\n\n")

        f.write("Instructions:\n")
        f.write("For every example, decide:\n")
        f.write("1. golden_selected: yes / no\n")
        f.write("2. golden_intent: one of the 10 final taxonomy intents\n")
        f.write("3. confidence: high / medium / low\n")
        f.write(
            "4. case_type: routine / rare / ambiguous / "
            "low_information / escalation_relevant\n"
        )
        f.write("5. escalation_expected: yes / no / uncertain\n")
        f.write("6. annotation_notes: short reason\n")
        f.write("\n")
        f.write("IMPORTANT: candidate_group and signal_* fields are sampling aids only.\n")
        f.write("They are NOT ground-truth labels.\n")
        f.write("\n")
        f.write("=" * 80 + "\n\n")

        for _, row in df.iterrows():
            order = int(row["review_order"])

            f.write(f"EXAMPLE {order}\n")
            f.write("-" * 80 + "\n")

            f.write(
                f"Customer tweet ID: "
                f"{row['customer_tweet_id']}\n"
            )

            f.write(
                f"Customer message:\n"
                f"{row['customer_text']}\n\n"
            )

            f.write(
                f"Historical brand response:\n"
                f"{row['brand_response']}\n\n"
            )

            f.write("Sampling information:\n")
            f.write(
                f"  candidate_group: "
                f"{row['candidate_group']}\n"
            )
            f.write(
                f"  review_priority: "
                f"{row['review_priority']}\n"
            )
            f.write(
                f"  topic_signal_count: "
                f"{row['topic_signal_count']}\n"
            )
            f.write(
                f"  possible_escalation: "
                f"{row['signal_possible_escalation']}\n"
            )
            f.write(
                f"  short_message: "
                f"{row['signal_short_message']}\n"
            )
            f.write(
                f"  low_information_signal: "
                f"{row['signal_low_information']}\n"
            )
            f.write("\n")

            f.write("HUMAN ANNOTATION:\n")
            f.write("  golden_selected: \n")
            f.write("  golden_intent: \n")
            f.write("  confidence: \n")
            f.write("  case_type: \n")
            f.write("  escalation_expected: \n")
            f.write("  annotation_notes: \n")

            f.write("\n")
            f.write("=" * 80 + "\n\n")

    print("Golden Batch 3 review file created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Examples included: {len(df)}")


if __name__ == "__main__":
    main()