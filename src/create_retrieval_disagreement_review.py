import pandas as pd


INPUT_PATH = "results/retrieval_method_comparison.csv"
OUTPUT_PATH = "results/retrieval_disagreement_review_sample.csv"

SAMPLE_SIZE = 30
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    disagreements = df[
        df["tfidf_in_chroma_top3"] == False
    ].copy()

    print("=" * 70)
    print("RETRIEVAL DISAGREEMENT REVIEW SAMPLE")
    print("=" * 70)

    print(f"\nTotal comparison cases: {len(df)}")
    print(f"Disagreement cases: {len(disagreements)}")

    sample_size = min(
        SAMPLE_SIZE,
        len(disagreements)
    )

    sample = disagreements.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    ).copy()

    # Add human-review columns.
    sample["human_preference"] = ""
    sample["review_reason"] = ""

    # Keep the most useful columns together.
    columns = [
        "query",

        "tfidf_result",

        "chroma_result_1",
        "chroma_result_2",
        "chroma_result_3",

        "chroma_id_1",
        "chroma_id_2",
        "chroma_id_3",

        "human_preference",
        "review_reason",
    ]

    sample = sample[columns]

    sample.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Sample size: {len(sample)}")
    print(f"Random seed: {RANDOM_SEED}")

    print("\nReview categories:")
    print("  better_tfidf  = TF-IDF retrieval is more useful")
    print("  better_chroma = ChromaDB retrieval is more useful")
    print("  similar       = both are approximately equally useful")
    print("  unclear       = insufficient context to decide")

    print(f"\nOutput saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()