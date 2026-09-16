import pandas as pd


INPUT_PATH = "results/retrieval_disagreement_review_sample.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("RETRIEVAL DISAGREEMENT ANALYSIS")
    print("=" * 70)

    print(f"\nCases reviewed: {len(df)}")

    counts = df["human_preference"].value_counts()

    print("\n--- Retrieval Preference ---")

    for label in [
        "better_chroma",
        "better_tfidf",
        "similar",
        "unclear",
    ]:
        count = counts.get(label, 0)
        percentage = count / len(df) * 100

        print(
            f"{label:15s}: "
            f"{count:3d} ({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Directional comparison
    # ---------------------------------------------------------

    better_chroma = int(
        (df["human_preference"] == "better_chroma").sum()
    )

    better_tfidf = int(
        (df["human_preference"] == "better_tfidf").sum()
    )

    similar = int(
        (df["human_preference"] == "similar").sum()
    )

    print("\n--- Directional Result ---")

    print(f"ChromaDB preferred: {better_chroma}")
    print(f"TF-IDF preferred:   {better_tfidf}")
    print(f"Similar:            {similar}")

    if better_chroma > better_tfidf:
        conclusion = (
            "ChromaDB was preferred more often in the reviewed "
            "disagreement cases."
        )
    elif better_tfidf > better_chroma:
        conclusion = (
            "TF-IDF was preferred more often in the reviewed "
            "disagreement cases."
        )
    else:
        conclusion = (
            "Neither method was preferred more often in the "
            "reviewed disagreement cases."
        )

    print(f"\nConclusion: {conclusion}")

    print("\n--- Review Reasons ---")

    for _, row in df.iterrows():
        print(
            f"\nQuery: {row['query']}"
        )
        print(
            f"Preference: {row['human_preference']}"
        )
        print(
            f"Reason: {row['review_reason']}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()