import pandas as pd


INPUT_PATH = "results/retrieval_relevance_review_sample.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("RETRIEVAL HUMAN REVIEW ANALYSIS")
    print("=" * 70)

    print(f"\nCases reviewed: {len(df)}")

    # ---------------------------------------------------------
    # 1. Overall human relevance distribution
    # ---------------------------------------------------------

    relevance_columns = [
        "retrieved_1_relevance",
        "retrieved_2_relevance",
        "retrieved_3_relevance",
    ]

    all_relevance = pd.concat(
        [df[col] for col in relevance_columns],
        ignore_index=True,
    )

    print("\n--- Overall Retrieved-Example Relevance ---")

    counts = all_relevance.value_counts()

    for label in ["relevant", "partial", "not_relevant"]:
        count = counts.get(label, 0)
        percentage = count / len(all_relevance) * 100
        print(f"{label:15s}: {count:3d} ({percentage:6.2f}%)")

    # ---------------------------------------------------------
    # 2. Overall case-level review
    # ---------------------------------------------------------

    print("\n--- Overall Case Review ---")

    case_counts = df["overall_review"].value_counts()

    for label in ["good", "mixed", "poor"]:
        count = case_counts.get(label, 0)
        percentage = count / len(df) * 100
        print(f"{label:15s}: {count:3d} ({percentage:6.2f}%)")

    # ---------------------------------------------------------
    # 3. Retrieval quality by rank
    # ---------------------------------------------------------

    print("\n--- Relevance By Retrieval Rank ---")

    for rank in [1, 2, 3]:
        column = f"retrieved_{rank}_relevance"

        print(f"\nRank {rank}:")

        rank_counts = df[column].value_counts()

        for label in ["relevant", "partial", "not_relevant"]:
            count = rank_counts.get(label, 0)
            percentage = count / len(df) * 100
            print(f"  {label:13s}: {count:3d} ({percentage:6.2f}%)")

    # ---------------------------------------------------------
    # 4. Cases containing not_relevant retrievals
    # ---------------------------------------------------------

    print("\n--- Cases With Not-Relevant Retrievals ---")

    not_relevant_mask = (
        (df["retrieved_1_relevance"] == "not_relevant")
        | (df["retrieved_2_relevance"] == "not_relevant")
        | (df["retrieved_3_relevance"] == "not_relevant")
    )

    not_relevant_cases = df[not_relevant_mask]

    print(f"Cases: {len(not_relevant_cases)}")

    if len(not_relevant_cases) > 0:
        print(
            not_relevant_cases[
                [
                    "case_id",
                    "customer_text",
                    "overall_review",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # 5. Save summary
    # ---------------------------------------------------------

    summary = {
        "cases_reviewed": len(df),
        "retrieved_examples_reviewed": len(all_relevance),
        "relevant_examples": int(
            (all_relevance == "relevant").sum()
        ),
        "partial_examples": int(
            (all_relevance == "partial").sum()
        ),
        "not_relevant_examples": int(
            (all_relevance == "not_relevant").sum()
        ),
        "good_cases": int(
            (df["overall_review"] == "good").sum()
        ),
        "mixed_cases": int(
            (df["overall_review"] == "mixed").sum()
        ),
        "poor_cases": int(
            (df["overall_review"] == "poor").sum()
        ),
        "cases_with_not_relevant": int(
            not_relevant_mask.sum()
        ),
    }

    output_path = "results/retrieval_human_review_summary.csv"

    pd.DataFrame([summary]).to_csv(
        output_path,
        index=False,
    )

    print("\n--- Summary Saved ---")
    print(output_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()