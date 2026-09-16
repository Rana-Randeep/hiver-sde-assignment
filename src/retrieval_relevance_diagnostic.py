import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, "src")

from intent_classifier import predict_intent
from query_vector_retrieval import retrieve_top_k


def get_intent(text):
    """
    Return the predicted intent from the existing classifier.
    """
    result = predict_intent(text)

    return result["intent"]


def main():
    print("=" * 70)
    print("RETRIEVAL-INTENT ALIGNMENT DIAGNOSTIC")
    print("=" * 70)

    df = pd.read_csv("data/glocare_pairs.csv")

    queries = (
        df["customer_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"Unique customer queries: {len(queries)}")
    print()

    total_queries = 0
    total_retrieved_examples = 0

    exact_match_counts = Counter()

    alignment_distribution = Counter()

    failed_queries = 0

    # ---------------------------------------------------------
    # Evaluate every unique customer query.
    # ---------------------------------------------------------

    for query_number, query in enumerate(queries, start=1):

        try:
            query_result = predict_intent(query)

            query_intent = query_result["intent"]

            retrieved_examples = retrieve_top_k(
                query,
                top_k=3
            )

            if not retrieved_examples:
                continue

            total_queries += 1
            total_retrieved_examples += len(retrieved_examples)

            matching_examples = 0

            for result in retrieved_examples:

                retrieved_intent = get_intent(
                    result["historical_customer_message"]
                )

                if retrieved_intent == query_intent:
                    matching_examples += 1

            # -------------------------------------------------
            # How many of top-3 agree with query intent?
            # -------------------------------------------------

            alignment_distribution[matching_examples] += 1

            exact_match_counts[query_intent] += matching_examples

        except Exception as exc:
            failed_queries += 1

            print(
                f"WARNING: failed query {query_number}: {exc}"
            )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if query_number % 500 == 0:
            print(
                f"Processed {query_number}/{len(queries)} queries..."
            )

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Queries evaluated: {total_queries}")
    print(f"Queries with errors: {failed_queries}")
    print(
        f"Retrieved examples evaluated: "
        f"{total_retrieved_examples}"
    )

    print()
    print("TOP-3 INTENT ALIGNMENT")
    print("-" * 70)

    for matching_count in [0, 1, 2, 3]:

        count = alignment_distribution[matching_count]

        percentage = (
            count / total_queries * 100
            if total_queries
            else 0
        )

        print(
            f"{matching_count}/3 matching: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Overall retrieved-example alignment
    # ---------------------------------------------------------

    total_matching = sum(
        count * matching_count
        for matching_count, count
        in alignment_distribution.items()
    )

    total_possible = total_queries * 3

    alignment_rate = (
        total_matching / total_possible * 100
        if total_possible
        else 0
    )

    print()
    print("OVERALL ALIGNMENT")
    print("-" * 70)

    print(
        f"Retrieved examples matching query intent: "
        f"{total_matching}/{total_possible}"
    )

    print(
        f"Intent alignment rate: "
        f"{alignment_rate:.2f}%"
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()