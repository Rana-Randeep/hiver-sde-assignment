import sys
from collections import Counter

import pandas as pd

# Allow imports from the src directory when running this file
# from the project root.
sys.path.insert(0, "src")

from query_vector_retrieval import retrieve_top_k


def classify_diversity(results):
    """
    Classify the top-k retrieval pattern using
    unique original customer tweet IDs.
    """

    tweet_ids = [
        str(result["historical_customer_tweet_id"])
        for result in results
    ]

    unique_count = len(set(tweet_ids))

    if unique_count == len(tweet_ids):
        return "all_distinct"

    if unique_count == 1:
        return "all_same"

    return "one_duplicate"


def main():
    print("=" * 70)
    print("CHROMADB RETRIEVAL DIVERSITY DIAGNOSTIC")
    print("=" * 70)

    df = pd.read_csv("data/glocare_pairs.csv")

    # Use unique customer messages as evaluation queries.
    queries = (
        df["customer_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"Unique customer queries: {len(queries)}")
    print()

    diversity_counts = Counter()

    total_results = 0
    total_distance = 0.0

    duplicate_customer_text_queries = 0
    repeated_customer_id_queries = 0

    processed_queries = 0
    failed_queries = 0

    for query_number, query in enumerate(queries, start=1):

        try:
            results = retrieve_top_k(
                query,
                top_k=3
            )
        except Exception as exc:
            failed_queries += 1

            print(
                f"WARNING: retrieval failed for query "
                f"{query_number}: {exc}"
            )

            continue

        if not results:
            continue

        processed_queries += 1
        total_results += len(results)

        # ---------------------------------------------------------
        # Distance
        # ---------------------------------------------------------

        distances = [
            float(result["distance"])
            for result in results
        ]

        total_distance += sum(distances)

        # ---------------------------------------------------------
        # Customer-text diversity
        # ---------------------------------------------------------

        customer_texts = [
            result["historical_customer_message"]
            for result in results
        ]

        unique_customer_texts = len(set(customer_texts))

        if unique_customer_texts < len(customer_texts):
            duplicate_customer_text_queries += 1

        # ---------------------------------------------------------
        # Original customer tweet-ID diversity
        # ---------------------------------------------------------

        tweet_ids = [
            str(result["historical_customer_tweet_id"])
            for result in results
        ]

        unique_tweet_ids = len(set(tweet_ids))

        if unique_tweet_ids < len(tweet_ids):
            repeated_customer_id_queries += 1

        diversity_pattern = classify_diversity(results)

        diversity_counts[diversity_pattern] += 1

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        if query_number % 500 == 0:
            print(
                f"Processed {query_number}/{len(queries)} queries..."
            )

    # -------------------------------------------------------------
    # Final statistics
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Queries in evaluation set: {len(queries)}")
    print(f"Queries successfully retrieved: {processed_queries}")
    print(f"Queries with retrieval errors: {failed_queries}")
    print(f"Retrieved results: {total_results}")

    if total_results > 0:
        average_distance = total_distance / total_results
        print(
            f"Average retrieval distance: "
            f"{average_distance:.4f}"
        )
    else:
        print("Average retrieval distance: N/A")

    print()
    print("TOP-3 DIVERSITY BY ORIGINAL CUSTOMER TWEET")
    print("-" * 70)

    for pattern in [
        "all_distinct",
        "one_duplicate",
        "all_same",
    ]:
        count = diversity_counts[pattern]

        percentage = (
            count / processed_queries * 100
            if processed_queries
            else 0
        )

        print(
            f"{pattern:15s}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("DUPLICATION SIGNALS")
    print("-" * 70)

    text_percentage = (
        duplicate_customer_text_queries
        / processed_queries
        * 100
        if processed_queries
        else 0
    )

    id_percentage = (
        repeated_customer_id_queries
        / processed_queries
        * 100
        if processed_queries
        else 0
    )

    print(
        "Queries with duplicate customer text in top-3: "
        f"{duplicate_customer_text_queries} "
        f"({text_percentage:.2f}%)"
    )

    print(
        "Queries with repeated customer tweet ID in top-3: "
        f"{repeated_customer_id_queries} "
        f"({id_percentage:.2f}%)"
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()