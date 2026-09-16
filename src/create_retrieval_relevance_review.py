import sys
import random

import pandas as pd

sys.path.insert(0, "src")

from intent_classifier import predict_intent
from query_vector_retrieval import retrieve_top_k


RANDOM_SEED = 42

ONE_OF_THREE_SAMPLE = 15
TWO_OF_THREE_SAMPLE = 15


def get_intent(text):
    result = predict_intent(text)
    return result["intent"], result["confidence"]


def get_alignment_count(query_intent, retrieved_examples):
    matching_count = 0

    retrieved_intents = []

    for result in retrieved_examples:
        intent, confidence = get_intent(
            result["historical_customer_message"]
        )

        retrieved_intents.append(
            {
                "intent": intent,
                "confidence": confidence,
            }
        )

        if intent == query_intent:
            matching_count += 1

    return matching_count, retrieved_intents


def build_case(
    case_id,
    query,
    query_intent,
    query_confidence,
    retrieved_examples,
    retrieved_intents,
    alignment_count,
):
    case = {
        "case_id": case_id,
        "query": query,
        "query_intent": query_intent,
        "query_confidence": query_confidence,
        "alignment_count": alignment_count,
    }

    for index, (result, intent_info) in enumerate(
        zip(retrieved_examples, retrieved_intents),
        start=1,
    ):
        case[f"retrieved_{index}_intent"] = intent_info["intent"]
        case[f"retrieved_{index}_confidence"] = (
            intent_info["confidence"]
        )

        case[f"retrieved_{index}_distance"] = result["distance"]

        case[f"retrieved_{index}_customer_message"] = (
            result["historical_customer_message"]
        )

        case[f"retrieved_{index}_response"] = (
            result["historical_gloCare_response"]
        )

        case[f"retrieved_{index}_tweet_id"] = (
            result["historical_customer_tweet_id"]
        )

        # Empty columns for human review.
        case[f"retrieved_{index}_relevance"] = ""
        case[f"retrieved_{index}_review_reason"] = ""

    case["overall_review"] = ""

    return case


def main():
    print("=" * 70)
    print("CREATE RETRIEVAL RELEVANCE REVIEW SAMPLE")
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

    one_of_three_cases = []
    two_of_three_cases = []

    for query_number, query in enumerate(
        queries,
        start=1,
    ):
        try:
            query_intent, query_confidence = get_intent(query)

            retrieved_examples = retrieve_top_k(
                query,
                top_k=3,
            )

            if len(retrieved_examples) != 3:
                continue

            alignment_count, retrieved_intents = (
                get_alignment_count(
                    query_intent,
                    retrieved_examples,
                )
            )

            if alignment_count not in [1, 2]:
                continue

            case_data = (
                query,
                query_intent,
                query_confidence,
                retrieved_examples,
                retrieved_intents,
                alignment_count,
            )

            if alignment_count == 1:
                one_of_three_cases.append(case_data)

            elif alignment_count == 2:
                two_of_three_cases.append(case_data)

        except Exception as exc:
            print(
                f"WARNING: failed query {query_number}: {exc}"
            )

        if query_number % 500 == 0:
            print(
                f"Processed "
                f"{query_number}/{len(queries)} queries..."
            )

    # ---------------------------------------------------------
    # Reproducible random sampling.
    # ---------------------------------------------------------

    random.seed(RANDOM_SEED)

    one_sample = random.sample(
        one_of_three_cases,
        min(
            ONE_OF_THREE_SAMPLE,
            len(one_of_three_cases),
        ),
    )

    two_sample = random.sample(
        two_of_three_cases,
        min(
            TWO_OF_THREE_SAMPLE,
            len(two_of_three_cases),
        ),
    )

    review_rows = []

    case_id = 1

    for case_data in one_sample:
        (
            query,
            query_intent,
            query_confidence,
            retrieved_examples,
            retrieved_intents,
            alignment_count,
        ) = case_data

        review_rows.append(
            build_case(
                case_id,
                query,
                query_intent,
                query_confidence,
                retrieved_examples,
                retrieved_intents,
                alignment_count,
            )
        )

        case_id += 1

    for case_data in two_sample:
        (
            query,
            query_intent,
            query_confidence,
            retrieved_examples,
            retrieved_intents,
            alignment_count,
        ) = case_data

        review_rows.append(
            build_case(
                case_id,
                query,
                query_intent,
                query_confidence,
                retrieved_examples,
                retrieved_intents,
                alignment_count,
            )
        )

        case_id += 1

    output_path = (
        "results/retrieval_relevance_review_sample.csv"
    )

    review_df = pd.DataFrame(review_rows)

    review_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAMPLE CREATED")
    print("=" * 70)

    print(
        f"1/3 alignment candidates found: "
        f"{len(one_of_three_cases)}"
    )

    print(
        f"2/3 alignment candidates found: "
        f"{len(two_of_three_cases)}"
    )

    print(
        f"1/3 cases sampled: "
        f"{len(one_sample)}"
    )

    print(
        f"2/3 cases sampled: "
        f"{len(two_sample)}"
    )

    print(
        f"Total review cases: "
        f"{len(review_rows)}"
    )

    print()
    print(f"Output file: {output_path}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()