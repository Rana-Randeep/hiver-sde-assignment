import pandas as pd
from collections import Counter

from response_retriever import retrieve_top1
from query_vector_retrieval import retrieve_top_k


DATA_PATH = "data/glocare_pairs.csv"
TOP_K = 3


def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


def main():
    df = pd.read_csv(DATA_PATH)

    # Use the same unique-query population used by the previous
    # retrieval diagnostics.
    queries = (
        df["customer_text"]
        .dropna()
        .map(normalize_text)
        .drop_duplicates()
        .tolist()
    )

    print("=" * 70)
    print("TF-IDF VS CHROMADB RETRIEVAL COMPARISON")
    print("=" * 70)

    print(f"\nUnique queries evaluated: {len(queries)}")

    rows = []
    errors = 0

    for i, query in enumerate(queries, start=1):

        try:
            # -------------------------------------------------
            # TF-IDF baseline
            # -------------------------------------------------
            tfidf_result = retrieve_top1(query)

            # retrieve_response() may return a dictionary.
            # We only need the retrieved historical customer
            # message for comparison.
            if isinstance(tfidf_result, dict):
                tfidf_message = normalize_text(
                    tfidf_result.get("historical_customer_message", "")
                )
            else:
                tfidf_message = normalize_text(tfidf_result)

            # -------------------------------------------------
            # ChromaDB
            # -------------------------------------------------
            chroma_results = retrieve_top_k(
                query,
                top_k=TOP_K,
            )

            chroma_messages = [
                normalize_text(
                    result.get("historical_customer_message", "")
                )
                for result in chroma_results
            ]

            chroma_ids = [
                normalize_text(
                    result.get("historical_customer_tweet_id", "")
                )
                for result in chroma_results
            ]

            # -------------------------------------------------
            # Compare whether TF-IDF's result appears in
            # ChromaDB's top-3.
            # -------------------------------------------------
            tfidf_in_chroma = tfidf_message in chroma_messages

            rows.append(
                {
                    "query": query,
                    "tfidf_result": tfidf_message,
                    "chroma_result_1": (
                        chroma_messages[0]
                        if len(chroma_messages) > 0
                        else ""
                    ),
                    "chroma_result_2": (
                        chroma_messages[1]
                        if len(chroma_messages) > 1
                        else ""
                    ),
                    "chroma_result_3": (
                        chroma_messages[2]
                        if len(chroma_messages) > 2
                        else ""
                    ),
                    "chroma_id_1": (
                        chroma_ids[0]
                        if len(chroma_ids) > 0
                        else ""
                    ),
                    "chroma_id_2": (
                        chroma_ids[1]
                        if len(chroma_ids) > 1
                        else ""
                    ),
                    "chroma_id_3": (
                        chroma_ids[2]
                        if len(chroma_ids) > 2
                        else ""
                    ),
                    "tfidf_in_chroma_top3": tfidf_in_chroma,
                }
            )

        except Exception as exc:
            errors += 1

            if errors <= 5:
                print(
                    f"\nError on query {i}: {query}\n"
                    f"{type(exc).__name__}: {exc}"
                )

    result_df = pd.DataFrame(rows)

    output_path = "results/retrieval_method_comparison.csv"
    result_df.to_csv(output_path, index=False)

    print("\n--- Evaluation Summary ---")

    print(f"Queries successfully evaluated: {len(result_df)}")
    print(f"Errors: {errors}")

    if len(result_df) > 0:
        overlap_count = int(
            result_df["tfidf_in_chroma_top3"].sum()
        )

        overlap_percentage = (
            overlap_count / len(result_df) * 100
        )

        print(
            f"TF-IDF result also present in ChromaDB top-3: "
            f"{overlap_count} ({overlap_percentage:.2f}%)"
        )

        disagreement_count = len(result_df) - overlap_count

        print(
            f"Retrieval disagreement: "
            f"{disagreement_count} "
            f"({disagreement_count / len(result_df) * 100:.2f}%)"
        )

    print(f"\nDetailed comparison saved to: {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()