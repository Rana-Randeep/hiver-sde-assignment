from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "results"
    / "tfidf_retrieval_golden_review.csv"
)

DB_PATH = PROJECT_ROOT / "vector_db" / "chroma"

COLLECTION_NAME = "glocare_historical_responses"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "vector_retrieval_golden_review.csv"
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("Loading locked golden set...")

golden = pd.read_csv(GOLDEN_PATH)

print(f"Golden examples loaded: {len(golden):,}")

required_columns = [
    "customer_tweet_id",
    "golden_customer_text",
    "golden_intent",
    "relevance",
]

missing_columns = [
    column
    for column in required_columns
    if column not in golden.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required golden-set columns: {missing_columns}"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# LOAD CHROMADB
# ============================================================

print(f"Opening ChromaDB at: {DB_PATH}")

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Collection: {COLLECTION_NAME}")
print(f"Indexed records: {collection.count():,}")


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_top_k(
    customer_message,
    customer_tweet_id,
    top_k=TOP_K,
):

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        where={
            "customer_tweet_id": {
                "$ne": str(customer_tweet_id)
            }
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for rank in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][rank]

        matches.append(
            {
                "rank": rank + 1,
                "distance": float(
                    results["distances"][0][rank]
                ),
                "historical_customer_message": (
                    results["documents"][0][rank]
                ),
                "historical_gloCare_response": (
                    metadata["brand_response"]
                ),
                "historical_customer_tweet_id": (
                    metadata["customer_tweet_id"]
                ),
            }
        )

    return matches

# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("Running vector retrieval evaluation...")

records = []

for index, row in golden.iterrows():

    customer_message = str(
        row["golden_customer_text"]
    )

    matches = retrieve_top_k(
        customer_message,
        row["customer_tweet_id"],
    )

    record = row.to_dict()

    # --------------------------------------------------------
    # Store top-1 result
    # --------------------------------------------------------

    top1 = matches[0]

    record["vector_top1_distance"] = (
        top1["distance"]
    )

    record["vector_top1_customer_text"] = (
        top1["historical_customer_message"]
    )

    record["vector_top1_response"] = (
        top1["historical_gloCare_response"]
    )

    record["vector_top1_tweet_id"] = (
        top1["historical_customer_tweet_id"]
    )

    # --------------------------------------------------------
    # Store top-k results
    # --------------------------------------------------------

    for match in matches:

        rank = match["rank"]

        record[
            f"vector_rank{rank}_distance"
        ] = match["distance"]

        record[
            f"vector_rank{rank}_customer_text"
        ] = match[
            "historical_customer_message"
        ]

        record[
            f"vector_rank{rank}_response"
        ] = match[
            "historical_gloCare_response"
        ]

        record[
            f"vector_rank{rank}_tweet_id"
        ] = match[
            "historical_customer_tweet_id"
        ]

    records.append(record)

    if (index + 1) % 25 == 0:
        print(
            f"Evaluated {index + 1:,}/{len(golden):,}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(records)

results.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VECTOR RETRIEVAL EVALUATION COMPLETE")
print("=" * 70)

print(f"Examples evaluated: {len(results):,}")

print()
print("Saved results to:")
print(OUTPUT_PATH)

print()
print("Original golden relevance labels preserved.")
print("No relevance labels were changed or inferred.")