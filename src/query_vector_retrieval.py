from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "vector_db" / "chroma"
COLLECTION_NAME = "glocare_historical_responses"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# LOAD VECTOR INDEX
# ============================================================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

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

def retrieve_top_k(customer_message, top_k=TOP_K):
    """
    Retrieve the top-k semantically similar historical
    GloCare customer-support interactions.

    A larger candidate pool is retrieved first so that
    multiple replies belonging to the same customer tweet
    do not consume multiple top-k evidence slots.
    """

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True,
    )[0]

    # Retrieve extra candidates so we can enforce diversity.
    candidate_k = max(top_k * 5, top_k)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    seen_customer_tweet_ids = set()

    for index in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][index]

        customer_tweet_id = metadata["customer_tweet_id"]

        # Skip additional replies belonging to the same
        # historical customer message/thread.
        if customer_tweet_id in seen_customer_tweet_ids:
            continue

        seen_customer_tweet_ids.add(customer_tweet_id)

        matches.append(
            {
                "rank": len(matches) + 1,
                "distance": float(
                    results["distances"][0][index]
                ),
                "historical_customer_message": (
                    results["documents"][0][index]
                ),
                "historical_gloCare_response": (
                    metadata["brand_response"]
                ),
                "historical_customer_tweet_id": (
                    customer_tweet_id
                ),
            }
        )

        if len(matches) >= top_k:
            break

    return matches


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_messages = [
        "my internet is not working",
        "I cannot use my mobile data",
        "I was charged for something I did not subscribe to",
    ]

    for message in test_messages:

        print()
        print("=" * 80)
        print("CUSTOMER MESSAGE")
        print("=" * 80)
        print(message)

        matches = retrieve_top_k(message)

        for match in matches:

            print()
            print(f"--- Rank {match['rank']} ---")
            print(f"Distance: {match['distance']:.4f}")

            print("\nHistorical customer message:")
            print(match["historical_customer_message"])

            print("\nHistorical GloCare response:")
            print(match["historical_gloCare_response"])

            print("\nHistorical customer tweet ID:")
            print(match["historical_customer_tweet_id"])