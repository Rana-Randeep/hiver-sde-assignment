from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path("data/glocare_pairs.csv")
DB_PATH = Path("vector_db/chroma")
COLLECTION_NAME = "glocare_historical_responses"

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 128


def main():
    # -----------------------------
    # 1. Load historical data
    # -----------------------------

    print("Loading GloCare historical data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded rows: {len(df):,}")

    required_columns = [
        "customer_tweet_id",
        "customer_author_id",
        "customer_created_at",
        "customer_text",
        "brand_tweet_id",
        "brand_created_at",
        "brand_response",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove rows where either side of the interaction is missing.
    df = df.dropna(
        subset=["customer_text", "brand_response"]
    ).copy()

    print(f"Rows after removing missing text: {len(df):,}")

    # -----------------------------
    # 2. Load embedding model
    # -----------------------------

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # 3. Create persistent ChromaDB
    # -----------------------------

    print(f"Opening ChromaDB at: {DB_PATH}")

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Delete an existing collection so that rerunning this script
    # creates a clean index rather than duplicating records.
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        print(
            f"Deleting existing collection: {COLLECTION_NAME}"
        )
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "GloCare historical customer support interactions"
        },
    )

    print(f"Created collection: {COLLECTION_NAME}")

    # -----------------------------
    # 4. Prepare data
    # -----------------------------

    customer_texts = (
        df["customer_text"]
        .astype(str)
        .tolist()
    )

    print("Generating embeddings...")

    # -----------------------------
    # 5. Generate embeddings + index
    # -----------------------------

    total_rows = len(df)

    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)

        batch_df = df.iloc[start:end]

        batch_texts = (
            batch_df["customer_text"]
            .astype(str)
            .tolist()
        )

        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        ids = [
            f"interaction_{index}"
            for index in batch_df.index
        ]

        documents = batch_texts

        metadatas = []

        for _, row in batch_df.iterrows():
            metadatas.append(
                {
                    "customer_tweet_id": str(
                        row["customer_tweet_id"]
                    ),
                    "customer_author_id": str(
                        row["customer_author_id"]
                    ),
                    "customer_created_at": str(
                        row["customer_created_at"]
                    ),
                    "brand_tweet_id": str(
                        row["brand_tweet_id"]
                    ),
                    "brand_created_at": str(
                        row["brand_created_at"]
                    ),
                    "brand_response": str(
                        row["brand_response"]
                    ),
                }
            )

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"Indexed {end:,}/{total_rows:,} rows"
        )

    # -----------------------------
    # 6. Verify index
    # -----------------------------

    count = collection.count()

    print()
    print("Vector index build complete.")
    print(f"Expected rows: {total_rows:,}")
    print(f"Indexed rows:  {count:,}")

    if count != total_rows:
        raise RuntimeError(
            "Indexed row count does not match expected row count."
        )

    print("Index verification: PASSED")


if __name__ == "__main__":
    main()