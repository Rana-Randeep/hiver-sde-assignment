from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    test_texts = [
        "My internet connection is not working",
        "I cannot access mobile data",
        "Thank you for your help",
    ]

    embeddings = model.encode(test_texts)

    print(f"Number of texts: {len(test_texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")


if __name__ == "__main__":
    main()