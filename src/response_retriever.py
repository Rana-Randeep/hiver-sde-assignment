from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "glocare_pairs.csv"


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_text", "brand_response"]
).copy()

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

df["brand_response"] = (
    df["brand_response"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["customer_text"] != "")
    & (df["brand_response"] != "")
].copy()

df = df.reset_index(drop=True)


# ============================================================
# BUILD TF-IDF REPRESENTATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

customer_matrix = vectorizer.fit_transform(
    df["customer_text"]
)


# ============================================================
# TOP-1 RETRIEVAL
# ============================================================

def retrieve_top1(customer_message):
    """
    Retrieve the most similar historical GloCare
    customer-support interaction.
    """

    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        customer_matrix
    ).ravel()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_row = df.iloc[best_index]

    return {
        "query": customer_message,
        "similarity": float(best_score),
        "historical_customer_message": (
            best_row["customer_text"]
        ),
        "historical_gloCare_response": (
            best_row["brand_response"]
        ),
        "historical_customer_tweet_id": (
            best_row["customer_tweet_id"]
        ),
    }


# ============================================================
# SIMPLE MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_message = "my internet is not working"

    result = retrieve_top1(test_message)

    print("=" * 70)
    print("HISTORICAL RESPONSE RETRIEVAL TEST")
    print("=" * 70)

    print("\nCustomer message:")
    print(result["query"])

    print("\nSimilarity:")
    print(round(result["similarity"], 4))

    print("\nHistorical customer message:")
    print(result["historical_customer_message"])

    print("\nHistorical GloCare response:")
    print(result["historical_gloCare_response"])

    print("\nHistorical tweet ID:")
    print(result["historical_customer_tweet_id"])