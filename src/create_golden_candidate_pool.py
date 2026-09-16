from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = Path("data/glocare_pairs.csv")
PILOT_FILE = Path("results/pilot_candidates_for_labeling.csv")
OUTPUT_FILE = Path("results/golden_candidate_pool.csv")

RANDOM_STATE = 42
POOL_SIZE = 400


def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    df = pd.read_csv(INPUT_FILE)
    pilot = pd.read_csv(PILOT_FILE)

    df["customer_text_clean"] = clean_text(df["customer_text"])
    pilot["customer_text_clean"] = clean_text(pilot["customer_text"])

    # Remove rows already used in the pilot.
    pilot_ids = set(pilot["customer_tweet_id"].astype(str))

    remaining = df[
        ~df["customer_tweet_id"].astype(str).isin(pilot_ids)
    ].copy()

    print(f"Original working rows: {len(df)}")
    print(f"Pilot rows excluded: {len(pilot)}")
    print(f"Remaining rows: {len(remaining)}")
    print()

    # ---------------------------------------------------------
    # 1. Create a broad random candidate sample.
    # ---------------------------------------------------------
    random_sample = remaining.sample(
        n=min(200, len(remaining)),
        random_state=RANDOM_STATE
    ).copy()

    # ---------------------------------------------------------
    # 2. Add diverse examples using TF-IDF similarity.
    # ---------------------------------------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000
    )

    matrix = vectorizer.fit_transform(remaining["customer_text_clean"])

    # Select examples that are less similar to the already-selected
    # random sample. This gives us lexical diversity.
    selected_indices = set(random_sample.index)

    candidate_indices = [
        idx for idx in remaining.index
        if idx not in selected_indices
    ]

    selected_matrix = matrix[
        [remaining.index.get_loc(idx) for idx in selected_indices]
    ]

    candidate_matrix = matrix[
        [remaining.index.get_loc(idx) for idx in candidate_indices]
    ]

    similarities = cosine_similarity(
        candidate_matrix,
        selected_matrix
    )

    max_similarity = similarities.max(axis=1)

    diversity_df = pd.DataFrame({
        "index": candidate_indices,
        "max_similarity_to_selected": max_similarity
    })

    # Lower similarity = more lexically diverse.
    diversity_df = diversity_df.sort_values(
        "max_similarity_to_selected"
    )

    diverse_count = min(
        200,
        len(diversity_df)
    )

    diverse_indices = diversity_df.head(
        diverse_count
    )["index"].tolist()

    diverse_sample = remaining.loc[diverse_indices].copy()

    # ---------------------------------------------------------
    # 3. Combine random + diverse samples.
    # ---------------------------------------------------------
    pool = pd.concat(
        [random_sample, diverse_sample],
        ignore_index=True
    )

    pool = pool.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    # If larger than desired pool, sample down reproducibly.
    if len(pool) > POOL_SIZE:
        pool = pool.sample(
            n=POOL_SIZE,
            random_state=RANDOM_STATE
        )

    pool = pool.reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Remove helper column before saving.
    # ---------------------------------------------------------
    pool = pool.drop(
        columns=["customer_text_clean"],
        errors="ignore"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pool.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Golden candidate pool created.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Candidate pool size: {len(pool)}")
    print()

    print("Candidate pool columns:")
    print(list(pool.columns))

    print()
    print("First 10 candidate examples:")
    print(
        pool[
            ["customer_tweet_id", "customer_text"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()