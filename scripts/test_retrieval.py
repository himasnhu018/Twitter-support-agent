import argparse

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/processed/amazon_support.faiss"
METADATA_PATH = "data/processed/amazon_support_metadata.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Test semantic retrieval against Amazon support history."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Customer message to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of historical cases to retrieve.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("AMAZON SUPPORT RETRIEVAL TEST")
    print("=" * 100)

    print("\nLoading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")
    metadata = pd.read_parquet(METADATA_PATH)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # Embed incoming customer message
    # ---------------------------------------------------------

    query_embedding = model.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        args.top_k,
    )

    print()
    print("CUSTOMER QUERY")
    print("-" * 100)
    print(args.query)

    print()
    print(f"TOP {args.top_k} HISTORICAL CASES")
    print("=" * 100)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata.iloc[idx]

        print()
        print(f"#{rank}  Similarity: {score:.4f}")
        print("-" * 100)

        print("HISTORICAL CUSTOMER:")
        print(row["customer_text"])

        print()
        print("HISTORICAL AMAZONHELP RESPONSE:")
        print(row["agent_response"])

        print()
        print(f"Conversation: {row['component_id']}")
        print(
            f"Customer tweet: {row['customer_tweet_id']} | "
            f"Agent tweet: {row['agent_tweet_id']}"
        )

        print("-" * 100)


if __name__ == "__main__":
    main()