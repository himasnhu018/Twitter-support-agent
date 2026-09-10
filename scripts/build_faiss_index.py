import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT = "data/processed/amazon_support_cases.parquet"
INDEX_OUTPUT = "data/processed/amazon_support.faiss"
METADATA_OUTPUT = "data/processed/amazon_support_metadata.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def main():
    print("=" * 80)
    print("BUILDING AMAZON SUPPORT FAISS INDEX")
    print("=" * 80)

    print("\nLoading support cases...")
    df = pd.read_parquet(INPUT)

    df = df.reset_index(drop=True)

    print(f"Support cases: {len(df):,}")

    texts = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("\nEncoding customer messages...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    print(f"\nEmbedding shape: {embeddings.shape}")

    # ---------------------------------------------------------
    # FAISS
    #
    # Since embeddings are normalized:
    # inner product == cosine similarity
    # ---------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print("\nAdding embeddings to FAISS...")
    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal:,}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(INDEX_OUTPUT),
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_OUTPUT
    )

    # Metadata rows must remain in exactly the same order
    # as vectors in the FAISS index.
    metadata = df[
        [
            "customer_tweet_id",
            "agent_tweet_id",
            "customer_text",
            "agent_response",
            "component_id",
            "customer_created_at",
            "agent_created_at",
        ]
    ].copy()

    metadata.to_parquet(
        METADATA_OUTPUT,
        index=False
    )

    print("\n" + "=" * 80)
    print("INDEX COMPLETE")
    print("=" * 80)

    print(f"Vectors:  {index.ntotal:,}")
    print(f"Dimension: {dimension}")
    print(f"Index:    {INDEX_OUTPUT}")
    print(f"Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()