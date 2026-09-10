import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INTENTS = [
    "DELIVERY_DELAY",
    "DELIVERY_TRACKING",
    "DELIVERY_FAILURE",
    "ORDER_CHANGE",
    "RETURN",
    "REFUND",
    "PAYMENT",
    "PRIME",
    "TECHNICAL",
    "DIGITAL_CONTENT",
    "SELLER_FRAUD",
    "ORDER_PRODUCT",
    "SUPPORT_COMPLAINT",
    "OTHER",
]


def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


def load_data():
    golden_path = Path("data/golden/golden_set.csv")
    index_path = Path("data/processed/amazon_support.faiss")
    metadata_path = Path("data/processed/amazon_support_metadata.parquet")

    if not golden_path.exists():
        raise FileNotFoundError(f"Missing: {golden_path}")

    if not index_path.exists():
        raise FileNotFoundError(f"Missing: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing: {metadata_path}")

    golden = pd.read_csv(golden_path)
    metadata = pd.read_parquet(metadata_path)
    index = faiss.read_index(str(index_path))

    return golden, metadata, index


def evaluate(args):
    print("=" * 80)
    print("AMAZONHELP RETRIEVAL EVALUATION")
    print("=" * 80)

    golden, metadata, index = load_data()

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required_golden = {"tweet_id", "text", "gold_intent"}
    missing = required_golden - set(golden.columns)

    if missing:
        raise ValueError(
            f"Golden set is missing columns: {sorted(missing)}"
        )

    required_metadata = {
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_response",
    }

    missing = required_metadata - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing)}"
        )

    print(f"Golden examples: {len(golden)}")
    print(f"Retrieval corpus: {len(metadata)}")
    print(f"FAISS vectors:    {index.ntotal}")
    print(f"Top-k:            {args.k}")
    print()

    # ------------------------------------------------------------------
    # Load embedding model
    # ------------------------------------------------------------------
    print("Loading embedding model...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # Prepare queries
    # ------------------------------------------------------------------
    golden = golden.copy()

    golden["text"] = golden["text"].map(normalize_text)

    queries = golden["text"].tolist()

    print(f"Encoding {len(queries)} golden queries...")

    embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ------------------------------------------------------------------
    # Search FAISS
    # ------------------------------------------------------------------
    print()
    print("Searching FAISS...")

    scores, indices = index.search(
        embeddings.astype("float32"),
        args.k,
    )

    # ------------------------------------------------------------------
    # Build evaluation rows
    # ------------------------------------------------------------------
    results = []

    for i, row in golden.iterrows():

        query_id = str(row["tweet_id"])
        query_text = normalize_text(row["text"])
        gold_intent = str(row["gold_intent"])

        for rank in range(args.k):

            metadata_idx = int(indices[i][rank])

            if metadata_idx < 0:
                continue

            retrieved = metadata.iloc[metadata_idx]

            results.append(
                {
                    "query_index": i,
                    "query_tweet_id": query_id,
                    "gold_intent": gold_intent,
                    "query_text": query_text,
                    "rank": rank + 1,
                    "similarity": float(scores[i][rank]),
                    "retrieved_customer_tweet_id": str(
                        retrieved["customer_tweet_id"]
                    ),
                    "retrieved_agent_tweet_id": str(
                        retrieved["agent_tweet_id"]
                    ),
                    "retrieved_customer_text": normalize_text(
                        retrieved["customer_text"]
                    ),
                    "retrieved_agent_response": normalize_text(
                        retrieved["agent_response"]
                    ),
                }
            )

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Save raw retrieval results
    # ------------------------------------------------------------------
    output_path = Path(
        "data/golden/retrieval_evaluation_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Basic metrics
    # ------------------------------------------------------------------
    top1 = (
        results_df[results_df["rank"] == 1]
        .groupby("query_index")
        .first()
    )

    mean_top1_similarity = top1["similarity"].mean()

    mean_topk_similarity = (
        results_df.groupby("query_index")["similarity"]
        .mean()
        .mean()
    )

    print()
    print("=" * 80)
    print("RETRIEVAL METRICS")
    print("=" * 80)

    print(
        f"Mean top-1 cosine similarity: {mean_top1_similarity:.4f}"
    )

    print(
        f"Mean top-{args.k} similarity:    {mean_topk_similarity:.4f}"
    )

    # ------------------------------------------------------------------
    # Intent-aware retrieval analysis
    #
    # IMPORTANT:
    # The retrieval corpus does not have gold intent labels.
    # Therefore this is NOT a true retrieval accuracy metric.
    #
    # We use Gemini's evaluated intent only as a diagnostic:
    # does the retrieved case's customer message appear semantically
    # aligned with the golden intent?
    #
    # This is intentionally reported as a diagnostic rather than
    # ground-truth retrieval accuracy.
    # ------------------------------------------------------------------

    print()
    print("NOTE:")
    print(
        "This evaluation does not claim retrieval accuracy because "
        "historical support cases do not have gold intent labels."
    )

    # ------------------------------------------------------------------
    # Show qualitative examples
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("QUALITATIVE RETRIEVAL EXAMPLES")
    print("=" * 80)

    example_indices = np.linspace(
        0,
        len(golden) - 1,
        min(args.examples, len(golden)),
        dtype=int,
    )

    for example_number, query_idx in enumerate(example_indices, 1):

        query_row = golden.iloc[query_idx]

        print()
        print("-" * 80)
        print(f"Example {example_number}")
        print(f"Tweet ID:   {query_row['tweet_id']}")
        print(f"Gold intent: {query_row['gold_intent']}")
        print(f"Customer:   {normalize_text(query_row['text'])}")

        retrieved_rows = results_df[
            results_df["query_index"] == query_idx
        ].sort_values("rank")

        for _, retrieved in retrieved_rows.head(3).iterrows():

            print()
            print(
                f"  Rank {int(retrieved['rank'])} "
                f"(similarity={retrieved['similarity']:.4f})"
            )

            print(
                "  Historical customer: "
                + retrieved["retrieved_customer_text"]
            )

            print(
                "  Historical response:  "
                + retrieved["retrieved_agent_response"]
            )

    # ------------------------------------------------------------------
    # Similarity distribution
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 80)

    print(
        top1["similarity"]
        .describe()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Save compact top-1 file
    # ------------------------------------------------------------------
    top1_output = Path(
        "data/golden/retrieval_top1.csv"
    )

    top1.reset_index().to_csv(
        top1_output,
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(f"Full retrieval results: {output_path}")
    print(f"Top-1 results:          {top1_output}")
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of historical cases to retrieve",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=8,
        help="Number of qualitative examples to print",
    )

    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    evaluate(args)