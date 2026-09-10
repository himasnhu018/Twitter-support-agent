import pandas as pd
from collections import defaultdict, deque

INPUT = "data/processed/amazonhelp_conversation_tweets.parquet"
GOLDEN = "data/golden/golden_set.csv"

OUTPUT_CORPUS = "data/processed/amazon_retrieval_corpus.parquet"
OUTPUT_STATS = "data/processed/leakage_split_stats.csv"


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def main():
    print("Loading conversation data...")
    df = pd.read_parquet(INPUT)

    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN)

    # Normalize IDs
    df["tweet_id_clean"] = df["tweet_id"].apply(normalize_id)
    df["parent_tweet_id_clean"] = df["in_response_to_tweet_id"].apply(
        normalize_id
    )

    golden_ids = set(
        golden["tweet_id"].apply(normalize_id)
    )

    print(f"Conversation tweets: {len(df):,}")
    print(f"Golden examples: {len(golden_ids):,}")

    # ---------------------------------------------------------
    # Build undirected graph:
    #
    # tweet <-> parent tweet
    #
    # Connected components represent conversation components.
    # ---------------------------------------------------------
    tweet_ids = set(df["tweet_id_clean"].dropna())

    graph = defaultdict(set)

    for _, row in df.iterrows():
        tweet_id = row["tweet_id_clean"]
        parent_id = row["parent_tweet_id_clean"]

        if tweet_id is None:
            continue

        graph[tweet_id]

        if parent_id is not None and parent_id in tweet_ids:
            graph[tweet_id].add(parent_id)
            graph[parent_id].add(tweet_id)

    # ---------------------------------------------------------
    # Find connected components
    # ---------------------------------------------------------

    component_id = {}
    current_component = 0

    for tweet_id in graph:
        if tweet_id in component_id:
            continue

        current_component += 1

        queue = deque([tweet_id])
        component_id[tweet_id] = current_component

        while queue:
            current = queue.popleft()

            for neighbor in graph[current]:
                if neighbor not in component_id:
                    component_id[neighbor] = current_component
                    queue.append(neighbor)

    df["component_id"] = df["tweet_id_clean"].map(component_id)

    print(f"Conversation components: {current_component:,}")

    # ---------------------------------------------------------
    # Identify components containing golden examples
    # ---------------------------------------------------------

    golden_components = set(
        df.loc[
            df["tweet_id_clean"].isin(golden_ids),
            "component_id"
        ].dropna()
    )

    print(f"Components containing golden examples: {len(golden_components):,}")

    # ---------------------------------------------------------
    # Remove entire golden components
    # ---------------------------------------------------------

    leakage_safe = df[
        ~df["component_id"].isin(golden_components)
    ].copy()

    print(f"Original rows: {len(df):,}")
    print(f"Removed rows: {len(df) - len(leakage_safe):,}")
    print(f"Remaining rows: {len(leakage_safe):,}")

    # ---------------------------------------------------------
    # Save corpus
    # ---------------------------------------------------------

    leakage_safe.to_parquet(
        OUTPUT_CORPUS,
        index=False
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    stats = pd.DataFrame([
        {
            "metric": "original_tweets",
            "value": len(df),
        },
        {
            "metric": "golden_examples",
            "value": len(golden_ids),
        },
        {
            "metric": "total_components",
            "value": current_component,
        },
        {
            "metric": "golden_components",
            "value": len(golden_components),
        },
        {
            "metric": "removed_tweets",
            "value": len(df) - len(leakage_safe),
        },
        {
            "metric": "retrieval_corpus_tweets",
            "value": len(leakage_safe),
        },
        {
            "metric": "retrieval_corpus_components",
            "value": leakage_safe["component_id"].nunique(),
        },
    ])

    stats.to_csv(
        OUTPUT_STATS,
        index=False
    )

    print()
    print("Saved:")
    print(f"  {OUTPUT_CORPUS}")
    print(f"  {OUTPUT_STATS}")


if __name__ == "__main__":
    main()