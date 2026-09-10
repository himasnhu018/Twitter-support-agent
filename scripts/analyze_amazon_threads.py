from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


DATA_FILE = Path(
    "data/processed/amazonhelp_conversation_tweets.parquet"
)

BRAND = "AmazonHelp"


def normalize_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def main():

    print("=" * 70)
    print("AMAZONHELP THREAD ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")

    df = pd.read_parquet(DATA_FILE)

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    df["tweet_id"] = df["tweet_id_clean"].map(
        normalize_id
    )

    df["parent_id"] = df["parent_tweet_id"].map(
        normalize_id
    )

    df["author"] = (
        df["author_id"]
        .astype(str)
        .str.strip()
    )

    df["is_brand"] = df["author"].eq(BRAND)

    df["is_customer"] = ~df["is_brand"]

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    print(f"\nTotal tweets:       {len(df):,}")
    print(
        f"AmazonHelp tweets:  "
        f"{df['is_brand'].sum():,}"
    )
    print(
        f"Customer tweets:    "
        f"{df['is_customer'].sum():,}"
    )

    # ---------------------------------------------------------
    # Build parent -> children graph
    # ---------------------------------------------------------

    children = defaultdict(list)

    for _, row in df[
        ["tweet_id", "parent_id"]
    ].dropna().iterrows():

        children[row["parent_id"]].append(
            row["tweet_id"]
        )

    # ---------------------------------------------------------
    # Build tweet lookup
    # ---------------------------------------------------------

    tweets = {}

    for _, row in df.iterrows():

        tweets[row["tweet_id"]] = row

    # ---------------------------------------------------------
    # Find root tweets
    #
    # A root is a tweet whose parent is missing from
    # our extracted dataset.
    # ---------------------------------------------------------

    roots = []

    for tweet_id, row in tweets.items():

        parent_id = row["parent_id"]

        if parent_id is None or parent_id not in tweets:
            roots.append(tweet_id)

    print(
        f"\nRoot tweets:        "
        f"{len(roots):,}"
    )

    # ---------------------------------------------------------
    # Traverse each tree.
    # ---------------------------------------------------------

    visited_global = set()

    conversation_stats = []

    for root_id in roots:

        if root_id in visited_global:
            continue

        queue = [root_id]
        visited = set()

        while queue:

            current = queue.pop()

            if current in visited:
                continue

            visited.add(current)
            visited_global.add(current)

            for child in children.get(current, []):
                queue.append(child)

        if not visited:
            continue

        rows = [
            tweets[tweet_id]
            for tweet_id in visited
            if tweet_id in tweets
        ]

        turns = len(rows)

        brand_turns = sum(
            row["is_brand"]
            for row in rows
        )

        customer_turns = turns - brand_turns

        conversation_stats.append(
            {
                "root_id": root_id,
                "turns": turns,
                "brand_turns": brand_turns,
                "customer_turns": customer_turns,
            }
        )

    stats = pd.DataFrame(conversation_stats)

    # ---------------------------------------------------------
    # Conversation statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONVERSATION STATISTICS")
    print("=" * 70)

    print(
        f"\nConversation components: "
        f"{len(stats):,}"
    )

    if len(stats) > 0:

        print(
            f"Average tweets/conversation: "
            f"{stats['turns'].mean():.2f}"
        )

        print(
            f"Median tweets/conversation:  "
            f"{stats['turns'].median():.0f}"
        )

        print(
            f"Max tweets/conversation:      "
            f"{stats['turns'].max():,}"
        )

        # -----------------------------------------------------
        # Depth distribution
        # -----------------------------------------------------

        print("\nConversation depth:")
        print("-" * 50)

        depth_counts = Counter(
            stats["turns"]
        )

        for depth in sorted(
            depth_counts
        ):

            if depth <= 10:

                print(
                    f"{depth:>2} tweet(s): "
                    f"{depth_counts[depth]:>8,}"
                )

        greater_than_10 = (
            stats["turns"] > 10
        ).sum()

        if greater_than_10:

            print(
                f">10 tweets: "
                f"{greater_than_10:,}"
            )

        # -----------------------------------------------------
        # Useful support conversations
        # -----------------------------------------------------

        multi_turn = stats[
            stats["turns"] >= 3
        ]

        print(
            f"\n3+ turn conversations: "
            f"{len(multi_turn):,}"
        )

        print(
            f"5+ turn conversations: "
            f"{(stats['turns'] >= 5).sum():,}"
        )

        print(
            f"8+ turn conversations: "
            f"{(stats['turns'] >= 8).sum():,}"
        )

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------

    output = Path(
        "data/processed/"
        "amazon_conversation_statistics.csv"
    )

    stats.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved statistics to:\n{output}"
    )


if __name__ == "__main__":
    main()