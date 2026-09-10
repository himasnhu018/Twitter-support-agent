from pathlib import Path
import random

import pandas as pd


DATA_FILE = Path(
    "data/processed/amazonhelp_conversation_tweets.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/amazon_sample_conversations.txt"
)

BRAND = "AmazonHelp"

RANDOM_SEED = 42


def normalize_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def main():

    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("AMAZONHELP CONVERSATION SAMPLER")
    print("=" * 70)

    print("\nLoading data...")

    df = pd.read_parquet(DATA_FILE)

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

    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------
    # Build lookup
    # ---------------------------------------------------------

    tweets = {}

    for _, row in df.iterrows():
        tweets[row["tweet_id"]] = row

    # ---------------------------------------------------------
    # Parent -> children
    # ---------------------------------------------------------

    children = {}

    for _, row in df.iterrows():

        parent = row["parent_id"]

        if parent is None:
            continue

        children.setdefault(
            parent,
            []
        ).append(
            row["tweet_id"]
        )

    # ---------------------------------------------------------
    # Find roots
    # ---------------------------------------------------------

    roots = []

    for tweet_id, row in tweets.items():

        parent = row["parent_id"]

        if parent is None or parent not in tweets:
            roots.append(tweet_id)

    # ---------------------------------------------------------
    # Build threads
    # ---------------------------------------------------------

    conversations = []

    visited = set()

    for root_id in roots:

        if root_id in visited:
            continue

        queue = [root_id]
        thread_ids = set()

        while queue:

            current = queue.pop()

            if current in thread_ids:
                continue

            thread_ids.add(current)
            visited.add(current)

            for child in children.get(
                current,
                []
            ):
                queue.append(child)

        rows = [
            tweets[tweet_id]
            for tweet_id in thread_ids
        ]

        rows.sort(
            key=lambda x: x["created_at_parsed"]
        )

        # Only keep conversations containing
        # at least one customer and one brand message.

        has_brand = any(
            row["is_brand"]
            for row in rows
        )

        has_customer = any(
            not row["is_brand"]
            for row in rows
        )

        if not (has_brand and has_customer):
            continue

        conversations.append(
            {
                "root_id": root_id,
                "rows": rows,
                "length": len(rows),
            }
        )

    print(
        f"\nUsable conversations: "
        f"{len(conversations):,}"
    )

    # ---------------------------------------------------------
    # Sample by conversation length.
    # ---------------------------------------------------------

    buckets = {
        "2-3 turns": [],
        "4-5 turns": [],
        "6-8 turns": [],
        "9+ turns": [],
    }

    for conversation in conversations:

        length = conversation["length"]

        if 2 <= length <= 3:
            buckets["2-3 turns"].append(
                conversation
            )

        elif 4 <= length <= 5:
            buckets["4-5 turns"].append(
                conversation
            )

        elif 6 <= length <= 8:
            buckets["6-8 turns"].append(
                conversation
            )

        elif length >= 9:
            buckets["9+ turns"].append(
                conversation
            )

    # ---------------------------------------------------------
    # Sample 5 from each bucket.
    # ---------------------------------------------------------

    selected = []

    for bucket_name, bucket in buckets.items():

        sample_size = min(
            5,
            len(bucket)
        )

        samples = random.sample(
            bucket,
            sample_size
        )

        for conversation in samples:

            selected.append(
                (
                    bucket_name,
                    conversation
                )
            )

    # ---------------------------------------------------------
    # Write human-readable output.
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for index, (
            bucket_name,
            conversation,
        ) in enumerate(
            selected,
            start=1,
        ):

            f.write("\n")
            f.write("=" * 80)
            f.write("\n")

            f.write(
                f"CONVERSATION {index} "
                f"[{bucket_name}]\n"
            )

            f.write(
                f"Root ID: "
                f"{conversation['root_id']}\n"
            )

            f.write(
                f"Turns: "
                f"{conversation['length']}\n"
            )

            f.write("=" * 80)
            f.write("\n")

            for turn_number, row in enumerate(
                conversation["rows"],
                start=1,
            ):

                speaker = (
                    "AMAZON"
                    if row["is_brand"]
                    else "CUSTOMER"
                )

                f.write(
                    f"\n[{turn_number}] {speaker}\n"
                )

                f.write(
                    f"Tweet ID: "
                    f"{row['tweet_id']}\n"
                )

                f.write(
                    f"{row['text']}\n"
                )

    print(
        f"\nSample written to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nOpen this file and inspect the "
        "20 conversations."
    )


if __name__ == "__main__":
    main()