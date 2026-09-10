from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed")

BRAND = "AmazonHelp"

CHUNK_SIZE = 100_000


def normalize_id(value):
    """Convert tweet IDs to a consistent string representation."""

    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def main():

    print("=" * 70)
    print("AMAZONHELP DATA EXTRACTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # PASS 1
    #
    # Find all AmazonHelp tweet IDs.
    # ---------------------------------------------------------

    brand_tweet_ids = set()

    print("\nPASS 1: Finding AmazonHelp tweets...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        print(f"  Chunk {chunk_number}...")

        df["tweet_id"] = df["tweet_id"].map(
            normalize_id
        )

        mask = (
            df["author_id"]
            .astype(str)
            .str.strip()
            .eq(BRAND)
        )

        brand_tweet_ids.update(
            df.loc[mask, "tweet_id"].dropna()
        )

    print(
        f"\nFound {len(brand_tweet_ids):,} "
        f"{BRAND} tweets."
    )

    # ---------------------------------------------------------
    # PASS 2
    #
    # Extract:
    #
    # 1. Amazon tweets
    # 2. Direct customer replies to Amazon tweets
    #
    # We'll use these as our initial interaction corpus.
    # ---------------------------------------------------------

    print("\nPASS 2: Extracting related tweets...")

    rows = []

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        print(f"  Chunk {chunk_number}...")

        df["tweet_id_norm"] = df["tweet_id"].map(
            normalize_id
        )

        df["parent_id_norm"] = (
            df["in_response_to_tweet_id"]
            .map(normalize_id)
        )

        # Amazon tweets.
        brand_rows = df[
            df["author_id"]
            .astype(str)
            .str.strip()
            .eq(BRAND)
        ]

        # Customer tweets directly replying to Amazon.
        customer_rows = df[
            (
                df["inbound"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )
            &
            (
                df["parent_id_norm"]
                .isin(brand_tweet_ids)
            )
        ]

        selected = pd.concat(
            [
                brand_rows,
                customer_rows,
            ],
            ignore_index=True,
        )

        rows.append(selected)

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    # Remove duplicates.
    result = result.drop_duplicates(
        subset=["tweet_id_norm"]
    )

    # Rename normalized IDs.
    result = result.rename(
        columns={
            "tweet_id_norm": "tweet_id_clean",
            "parent_id_norm": "parent_tweet_id",
        }
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "amazonhelp_tweets.parquet"
    )

    result.to_parquet(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nExtracted tweets: "
        f"{len(result):,}"
    )

    print(
        f"AmazonHelp tweets: "
        f"{(
            result['author_id']
            .astype(str)
            .str.strip()
            .eq(BRAND)
        ).sum():,}"
    )

    print(
        f"Customer tweets: "
        f"{(
            result['inbound']
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('true')
        ).sum():,}"
    )

    print(
        f"\nSaved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()