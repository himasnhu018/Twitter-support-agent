from pathlib import Path

import pandas as pd


RAW_FILE = Path("data/raw/twcs.csv")
EXTRACTED_FILE = Path("data/processed/amazonhelp_tweets.parquet")
OUTPUT_FILE = Path("data/processed/amazonhelp_conversation_tweets.parquet")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


def normalize_id(value):
    """Normalize tweet IDs to consistent strings."""

    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def is_true(value):
    """Robustly interpret inbound values."""

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def split_response_ids(value):
    """
    response_tweet_id can contain multiple IDs.

    Example:
        '5,7'
        '9,6,10'
    """

    if pd.isna(value):
        return []

    result = []

    for value_part in str(value).split(","):
        normalized = normalize_id(value_part)

        if normalized is not None:
            result.append(normalized)

    return result


def main():

    print("=" * 70)
    print("AMAZONHELP CONVERSATION RECONSTRUCTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load our existing extracted dataset.
    # ---------------------------------------------------------

    print("\nLoading extracted AmazonHelp data...")

    extracted = pd.read_parquet(EXTRACTED_FILE)

    extracted["tweet_id_clean"] = (
        extracted["tweet_id_clean"]
        .map(normalize_id)
    )

    extracted["parent_tweet_id"] = (
        extracted["parent_tweet_id"]
        .map(normalize_id)
    )

    existing_ids = set(
        extracted["tweet_id_clean"].dropna()
    )

    print(
        f"Existing extracted tweets: "
        f"{len(existing_ids):,}"
    )

    # ---------------------------------------------------------
    # Find all parent IDs of AmazonHelp tweets.
    #
    # These are customer tweets that AmazonHelp responded to.
    # ---------------------------------------------------------

    amazon_rows = extracted[
        extracted["author_id"]
        .astype(str)
        .str.strip()
        .eq(BRAND)
    ]

    missing_parent_ids = set(
        amazon_rows["parent_tweet_id"].dropna()
    )

    missing_parent_ids -= existing_ids

    print(
        f"Missing parent tweets to recover: "
        f"{len(missing_parent_ids):,}"
    )

    # ---------------------------------------------------------
    # We also want children of Amazon tweets.
    #
    # Those are already mostly present, but we explicitly
    # collect them again to verify the relationships.
    # ---------------------------------------------------------

    amazon_tweet_ids = set(
        amazon_rows["tweet_id_clean"].dropna()
    )

    recovered_rows = []

    print("\nScanning raw dataset for missing context...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            RAW_FILE,
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

        df["tweet_id_clean"] = (
            df["tweet_id"]
            .map(normalize_id)
        )

        df["parent_tweet_id"] = (
            df["in_response_to_tweet_id"]
            .map(normalize_id)
        )

        # -----------------------------------------------------
        # 1. Missing parent tweets.
        #
        # Example:
        #
        # Customer tweet 100
        #       ↓
        # Amazon tweet 101
        #
        # We need tweet 100.
        # -----------------------------------------------------

        parent_rows = df[
            df["tweet_id_clean"]
            .isin(missing_parent_ids)
        ]

        if not parent_rows.empty:
            recovered_rows.append(parent_rows)

        # -----------------------------------------------------
        # 2. Any tweets directly replying to Amazon tweets.
        #
        # This gives us a consistency check and protects us
        # from extraction edge cases.
        # -----------------------------------------------------

        child_rows = df[
            df["parent_tweet_id"]
            .isin(amazon_tweet_ids)
        ]

        if not child_rows.empty:
            recovered_rows.append(child_rows)

    # ---------------------------------------------------------
    # Combine everything.
    # ---------------------------------------------------------

    if recovered_rows:

        recovered = pd.concat(
            recovered_rows,
            ignore_index=True,
        )

        print(
            f"\nRecovered raw rows: "
            f"{len(recovered):,}"
        )

        combined = pd.concat(
            [
                extracted,
                recovered,
            ],
            ignore_index=True,
        )

    else:

        combined = extracted

    # ---------------------------------------------------------
    # Normalize again.
    # ---------------------------------------------------------

    combined["tweet_id_clean"] = (
        combined["tweet_id_clean"]
        .map(normalize_id)
    )

    combined["parent_tweet_id"] = (
        combined["parent_tweet_id"]
        .map(normalize_id)
    )

    # ---------------------------------------------------------
    # Remove duplicates.
    # ---------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=["tweet_id_clean"]
    )

    # ---------------------------------------------------------
    # Sort chronologically.
    # ---------------------------------------------------------

    combined["created_at_parsed"] = pd.to_datetime(
    combined["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
    utc=True,
)

    combined = combined.sort_values(
        "created_at_parsed"
    )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    combined.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Statistics.
    # ---------------------------------------------------------

    amazon_count = (
        combined["author_id"]
        .astype(str)
        .str.strip()
        .eq(BRAND)
        .sum()
    )

    customer_count = (
        combined["inbound"]
        .map(is_true)
        .sum()
    )

    print("\n" + "=" * 70)
    print("CONVERSATION CONTEXT COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal tweets:       {len(combined):,}"
    )

    print(
        f"AmazonHelp tweets:  {amazon_count:,}"
    )

    print(
        f"Customer tweets:    {customer_count:,}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()