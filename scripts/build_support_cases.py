import pandas as pd


INPUT = "data/processed/amazon_retrieval_corpus.parquet"
OUTPUT = "data/processed/amazon_support_cases.parquet"


def normalize_id(value):
    """
    Normalize tweet IDs so values such as:
        12345
        12345.0
        "12345"
        " 12345 "
    become:
        "12345"
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def main():
    print("Loading leakage-safe corpus...")

    df = pd.read_parquet(INPUT)

    print(f"Rows loaded: {len(df):,}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = [
        "tweet_id",
        "in_response_to_tweet_id",
        "inbound",
        "text",
        "component_id",
        "created_at_parsed",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # Normalize tweet IDs
    # ---------------------------------------------------------

    df["tweet_id_clean"] = df["tweet_id"].apply(normalize_id)

    df["parent_tweet_id_clean"] = df[
        "in_response_to_tweet_id"
    ].apply(normalize_id)

    # Remove rows without a usable tweet ID.
    df = df[df["tweet_id_clean"].notna()].copy()

    print(f"Rows with valid tweet IDs: {len(df):,}")

    # ---------------------------------------------------------
    # Handle duplicate tweet IDs
    # ---------------------------------------------------------

    duplicate_count = df["tweet_id_clean"].duplicated().sum()

    if duplicate_count > 0:
        print(
            f"WARNING: Found {duplicate_count:,} duplicate tweet IDs."
        )
        print("Keeping the first occurrence of each tweet ID.")

        df = df.drop_duplicates(
            subset=["tweet_id_clean"],
            keep="first",
        ).copy()

    print(f"Unique tweets: {len(df):,}")

    # ---------------------------------------------------------
    # Create tweet lookup
    #
    # IMPORTANT:
    # drop=False keeps tweet_id_clean as a column as well as
    # the dataframe index.
    # ---------------------------------------------------------

    by_id = df.set_index(
        "tweet_id_clean",
        drop=False,
    )

    # ---------------------------------------------------------
    # Build support cases
    # ---------------------------------------------------------

    cases = []

    # Desired relationship:
    #
    # Customer tweet
    #       |
    #       | replied to
    #       v
    # AmazonHelp reply
    #
    # Therefore:
    #
    # current row  = AmazonHelp / agent
    # parent row   = customer
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        # -----------------------------------------------------
        # Current tweet must be an AmazonHelp / agent tweet.
        #
        # Based on the corpus convention:
        #   inbound=True  -> customer
        #   inbound=False -> AmazonHelp
        # -----------------------------------------------------

        if bool(row["inbound"]):
            continue

        # -----------------------------------------------------
        # Get parent tweet ID
        # -----------------------------------------------------

        parent_id = row["parent_tweet_id_clean"]

        if parent_id is None:
            continue

        # -----------------------------------------------------
        # Parent must exist in the corpus
        # -----------------------------------------------------

        if parent_id not in by_id.index:
            continue

        parent = by_id.loc[parent_id]

        # -----------------------------------------------------
        # Parent must be a customer tweet.
        #
        # Because duplicate IDs were removed above, `parent`
        # should be a single Series rather than a DataFrame.
        # -----------------------------------------------------

        if bool(parent["inbound"]):
            pass
        else:
            continue

        # -----------------------------------------------------
        # Create support case
        # -----------------------------------------------------

        cases.append(
            {
                "customer_tweet_id": parent["tweet_id_clean"],
                "agent_tweet_id": row["tweet_id_clean"],
                "customer_text": str(parent["text"]),
                "agent_response": str(row["text"]),
                "component_id": row["component_id"],
                "customer_created_at": parent["created_at_parsed"],
                "agent_created_at": row["created_at_parsed"],
            }
        )

    # ---------------------------------------------------------
    # Convert cases to DataFrame
    # ---------------------------------------------------------

    cases_df = pd.DataFrame(cases)

    if cases_df.empty:
        raise ValueError(
            "No support cases were created.\n"
            "Check the values of `inbound`, "
            "`tweet_id`, and `in_response_to_tweet_id`."
        )

    # ---------------------------------------------------------
    # Remove exact duplicate customer-agent pairs
    # ---------------------------------------------------------

    before_dedup = len(cases_df)

    cases_df = cases_df.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "agent_tweet_id",
        ]
    ).reset_index(drop=True)

    removed_duplicates = before_dedup - len(cases_df)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SUPPORT CASE DATASET")
    print("=" * 80)

    print()
    print(f"Support cases: {len(cases_df):,}")

    print(
        f"Unique customer messages: "
        f"{cases_df['customer_tweet_id'].nunique():,}"
    )

    print(
        f"Unique conversations: "
        f"{cases_df['component_id'].nunique():,}"
    )

    print(
        f"Duplicate cases removed: "
        f"{removed_duplicates:,}"
    )

    # ---------------------------------------------------------
    # Example cases
    # ---------------------------------------------------------

    print()
    print("Example cases:")
    print("=" * 100)

    for _, row in cases_df.head(10).iterrows():

        print()
        print("CUSTOMER:")
        print(row["customer_text"])

        print()
        print("AMAZONHELP:")
        print(row["agent_response"])

        print("-" * 100)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    cases_df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print(f"Saved: {OUTPUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()