from pathlib import Path
from collections import Counter

import pandas as pd


DATA_FILE = Path("data/raw/twcs.csv")
CHUNK_SIZE = 100_000


# Brands discovered during our initial dataset scan.
BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "VirginTrains",
    "Ask_Spectrum",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "sainsburys",
    "GWRHelp",
    "AskPlayStation",
    "ChipotleTweets",
    "VerizonSupport",
    "UPSHelp",
    "ATVIAssist",
    "O2",
    "Safaricom_Care",
    "idea_cares",
    "AskTarget",
    "AirAsiaSupport",
    "BofA_Help",
    "SW_Help",
]


def normalize_id(value):
    """
    Normalize tweet IDs.

    The CSV can cause IDs such as:
        123
        123.0
        "123"
        "123.0"

    to be represented differently.

    We normalize all of them to:
        "123"
    """

    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def main():

    print("=" * 70)
    print("HIVER SUPPORT AGENT - BRAND INTERACTION ANALYSIS")
    print("=" * 70)

    # =========================================================
    # PASS 1
    # Find all tweet IDs belonging to each brand.
    # =========================================================

    brand_tweet_ids = {
        brand: set()
        for brand in BRANDS
    }

    print("\nPASS 1: Collecting brand tweet IDs...")

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

        print(f"  Processing chunk {chunk_number}...")

        # Normalize tweet IDs.
        df["tweet_id"] = df["tweet_id"].map(normalize_id)

        # Find tweets belonging to each brand.
        for brand in BRANDS:

            ids = df.loc[
                df["author_id"]
                .astype(str)
                .str.strip()
                .eq(brand),
                "tweet_id",
            ]

            brand_tweet_ids[brand].update(
                ids.dropna()
            )

    # =========================================================
    # BRAND TWEET COUNTS
    # =========================================================

    print("\nBrand tweet counts:")
    print("-" * 70)

    for brand in BRANDS:

        print(
            f"{brand:<30}"
            f"{len(brand_tweet_ids[brand]):>10,}"
        )

    # =========================================================
    # CREATE:
    #
    # tweet_id -> brand
    #
    # Example:
    #
    # "12345" -> "sprintcare"
    # =========================================================

    tweet_to_brand = {}

    for brand, tweet_ids in brand_tweet_ids.items():

        for tweet_id in tweet_ids:

            tweet_to_brand[tweet_id] = brand

    # =========================================================
    # PASS 2
    #
    # Find customer tweets whose parent tweet belongs
    # to one of our brands.
    # =========================================================

    customer_messages = Counter()
    brand_replies = Counter()

    print("\nPASS 2: Finding customer replies to brand tweets...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "text",
                "in_response_to_tweet_id",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        print(f"  Processing chunk {chunk_number}...")

        # -----------------------------------------------------
        # Count brand replies.
        # -----------------------------------------------------

        for brand in BRANDS:

            brand_replies[brand] += int(
                df["author_id"]
                .astype(str)
                .str.strip()
                .eq(brand)
                .sum()
            )

        # -----------------------------------------------------
        # Select customer/inbound tweets.
        # -----------------------------------------------------

        customer_rows = df[
            df["inbound"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        ].copy()

        # -----------------------------------------------------
        # Normalize parent tweet IDs.
        # -----------------------------------------------------

        customer_rows["parent_id"] = (
            customer_rows[
                "in_response_to_tweet_id"
            ]
            .map(normalize_id)
        )

        # -----------------------------------------------------
        # Match customer tweet -> parent brand tweet.
        # -----------------------------------------------------

        for parent_id in customer_rows[
            "parent_id"
        ].dropna():

            brand = tweet_to_brand.get(parent_id)

            if brand is not None:

                customer_messages[brand] += 1

    # =========================================================
    # RESULTS
    # =========================================================

    print("\n" + "=" * 70)
    print("BRAND INTERACTION ANALYSIS")
    print("=" * 70)

    results = []

    for brand in BRANDS:

        replies = brand_replies[brand]
        customers = customer_messages[brand]

        results.append(
            {
                "brand": brand,
                "brand_replies": replies,
                "customer_messages": customers,
                "total_interactions": (
                    replies + customers
                ),
            }
        )

    # Sort by customer interaction volume.
    results.sort(
        key=lambda x: x["customer_messages"],
        reverse=True,
    )

    print(
        f"\n{'Brand':<25}"
        f"{'Brand replies':>15}"
        f"{'Customer msgs':>16}"
        f"{'Total':>14}"
    )

    print("-" * 70)

    for row in results:

        print(
            f"{row['brand']:<25}"
            f"{row['brand_replies']:>15,}"
            f"{row['customer_messages']:>16,}"
            f"{row['total_interactions']:>14,}"
        )


if __name__ == "__main__":
    main()