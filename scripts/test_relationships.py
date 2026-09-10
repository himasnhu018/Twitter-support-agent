from pathlib import Path
import pandas as pd


DATA_FILE = Path("data/raw/twcs.csv")


def normalize_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def main():
    df = pd.read_csv(
        DATA_FILE,
        nrows=1000,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    df["tweet_id"] = df["tweet_id"].map(normalize_id)
    df["parent_id"] = df["in_response_to_tweet_id"].map(normalize_id)

    # Identify brand tweets.
    brand_tweets = df[
        df["inbound"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "false"
    ].copy()

    # Build tweet_id -> brand mapping.
    tweet_to_brand = {}

    for _, row in brand_tweets.iterrows():
        tweet_to_brand[row["tweet_id"]] = row["author_id"]

    print("=" * 70)
    print("TESTING TWEET RELATIONSHIPS")
    print("=" * 70)

    found = 0

    # Find customer tweets whose parent is a brand tweet.
    customer_tweets = df[
        df["inbound"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "true"
    ]

    for _, customer in customer_tweets.iterrows():

        parent_id = customer["parent_id"]

        if parent_id in tweet_to_brand:

            brand = tweet_to_brand[parent_id]

            print("\n" + "-" * 70)
            print(f"Brand: {brand}")

            print(
                f"\nBRAND TWEET [{parent_id}]:"
            )

            brand_row = df[
                df["tweet_id"] == parent_id
            ].iloc[0]

            print(brand_row["text"])

            print(
                f"\nCUSTOMER TWEET [{customer['tweet_id']}]:"
            )

            print(customer["text"])

            found += 1

            if found >= 5:
                break

    print("\n" + "=" * 70)
    print(f"FOUND {found} BRAND -> CUSTOMER RELATIONSHIPS")
    print("=" * 70)


if __name__ == "__main__":
    main()