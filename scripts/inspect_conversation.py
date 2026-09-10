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
    print("=" * 70)
    print("TWCS CONVERSATION INSPECTOR")
    print("=" * 70)

    # Read the first 5,000 rows for an initial example.
    df = pd.read_csv(
        DATA_FILE,
        nrows=5000,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    df["tweet_id_norm"] = df["tweet_id"].map(normalize_id)
    df["parent_id_norm"] = df["in_response_to_tweet_id"].map(
        normalize_id
    )

    # Build ID -> row lookup.
    tweets = {}

    for _, row in df.iterrows():
        tweets[row["tweet_id_norm"]] = row

    # Find a brand tweet which has a customer response.
    for _, row in df.iterrows():

        if row["inbound"] is not False:
            continue

        response_ids = row["response_tweet_id"]

        if pd.isna(response_ids):
            continue

        # response_tweet_id may contain one or multiple IDs.
        response_list = str(response_ids).split(",")

        for response_id in response_list:
            response_id = normalize_id(response_id)

            if response_id in tweets:
                response = tweets[response_id]

                if response["inbound"] is True:

                    print("\nFOUND CONVERSATION")
                    print("-" * 70)

                    print(
                        f"\nBrand: {row['author_id']}"
                    )

                    print(
                        f"\nBrand tweet [{row['tweet_id_norm']}]:"
                    )

                    print(row["text"])

                    print(
                        f"\nCustomer reply [{response_id}]:"
                    )

                    print(response["text"])

                    print("\nParent relationship:")
                    print(
                        f"{response_id} "
                        f"-> "
                        f"{row['tweet_id_norm']}"
                    )

                    return

    print("\nNo conversation found in first 5,000 rows.")


if __name__ == "__main__":
    main()