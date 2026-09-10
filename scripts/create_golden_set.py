import pandas as pd
import re
from pathlib import Path

INPUT = "data/processed/amazonhelp_conversation_tweets.parquet"
OUTPUT = "data/golden/golden_set.csv"

N = 200
RANDOM_STATE = 42


def normalize_id(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("Loading dataset...")

    df = pd.read_parquet(INPUT)

    # Identify AmazonHelp tweets
    df["author_id_norm"] = df["author_id"].map(normalize_id)

    brand_mask = (
        (df["author_id_norm"] == "AmazonHelp")
        & (
            df["inbound"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("false")
        )
    )

    customers = df[~brand_mask].copy()

    # Clean messages
    customers["text_clean"] = customers["text"].map(clean_text)

    customers = customers[
        customers["text_clean"].str.len() >= 10
    ].copy()

    # Remove obvious non-English messages approximately.
    # This is deliberately conservative; ambiguous messages remain.
    def english_ratio(text):
        letters = re.findall(r"[A-Za-z]", text)

        if not letters:
            return 0

        ascii_letters = sum(c.isascii() for c in letters)

        return ascii_letters / len(letters)

    customers["english_ratio"] = customers["text_clean"].map(
        english_ratio
    )

    customers = customers[
        customers["english_ratio"] >= 0.70
    ].copy()

    # Message length buckets
    customers["word_count"] = (
        customers["text_clean"]
        .str.split()
        .str.len()
    )

    customers["length_bucket"] = pd.cut(
        customers["word_count"],
        bins=[0, 10, 20, 35, 1000],
        labels=[
            "short",
            "medium",
            "long",
            "very_long"
        ]
    )

    # We want diversity rather than simply taking the first 200 rows.
    #
    # Sampling across length buckets helps prevent the golden set
    # from being dominated by short tweets.
    bucket_counts = {
        "short": 50,
        "medium": 60,
        "long": 60,
        "very_long": 30,
    }

    samples = []

    for bucket, count in bucket_counts.items():

        subset = customers[
            customers["length_bucket"] == bucket
        ]

        count = min(count, len(subset))

        sample = subset.sample(
            n=count,
            random_state=RANDOM_STATE
        )

        samples.append(sample)

    golden = pd.concat(samples)

    golden = golden.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Keep exactly 200 if possible
    golden = golden.head(N)

    # Annotation columns
    golden["gold_intent"] = ""
    golden["gold_escalation"] = ""
    golden["gold_reason"] = ""

    golden["annotator_notes"] = ""

    output_columns = [
        "tweet_id",
        "text",
        "text_clean",
        "word_count",
        "gold_intent",
        "gold_escalation",
        "gold_reason",
        "annotator_notes",
    ]

    golden = golden[output_columns]

    Path("data/golden").mkdir(
        parents=True,
        exist_ok=True
    )

    golden.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print(f"Golden-set candidates: {len(golden)}")
    print(f"Saved to: {OUTPUT}")

    print()
    print("Length distribution:")
    print(
        golden["word_count"]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()