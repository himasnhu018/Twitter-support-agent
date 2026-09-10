import pandas as pd
import re
from collections import Counter

INPUT = "data/processed/amazonhelp_conversation_tweets.parquet"
OUTPUT = "data/processed/amazon_intent_discovery.csv"

BRAND = "AmazonHelp"


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

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtags but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove common Twitter artifacts
    text = text.replace("&amp;", " and ")
    text = text.replace("&gt;", " ")
    text = text.replace("&lt;", " ")

    # Keep letters/numbers
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def main():
    print("Loading dataset...")

    df = pd.read_parquet(INPUT)

    print(f"Total rows: {len(df):,}")

    # Normalize columns
    df["author_id_norm"] = df["author_id"].map(normalize_id)

    # Amazon tweets
    brand_mask = (
        (df["author_id_norm"] == BRAND)
        & (
            df["inbound"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("false")
        )
    )

    # Customer tweets
    customer_mask = ~brand_mask

    customers = df[customer_mask].copy()

    print(f"Customer messages: {len(customers):,}")

    # Clean text
    customers["clean_text"] = customers["text"].map(clean_text)

    # Remove empty messages
    customers = customers[
        customers["clean_text"].str.len() > 5
    ].copy()

    print(f"Usable customer messages: {len(customers):,}")

    # ---------------------------------------------------------
    # 1. Message length statistics
    # ---------------------------------------------------------

    customers["word_count"] = (
        customers["clean_text"]
        .str.split()
        .str.len()
    )

    print("\nMESSAGE LENGTH")
    print("----------------")
    print(customers["word_count"].describe())

    # ---------------------------------------------------------
    # 2. Most common words
    # ---------------------------------------------------------

    stopwords = {
        "the", "and", "to", "of", "a", "i", "is", "it",
        "for", "in", "on", "my", "you", "me", "this", "that",
        "amazon", "help", "please", "can", "have", "has",
        "with", "but", "was", "are", "be", "not", "your",
        "do", "what", "how", "why", "from", "at", "or",
        "we", "they", "so", "just", "im", "ive", "its",
        "been", "would", "will", "about", "there", "all",
        "as", "if", "when", "get", "got", "my", "hi",
        "hey", "thanks", "thank"
    }

    counter = Counter()

    for text in customers["clean_text"]:
        words = text.split()

        for word in words:
            if (
                len(word) >= 3
                and word not in stopwords
                and not word.isnumeric()
            ):
                counter[word] += 1

    print("\nTOP WORDS")
    print("----------------")

    for word, count in counter.most_common(100):
        print(f"{word:25} {count:,}")

    # ---------------------------------------------------------
    # 3. Useful bigrams
    # ---------------------------------------------------------

    bigram_counter = Counter()

    for text in customers["clean_text"]:
        words = [
            w for w in text.split()
            if w not in stopwords and len(w) >= 3
        ]

        for a, b in zip(words, words[1:]):
            bigram_counter[f"{a} {b}"] += 1

    print("\nTOP BIGRAMS")
    print("----------------")

    for phrase, count in bigram_counter.most_common(100):
        print(f"{phrase:30} {count:,}")

    # ---------------------------------------------------------
    # 4. Candidate keyword counts
    # ---------------------------------------------------------

    candidate_terms = {
        "delivery": [
            "delivery", "delivered", "deliver", "shipping",
            "shipment", "parcel", "package"
        ],

        "tracking": [
            "tracking", "track", "courier", "where"
        ],

        "delay": [
            "delay", "delayed", "late", "waiting", "wait"
        ],

        "return": [
            "return", "returning", "send back"
        ],

        "refund": [
            "refund", "refunded", "money back", "reimbursement"
        ],

        "payment": [
            "payment", "pay", "paid", "card", "charge",
            "charged", "billing"
        ],

        "prime": [
            "prime", "membership"
        ],

        "seller": [
            "seller", "vendor", "third party"
        ],

        "fraud": [
            "fraud", "fraudulent", "scam", "fake", "counterfeit"
        ],

        "account": [
            "account", "login", "password", "sign in"
        ],

        "technical": [
            "app", "website", "error", "bug", "glitch",
            "crash", "not working", "unable"
        ],

        "order": [
            "order", "ordered"
        ],

        "product": [
            "product", "item", "book", "phone", "game"
        ],

        "cancel": [
            "cancel", "cancellation"
        ],

        "complaint": [
            "complaint", "complain", "terrible", "worst",
            "disappointed", "unhappy", "pathetic"
        ]
    }

    print("\nCANDIDATE INTENT KEYWORDS")
    print("----------------")

    results = []

    for intent, terms in candidate_terms.items():

        mask = pd.Series(False, index=customers.index)

        for term in terms:
            mask |= customers["clean_text"].str.contains(
                re.escape(term),
                regex=True,
                na=False
            )

        count = int(mask.sum())
        percentage = count / len(customers) * 100

        results.append({
            "candidate_intent": intent,
            "matching_messages": count,
            "percentage": round(percentage, 2)
        })

        print(
            f"{intent:20} "
            f"{count:8,} "
            f"({percentage:.2f}%)"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT,
        index=False
    )

    print(f"\nSaved summary to: {OUTPUT}")

    # ---------------------------------------------------------
    # 5. Random examples for candidate categories
    # ---------------------------------------------------------

    print("\nRANDOM EXAMPLES")
    print("================")

    for intent, terms in candidate_terms.items():

        mask = pd.Series(False, index=customers.index)

        for term in terms:
            mask |= customers["clean_text"].str.contains(
                re.escape(term),
                regex=True,
                na=False
            )

        subset = customers[mask]

        if len(subset) == 0:
            continue

        print(f"\n### {intent.upper()}")

        sample = subset.sample(
            min(5, len(subset)),
            random_state=42
        )

        for _, row in sample.iterrows():
            print("-", row["text"][:300])


if __name__ == "__main__":
    main()