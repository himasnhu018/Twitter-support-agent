import pandas as pd

INPUT = "data/processed/amazon_support_cases.parquet"


def main():
    df = pd.read_parquet(INPUT)

    print("=" * 80)
    print("AMAZON SUPPORT CASE ANALYSIS")
    print("=" * 80)

    print(f"Cases: {len(df):,}")
    print(f"Unique customer messages: {df['customer_tweet_id'].nunique():,}")
    print(f"Unique conversations: {df['component_id'].nunique():,}")

    # ---------------------------------------------------------
    # Text lengths
    # ---------------------------------------------------------

    df["customer_words"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["response_words"] = (
        df["agent_response"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print()
    print("CUSTOMER MESSAGE LENGTH")
    print(f"Mean:   {df['customer_words'].mean():.2f}")
    print(f"Median: {df['customer_words'].median():.0f}")
    print(f"P90:    {df['customer_words'].quantile(.90):.0f}")
    print(f"Max:    {df['customer_words'].max():.0f}")

    print()
    print("AGENT RESPONSE LENGTH")
    print(f"Mean:   {df['response_words'].mean():.2f}")
    print(f"Median: {df['response_words'].median():.0f}")
    print(f"P90:    {df['response_words'].quantile(.90):.0f}")
    print(f"Max:    {df['response_words'].max():.0f}")

    # ---------------------------------------------------------
    # Empty / suspicious text
    # ---------------------------------------------------------

    empty_customer = (
        df["customer_text"].fillna("").astype(str).str.strip() == ""
    ).sum()

    empty_response = (
        df["agent_response"].fillna("").astype(str).str.strip() == ""
    ).sum()

    print()
    print("TEXT QUALITY")
    print(f"Empty customer messages: {empty_customer:,}")
    print(f"Empty agent responses:   {empty_response:,}")

    # ---------------------------------------------------------
    # Exact duplicate customer messages
    # ---------------------------------------------------------

    duplicate_customer = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .duplicated()
        .sum()
    )

    print()
    print("DUPLICATION")
    print(f"Duplicate customer texts: {duplicate_customer:,}")

    # ---------------------------------------------------------
    # Generic responses
    # ---------------------------------------------------------

    generic_patterns = [
        "let us know if you have any other questions",
        "please contact us",
        "we'll be happy to help",
        "we are happy to help",
        "please send us",
        "please fill out the form",
        "contact us",
    ]

    response_lower = (
        df["agent_response"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    generic_mask = False

    for pattern in generic_patterns:
        generic_mask = generic_mask | response_lower.str.contains(
            pattern,
            regex=False,
            na=False
        )

    print()
    print("GENERIC RESPONSE SIGNAL")
    print(
        f"Responses matching at least one generic pattern: "
        f"{generic_mask.sum():,} "
        f"({generic_mask.mean() * 100:.2f}%)"
    )

    # ---------------------------------------------------------
    # Show examples
    # ---------------------------------------------------------

    print()
    print("RANDOM EXAMPLES")
    print("=" * 80)

    sample = df.sample(
        min(10, len(df)),
        random_state=42
    )

    for _, row in sample.iterrows():
        print()
        print("CUSTOMER:")
        print(row["customer_text"])
        print()
        print("RESPONSE:")
        print(row["agent_response"])
        print("-" * 80)


if __name__ == "__main__":
    main()