import argparse
import pandas as pd


INPUT = "data/golden/golden_set.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(INPUT)

    # Golden examples are 1-indexed for annotation purposes.
    start_idx = args.start - 1
    end_idx = args.end

    batch = df.iloc[start_idx:end_idx].copy()

    print("=" * 100)
    print(f"GOLDEN SET EXAMPLES {args.start}-{args.end}")
    print("=" * 100)

    for i, (_, row) in enumerate(batch.iterrows(), start=args.start):
        print()
        print(f"#{i}")
        print("-" * 100)
        print(f"tweet_id: {row['tweet_id']}")
        print(f"text: {row['text']}")

        # Show existing labels if they exist.
        if pd.notna(row.get("gold_intent", None)):
            print(f"CURRENT INTENT: {row['gold_intent']}")

        if pd.notna(row.get("gold_escalation", None)):
            print(f"CURRENT ESCALATION: {row['gold_escalation']}")

        if pd.notna(row.get("gold_reason", None)):
            print(f"CURRENT REASON: {row['gold_reason']}")

        print("-" * 100)

    print()
    print("=" * 100)
    print(f"Showing {len(batch)} examples.")
    print("=" * 100)


if __name__ == "__main__":
    main()