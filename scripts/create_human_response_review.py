import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    input_path = (
        root
        / "data"
        / "golden"
        / "response_judge_predictions.csv"
    )

    output_path = (
        root
        / "data"
        / "golden"
        / "human_response_review.csv"
    )

    df = pd.read_csv(input_path)

    print("Loaded responses:", len(df))

    # ---------------------------------------------------------
    # Create useful sampling groups
    # ---------------------------------------------------------

    df["intent_match"] = (
        df["gold_intent"]
        == df["predicted_intent"]
    )

    # Quality buckets based on judge overall score.
    df["quality_bucket"] = pd.cut(
        df["overall_mean"],
        bins=[0, 3, 4, 4.5, 5.01],
        labels=[
            "low",
            "medium",
            "high",
            "very_high",
        ],
        include_lowest=True,
    )

    # ---------------------------------------------------------
    # Deterministic stratified sampling
    # ---------------------------------------------------------

    samples = []

    # 1. Lowest-quality responses.
    low_quality = (
        df.sort_values("overall_mean")
        .head(8)
    )

    samples.append(low_quality)

    # 2. Intent mistakes.
    intent_errors = (
        df[df["intent_match"] == False]
        .sort_values("overall_mean")
        .head(8)
    )

    samples.append(intent_errors)

    # 3. Low helpfulness.
    low_helpfulness = (
        df.sort_values(
            ["helpfulness", "overall_mean"]
        )
        .head(8)
    )

    samples.append(low_helpfulness)

    # 4. Cases with potentially important safety/grounding
    #    variation.
    grounding_cases = (
        df.sort_values(
            ["groundedness", "safety"]
        )
        .head(6)
    )

    samples.append(grounding_cases)

    # 5. High-quality examples.
    high_quality = (
        df.sort_values(
            "overall_mean",
            ascending=False,
        )
        .head(10)
    )

    samples.append(high_quality)

    # ---------------------------------------------------------
    # Combine and remove duplicates
    # ---------------------------------------------------------

    review = pd.concat(
        samples,
        ignore_index=True,
    )

    review = review.drop_duplicates(
        subset=["tweet_id"]
    )

    # If we have more than requested, take a deterministic
    # diverse subset.
    if len(review) > args.n:

        review = (
            review
            .sort_values(
                [
                    "overall_mean",
                    "tweet_id",
                ]
            )
            .head(args.n)
        )

    # If fewer than requested, fill from remaining examples.
    if len(review) < args.n:

        remaining = df[
            ~df["tweet_id"].isin(
                review["tweet_id"]
            )
        ]

        needed = args.n - len(review)

        fill = (
            remaining
            .sample(
                n=min(
                    needed,
                    len(remaining),
                ),
                random_state=42,
            )
        )

        review = pd.concat(
            [review, fill],
            ignore_index=True,
        )

    # ---------------------------------------------------------
    # Human-review columns
    # ---------------------------------------------------------

    review = review[
        [
            "tweet_id",
            "customer_text",
            "gold_intent",
            "predicted_intent",
            "generated_reply",
            "relevance",
            "groundedness",
            "helpfulness",
            "safety",
            "historical_grounding",
            "overall_mean",
            "reason",
        ]
    ].copy()

    # Add blank human fields.
    review["human_relevance"] = ""
    review["human_groundedness"] = ""
    review["human_helpfulness"] = ""
    review["human_safety"] = ""
    review["human_historical_grounding"] = ""
    review["human_reason"] = ""

    # Add an ordering number to make manual review easy.
    review.insert(
        0,
        "review_number",
        range(1, len(review) + 1),
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    review.to_csv(
        output_path,
        index=False,
    )

    print()
    print("========================================")
    print("HUMAN REVIEW SET CREATED")
    print("========================================")
    print(f"Examples: {len(review)}")
    print(f"Output:   {output_path}")
    print("========================================")

    print()
    print("Intent distribution:")
    print(
        review["gold_intent"]
        .value_counts()
        .to_string()
    )

    print()
    print("Intent errors included:")
    print(
        (~review["gold_intent"].eq(
            review["predicted_intent"]
        )).sum()
    )

    print()
    print("Judge overall score distribution:")
    print(
        review["overall_mean"]
        .describe()
        .to_string()
    )

    print()
    print("Review the CSV manually and fill ONLY:")
    print()
    print("  human_relevance")
    print("  human_groundedness")
    print("  human_helpfulness")
    print("  human_safety")
    print("  human_historical_grounding")
    print("  human_reason")


if __name__ == "__main__":
    main()