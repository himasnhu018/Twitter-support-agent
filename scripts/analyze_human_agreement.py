import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "relevance",
    "groundedness",
    "helpfulness",
    "safety",
    "historical_grounding",
]

HUMAN_COLUMNS = [
    "human_relevance",
    "human_groundedness",
    "human_helpfulness",
    "human_safety",
    "human_historical_grounding",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    input_path = (
        root
        / "data"
        / "golden"
        / "human_response_review.csv"
    )

    output_path = (
        root
        / "data"
        / "golden"
        / "human_agreement_results.csv"
    )

    print("Loading human review:")
    print(input_path)

    df = pd.read_csv(input_path)

    print(f"Rows loaded: {len(df)}")

    # ---------------------------------------------------------
    # Validate human scores
    # ---------------------------------------------------------

    missing_columns = [
        c
        for c in HUMAN_COLUMNS
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing human columns: {missing_columns}"
        )

    for column in HUMAN_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_counts = df[HUMAN_COLUMNS].isna().sum()

    print()
    print("Missing human scores:")
    print(missing_counts.to_string())

    if missing_counts.sum() > 0:
        raise ValueError(
            "Some human scores are missing. "
            "Complete all 40 reviews before running agreement analysis."
        )

    # ---------------------------------------------------------
    # Validate score range
    # ---------------------------------------------------------

    invalid_scores = []

    for column in HUMAN_COLUMNS:

        invalid = df[
            ~df[column].between(1, 5)
        ]

        if len(invalid) > 0:
            invalid_scores.append(
                (
                    column,
                    len(invalid),
                )
            )

    if invalid_scores:
        raise ValueError(
            f"Human scores outside 1-5: {invalid_scores}"
        )

    # ---------------------------------------------------------
    # Calculate agreement
    # ---------------------------------------------------------

    results = []

    print()
    print("========================================")
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("========================================")

    for score_column, human_column in zip(
        SCORE_COLUMNS,
        HUMAN_COLUMNS,
    ):

        judge = df[score_column].astype(float)
        human = df[human_column].astype(float)

        difference = (
            judge - human
        ).abs()

        exact_agreement = (
            judge == human
        ).mean()

        within_one = (
            difference <= 1
        ).mean()

        mean_judge = judge.mean()
        mean_human = human.mean()

        mean_absolute_difference = (
            difference.mean()
        )

        results.append(
            {
                "dimension": score_column,
                "n": len(df),
                "judge_mean": round(
                    mean_judge,
                    3,
                ),
                "human_mean": round(
                    mean_human,
                    3,
                ),
                "exact_agreement": round(
                    exact_agreement,
                    3,
                ),
                "within_1_agreement": round(
                    within_one,
                    3,
                ),
                "mean_absolute_difference": round(
                    mean_absolute_difference,
                    3,
                ),
            }
        )

        print()
        print(
            score_column.upper()
        )

        print(
            f"Judge mean:              "
            f"{mean_judge:.3f}"
        )

        print(
            f"Human mean:              "
            f"{mean_human:.3f}"
        )

        print(
            f"Exact agreement:         "
            f"{exact_agreement * 100:.1f}%"
        )

        print(
            f"Within-1 agreement:      "
            f"{within_one * 100:.1f}%"
        )

        print(
            f"Mean absolute difference:"
            f" {mean_absolute_difference:.3f}"
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Overall score comparison
    # ---------------------------------------------------------

    df["judge_overall"] = df[
        SCORE_COLUMNS
    ].mean(axis=1)

    df["human_overall"] = df[
        HUMAN_COLUMNS
    ].mean(axis=1)

    df["overall_difference"] = (
        df["judge_overall"]
        - df["human_overall"]
    ).abs()

    overall_exact = (
        df["judge_overall"].round(6)
        == df["human_overall"].round(6)
    ).mean()

    overall_within_half = (
        df["overall_difference"] <= 0.5
    ).mean()

    overall_within_one = (
        df["overall_difference"] <= 1.0
    ).mean()

    print()
    print("========================================")
    print("OVERALL SCORE AGREEMENT")
    print("========================================")

    print(
        f"Judge overall mean:       "
        f"{df['judge_overall'].mean():.3f}"
    )

    print(
        f"Human overall mean:       "
        f"{df['human_overall'].mean():.3f}"
    )

    print(
        f"Exact agreement:          "
        f"{overall_exact * 100:.1f}%"
    )

    print(
        f"Within 0.5 points:        "
        f"{overall_within_half * 100:.1f}%"
    )

    print(
        f"Within 1 point:           "
        f"{overall_within_one * 100:.1f}%"
    )

    # ---------------------------------------------------------
    # Bias
    # ---------------------------------------------------------

    df["judge_minus_human_overall"] = (
        df["judge_overall"]
        - df["human_overall"]
    )

    bias = (
        df["judge_minus_human_overall"]
        .mean()
    )

    print()
    print(
        "Mean judge-human difference:"
    )

    print(
        f"  {bias:+.3f}"
    )

    if bias > 0.25:
        print(
            "  Judge tends to score responses higher "
            "than the human."
        )
    elif bias < -0.25:
        print(
            "  Judge tends to score responses lower "
            "than the human."
        )
    else:
        print(
            "  No large systematic scoring bias detected."
        )

    # ---------------------------------------------------------
    # Per-example disagreements
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("LARGEST DISAGREEMENTS")
    print("========================================")

    disagreement_columns = []

    for score_column, human_column in zip(
        SCORE_COLUMNS,
        HUMAN_COLUMNS,
    ):

        column = (
            score_column
            + "_difference"
        )

        df[column] = (
            df[score_column]
            - df[human_column]
        ).abs()

        disagreement_columns.append(
            column
        )

    df["total_dimension_disagreement"] = (
        df[disagreement_columns]
        .sum(axis=1)
    )

    largest = (
        df.sort_values(
            "total_dimension_disagreement",
            ascending=False,
        )
        .head(10)
    )

    for _, row in largest.iterrows():

        print()
        print("----------------------------------------")

        print(
            f"Tweet ID: {row['tweet_id']}"
        )

        print(
            f"Customer: {row['customer_text']}"
        )

        print(
            f"Reply: {row['generated_reply']}"
        )

        print(
            f"Judge overall: "
            f"{row['judge_overall']:.2f}"
        )

        print(
            f"Human overall: "
            f"{row['human_overall']:.2f}"
        )

        print(
            f"Judge reason: "
            f"{row['reason']}"
        )

        for score_column, human_column in zip(
            SCORE_COLUMNS,
            HUMAN_COLUMNS,
        ):

            print(
                f"{score_column}: "
                f"judge={row[score_column]} "
                f"human={row[human_column]}"
            )

        print(
            f"Human reason: "
            f"{row.get('human_reason', '')}"
        )

    # ---------------------------------------------------------
    # Save dimension-level results
    # ---------------------------------------------------------

    results_df.to_csv(
        output_path,
        index=False,
    )

    # Save enriched human review for later failure analysis.
    enriched_path = (
        root
        / "data"
        / "golden"
        / "human_agreement_enriched.csv"
    )

    df.to_csv(
        enriched_path,
        index=False,
    )

    print()
    print("========================================")
    print("AGREEMENT ANALYSIS COMPLETE")
    print("========================================")

    print(
        f"Dimension results:"
        f" {output_path}"
    )

    print(
        f"Enriched review:"
        f" {enriched_path}"
    )


if __name__ == "__main__":
    main()