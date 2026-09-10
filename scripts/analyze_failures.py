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
        "--top",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    response_path = (
        root
        / "data"
        / "golden"
        / "response_judge_predictions.csv"
    )

    escalation_path = (
        root
        / "data"
        / "golden"
        / "escalation_predictions.csv"
    )

    agreement_path = (
        root
        / "data"
        / "golden"
        / "human_agreement_enriched.csv"
    )

    output_path = (
        root
        / "data"
        / "golden"
        / "failure_analysis.csv"
    )

    print("Loading evaluation files...")

    response = pd.read_csv(response_path)
    escalation = pd.read_csv(escalation_path)

    print(f"Response rows:    {len(response)}")
    print(f"Escalation rows:  {len(escalation)}")

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    response["tweet_id"] = (
        response["tweet_id"]
        .astype(str)
    )

    # Find escalation ID column.
    escalation_id = None

    for candidate in [
        "tweet_id",
        "customer_tweet_id",
        "id",
    ]:
        if candidate in escalation.columns:
            escalation_id = candidate
            break

    if escalation_id is None:
        raise ValueError(
            "Could not find tweet ID in escalation file. "
            f"Columns: {list(escalation.columns)}"
        )

    escalation["tweet_id"] = (
        escalation[escalation_id]
        .astype(str)
    )

    # ---------------------------------------------------------
    # Merge escalation data
    # ---------------------------------------------------------

    escalation_columns = [
        "tweet_id",
    ]

    for candidate in [
        "gold_escalation",
        "predicted_escalation",
        "gold_decision",
        "predicted_decision",
        "decision",
        "reason",
    ]:
        if candidate in escalation.columns:
            escalation_columns.append(candidate)

    escalation_small = escalation[
        escalation_columns
    ].drop_duplicates(
        subset=["tweet_id"]
    )

    df = response.merge(
        escalation_small,
        on="tweet_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Intent errors
    # ---------------------------------------------------------

    df["intent_error"] = (
        df["gold_intent"]
        != df["predicted_intent"]
    )

    # ---------------------------------------------------------
    # Response quality flags
    # ---------------------------------------------------------

    df["low_relevance"] = (
        df["relevance"] <= 2
    )

    df["low_groundedness"] = (
        df["groundedness"] <= 2
    )

    df["low_helpfulness"] = (
        df["helpfulness"] <= 2
    )

    df["low_safety"] = (
        df["safety"] <= 2
    )

    df["low_historical_grounding"] = (
        df["historical_grounding"] <= 2
    )

    # ---------------------------------------------------------
    # Overall response quality
    # ---------------------------------------------------------

    df["response_quality_low"] = (
        df["overall_mean"] < 3.5
    )

    # ---------------------------------------------------------
    # Escalation error detection
    # ---------------------------------------------------------

    if (
        "gold_escalation" in df.columns
        and "predicted_escalation" in df.columns
    ):

        df["escalation_error"] = (
            df["gold_escalation"]
            != df["predicted_escalation"]
        )

        # Unsafe auto-handle:
        df["unsafe_auto_handle"] = (
            (df["gold_escalation"] == "ESCALATE")
            & (df["predicted_escalation"] == "AUTO_HANDLE")
        )

        # Unnecessary escalation:
        df["unnecessary_escalation"] = (
            (df["gold_escalation"] == "AUTO_HANDLE")
            & (df["predicted_escalation"] == "ESCALATE")
        )

    else:

        df["escalation_error"] = False
        df["unsafe_auto_handle"] = False
        df["unnecessary_escalation"] = False

    # ---------------------------------------------------------
    # Failure priority score
    # ---------------------------------------------------------

    # Higher = more important to inspect.
    df["failure_priority"] = 0

    df.loc[
        df["low_safety"],
        "failure_priority",
    ] += 10

    df.loc[
        df["unsafe_auto_handle"],
        "failure_priority",
    ] += 10

    df.loc[
        df["low_groundedness"],
        "failure_priority",
    ] += 5

    df.loc[
        df["low_relevance"],
        "failure_priority",
    ] += 5

    df.loc[
        df["low_helpfulness"],
        "failure_priority",
    ] += 4

    df.loc[
        df["low_historical_grounding"],
        "failure_priority",
    ] += 3

    df.loc[
        df["intent_error"],
        "failure_priority",
    ] += 4

    df.loc[
        df["unnecessary_escalation"],
        "failure_priority",
    ] += 2

    # Lower response score = more severe.
    df["failure_priority"] += (
        5 - df["overall_mean"]
    ).clip(lower=0)

    # ---------------------------------------------------------
    # Failure category
    # ---------------------------------------------------------

    def classify_failure(row):

        if row["unsafe_auto_handle"]:
            return "UNSAFE_AUTO_HANDLE"

        if row["low_safety"]:
            return "SAFETY"

        if row["low_groundedness"]:
            return "GROUNDING"

        if row["low_relevance"]:
            return "RELEVANCE"

        if row["intent_error"]:
            return "INTENT_MISCLASSIFICATION"

        if row["low_helpfulness"]:
            return "HELPFULNESS"

        if row["low_historical_grounding"]:
            return "HISTORICAL_GROUNDING"

        if row["unnecessary_escalation"]:
            return "UNNECESSARY_ESCALATION"

        return "OTHER"

    df["failure_category"] = df.apply(
        classify_failure,
        axis=1,
    )

    # ---------------------------------------------------------
    # Rank failures
    # ---------------------------------------------------------

    failures = df[
        (df["failure_category"] != "OTHER")
    ].copy()

    failures = failures.sort_values(
        [
            "failure_priority",
            "overall_mean",
        ],
        ascending=[
            False,
            True,
        ],
    )

    top_failures = failures.head(
        args.top
    )

    # ---------------------------------------------------------
    # Print top failures
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("TOP FAILURE CASES")
    print("========================================")

    for index, (_, row) in enumerate(
        top_failures.iterrows(),
        start=1,
    ):

        print()
        print(
            f"FAILURE #{index}"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Tweet ID: {row['tweet_id']}"
        )

        print(
            f"Category: {row['failure_category']}"
        )

        print(
            f"Priority: {row['failure_priority']:.2f}"
        )

        print(
            f"Gold intent: "
            f"{row['gold_intent']}"
        )

        print(
            f"Predicted intent: "
            f"{row['predicted_intent']}"
        )

        if "gold_escalation" in row:
            print(
                f"Gold escalation: "
                f"{row['gold_escalation']}"
            )

        if "predicted_escalation" in row:
            print(
                f"Predicted escalation: "
                f"{row['predicted_escalation']}"
            )

        print()
        print(
            f"Customer:"
        )

        print(
            row["customer_text"]
        )

        print()
        print(
            f"Generated reply:"
        )

        print(
            row["generated_reply"]
        )

        print()
        print(
            "Scores:"
        )

        print(
            f"  Relevance:             "
            f"{row['relevance']}"
        )

        print(
            f"  Groundedness:          "
            f"{row['groundedness']}"
        )

        print(
            f"  Helpfulness:           "
            f"{row['helpfulness']}"
        )

        print(
            f"  Safety:                "
            f"{row['safety']}"
        )

        print(
            f"  Historical grounding:  "
            f"{row['historical_grounding']}"
        )

        print(
            f"  Overall:               "
            f"{row['overall_mean']}"
        )

        print()
        print(
            f"Judge reason:"
        )

        # print(
        #     row["reason"]
        # )

        judge_reason = ""

        if "reason_x" in row.index:
            judge_reason = row["reason_x"]
        elif "reason_y" in row.index:
            judge_reason = row["reason_y"]
        elif "reason" in row.index:
            judge_reason = row["reason"]

        print(
            judge_reason
        )

    # ---------------------------------------------------------
    # Category summary
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("FAILURE CATEGORY SUMMARY")
    print("========================================")

    category_counts = (
        df["failure_category"]
        .value_counts()
    )

    print(
        category_counts.to_string()
    )

    # ---------------------------------------------------------
    # Intent confusion
    # ---------------------------------------------------------

    intent_errors = df[
        df["intent_error"]
    ]

    print()
    print("========================================")
    print("INTENT ERROR SUMMARY")
    print("========================================")

    print(
        f"Intent errors: "
        f"{len(intent_errors)}"
    )

    if len(intent_errors) > 0:

        confusion = (
            intent_errors[
                [
                    "gold_intent",
                    "predicted_intent",
                ]
            ]
            .value_counts()
            .head(15)
        )

        print()
        print(
            "Most common confusion pairs:"
        )

        print(
            confusion.to_string()
        )

    # ---------------------------------------------------------
    # Response quality summary
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("RESPONSE QUALITY FAILURES")
    print("========================================")

    print(
        f"Low relevance (<=2): "
        f"{df['low_relevance'].sum()}"
    )

    print(
        f"Low groundedness (<=2): "
        f"{df['low_groundedness'].sum()}"
    )

    print(
        f"Low helpfulness (<=2): "
        f"{df['low_helpfulness'].sum()}"
    )

    print(
        f"Low safety (<=2): "
        f"{df['low_safety'].sum()}"
    )

    print(
        f"Low historical grounding (<=2): "
        f"{df['low_historical_grounding'].sum()}"
    )

    print(
        f"Overall quality <3.5: "
        f"{df['response_quality_low'].sum()}"
    )

    # ---------------------------------------------------------
    # Escalation summary
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("ESCALATION FAILURE SUMMARY")
    print("========================================")

    print(
        f"Unsafe auto-handles: "
        f"{df['unsafe_auto_handle'].sum()}"
    )

    print(
        f"Unnecessary escalations: "
        f"{df['unnecessary_escalation'].sum()}"
    )

    print(
        f"Total escalation errors: "
        f"{df['escalation_error'].sum()}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    columns_to_save = [
        "tweet_id",
        "customer_text",
        "gold_intent",
        "predicted_intent",
        "intent_error",
        "generated_reply",
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "historical_grounding",
        "overall_mean",
        "failure_category",
        "failure_priority",
        "low_relevance",
        "low_groundedness",
        "low_helpfulness",
        "low_safety",
        "low_historical_grounding",
        "response_quality_low",
        "gold_escalation",
        "predicted_escalation",
        "escalation_error",
        "unsafe_auto_handle",
        "unnecessary_escalation",
        "reason",
    ]

    columns_to_save = [
        c
        for c in columns_to_save
        if c in df.columns
    ]

    df[
        columns_to_save
    ].sort_values(
        [
            "failure_priority",
            "overall_mean",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        output_path,
        index=False,
    )

    print()
    print("========================================")
    print("FAILURE ANALYSIS COMPLETE")
    print("========================================")

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()