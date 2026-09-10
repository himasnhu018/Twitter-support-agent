import argparse
from pathlib import Path

import pandas as pd


def pct(value):
    return f"{value * 100:.1f}%"


def fmt(value, digits=3):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_intent_results(root):
    path = (
        root
        / "data"
        / "golden"
        / "gemini_classifier_predictions.csv"
    )

    df = pd.read_csv(path)

    gold = find_column(
        df,
        ["gold_intent", "intent"],
    )

    predicted = find_column(
        df,
        ["predicted_intent", "intent_prediction"],
    )

    if gold is None or predicted is None:
        raise ValueError(
            f"Could not identify intent columns in {path}. "
            f"Columns: {list(df.columns)}"
        )

    correct = (
        df[gold].astype(str)
        == df[predicted].astype(str)
    )

    accuracy = correct.mean()

    # Calculate macro F1 without sklearn dependency.
    labels = sorted(
        set(df[gold].astype(str))
        | set(df[predicted].astype(str))
    )

    f1_values = []

    for label in labels:

        tp = (
            (df[gold] == label)
            & (df[predicted] == label)
        ).sum()

        fp = (
            (df[gold] != label)
            & (df[predicted] == label)
        ).sum()

        fn = (
            (df[gold] == label)
            & (df[predicted] != label)
        ).sum()

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values)

    return {
        "n": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def load_baseline_results(root):
    results = {}

    # These files are the scripts' evaluation outputs.
    # We calculate metrics directly from the prediction files
    # when possible.

    candidates = [
        (
            "Semantic baseline",
            root
            / "data"
            / "golden"
            / "semantic_classifier_predictions.csv",
        ),
        (
            "Rule baseline",
            root
            / "data"
            / "golden"
            / "rule_baseline_predictions.csv",
        ),
    ]

    for name, path in candidates:

        if not path.exists():
            continue

        df = pd.read_csv(path)

        gold = find_column(
            df,
            ["gold_intent", "intent"],
        )

        predicted = find_column(
            df,
            ["predicted_intent", "prediction"],
        )

        if gold is None or predicted is None:
            continue

        correct = (
            df[gold].astype(str)
            == df[predicted].astype(str)
        )

        labels = sorted(
            set(df[gold].astype(str))
            | set(df[predicted].astype(str))
        )

        f1_values = []

        for label in labels:

            tp = (
                (df[gold] == label)
                & (df[predicted] == label)
            ).sum()

            fp = (
                (df[gold] != label)
                & (df[predicted] == label)
            ).sum()

            fn = (
                (df[gold] == label)
                & (df[predicted] != label)
            ).sum()

            precision = (
                tp / (tp + fp)
                if tp + fp > 0
                else 0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn > 0
                else 0
            )

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (
                    2
                    * precision
                    * recall
                    / (precision + recall)
                )

            f1_values.append(f1)

        results[name] = {
            "n": len(df),
            "accuracy": correct.mean(),
            "macro_f1": sum(f1_values) / len(f1_values),
        }

    return results


def load_retrieval_results(root):
    path = (
        root
        / "data"
        / "golden"
        / "retrieval_evaluation_results.csv"
    )

    df = pd.read_csv(path)

    similarity = find_column(
        df,
        ["similarity", "score"],
    )

    rank = find_column(
        df,
        ["rank"],
    )

    if similarity is None:
        raise ValueError(
            "Retrieval similarity column not found."
        )

    df[similarity] = pd.to_numeric(
        df[similarity],
        errors="coerce",
    )

    if rank is not None:

        df[rank] = pd.to_numeric(
            df[rank],
            errors="coerce",
        )

        top1 = df[
            df[rank] == 1
        ][similarity]

        top5 = df[
            df[rank] <= 5
        ][similarity]

    else:

        top1 = df[similarity]
        top5 = df[similarity]

    return {
        "n_queries": top1.nunique()
        if rank is None
        else len(top1),
        "top1_mean": top1.mean(),
        "top5_mean": top5.mean(),
    }


def load_response_results(root):
    path = (
        root
        / "data"
        / "golden"
        / "response_judge_predictions.csv"
    )

    df = pd.read_csv(path)

    dimensions = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "historical_grounding",
    ]

    result = {
        "n": len(df),
    }

    for column in dimensions:

        if column not in df.columns:
            continue

        result[
            f"{column}_mean"
        ] = df[column].mean()

        result[
            f"{column}_ge4"
        ] = (
            df[column] >= 4
        ).mean()

    if "overall_mean" in df.columns:

        result["overall_mean"] = (
            df["overall_mean"].mean()
        )

    if "safety" in df.columns:

        result["safety_le2"] = (
            df["safety"] <= 2
        ).sum()

    if "overall_mean" in df.columns:

        result["overall_lt3_5"] = (
            df["overall_mean"] < 3.5
        ).sum()

    return result


def load_escalation_results(root):
    path = (
        root
        / "data"
        / "golden"
        / "escalation_predictions.csv"
    )

    df = pd.read_csv(path)

    gold = find_column(
        df,
        [
            "gold_escalation",
            "gold_decision",
        ],
    )

    predicted = find_column(
        df,
        [
            "predicted_escalation",
            "predicted_decision",
            "decision",
        ],
    )

    if gold is None or predicted is None:
        raise ValueError(
            f"Could not find escalation columns. "
            f"Columns: {list(df.columns)}"
        )

    gold = df[gold].astype(str)
    predicted = df[predicted].astype(str)

    correct = (
        gold == predicted
    )

    escalate_gold = (
        gold == "ESCALATE"
    )

    escalate_predicted = (
        predicted == "ESCALATE"
    )

    unsafe_auto = (
        escalate_gold
        & ~escalate_predicted
    )

    auto_gold = (
        gold == "AUTO_HANDLE"
    )

    unnecessary_escalation = (
        auto_gold
        & escalate_predicted
    )

    escalation_recall = (
        (
            escalate_gold
            & escalate_predicted
        ).sum()
        / escalate_gold.sum()
        if escalate_gold.sum() > 0
        else 0
    )

    return {
        "n": len(df),
        "accuracy": correct.mean(),
        "escalation_recall": escalation_recall,
        "unsafe_auto_handle_rate": (
            unsafe_auto.sum()
            / escalate_gold.sum()
            if escalate_gold.sum() > 0
            else 0
        ),
        "auto_handle_precision": (
            (
                auto_gold
                & ~(
                    predicted
                    != "AUTO_HANDLE"
                )
            ).sum()
            / (
                predicted == "AUTO_HANDLE"
            ).sum()
            if (
                predicted == "AUTO_HANDLE"
            ).sum() > 0
            else 0
        ),
        "unsafe_auto_handles": int(
            unsafe_auto.sum()
        ),
        "unnecessary_escalations": int(
            unnecessary_escalation.sum()
        ),
    }


def load_failure_results(root):
    path = (
        root
        / "data"
        / "golden"
        / "failure_analysis.csv"
    )

    df = pd.read_csv(path)

    return {
        "intent_errors": int(
            df["intent_error"].sum()
        ),
        "unsafe_auto_handles": int(
            df["unsafe_auto_handle"].sum()
        ),
        "unnecessary_escalations": int(
            df["unnecessary_escalation"].sum()
        ),
        "low_relevance": int(
            df["low_relevance"].sum()
        ),
        "low_groundedness": int(
            df["low_groundedness"].sum()
        ),
        "low_helpfulness": int(
            df["low_helpfulness"].sum()
        ),
        "low_safety": int(
            df["low_safety"].sum()
        ),
        "low_historical_grounding": int(
            df["low_historical_grounding"].sum()
        ),
        "response_quality_low": int(
            df["response_quality_low"].sum()
        ),
    }


def load_human_agreement(root):
    path = (
        root
        / "data"
        / "golden"
        / "human_agreement_results.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    print()
    print("========================================")
    print("FINAL EVALUATION")
    print("========================================")

    # ---------------------------------------------------------
    # Load everything
    # ---------------------------------------------------------

    intent = load_intent_results(root)

    print()
    print(
        f"Gemini intent evaluation: "
        f"{intent['n']} examples"
    )

    baselines = load_baseline_results(root)

    retrieval = load_retrieval_results(root)

    response = load_response_results(root)

    escalation = load_escalation_results(root)

    failures = load_failure_results(root)

    human = load_human_agreement(root)

    # ---------------------------------------------------------
    # Metrics table
    # ---------------------------------------------------------

    rows = []

    rows.append(
        {
            "system": "Gemini intent classifier",
            "evaluation": "Intent classification",
            "metric": "Accuracy",
            "value": intent["accuracy"],
            "unit": "fraction",
        }
    )

    rows.append(
        {
            "system": "Gemini intent classifier",
            "evaluation": "Intent classification",
            "metric": "Macro F1",
            "value": intent["macro_f1"],
            "unit": "fraction",
        }
    )

    for name, values in baselines.items():

        rows.append(
            {
                "system": name,
                "evaluation": "Intent classification",
                "metric": "Accuracy",
                "value": values["accuracy"],
                "unit": "fraction",
            }
        )

        rows.append(
            {
                "system": name,
                "evaluation": "Intent classification",
                "metric": "Macro F1",
                "value": values["macro_f1"],
                "unit": "fraction",
            }
        )

    rows.extend(
        [
            {
                "system": "FAISS + MiniLM",
                "evaluation": "Retrieval",
                "metric": "Top-1 mean cosine similarity",
                "value": retrieval["top1_mean"],
                "unit": "score",
            },
            {
                "system": "FAISS + MiniLM",
                "evaluation": "Retrieval",
                "metric": "Top-5 mean cosine similarity",
                "value": retrieval["top5_mean"],
                "unit": "score",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Overall judge mean",
                "value": response.get(
                    "overall_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Relevance mean",
                "value": response.get(
                    "relevance_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Groundedness mean",
                "value": response.get(
                    "groundedness_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Helpfulness mean",
                "value": response.get(
                    "helpfulness_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Safety mean",
                "value": response.get(
                    "safety_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini + retrieval",
                "evaluation": "Response",
                "metric": "Historical grounding mean",
                "value": response.get(
                    "historical_grounding_mean",
                    float("nan"),
                ),
                "unit": "1-5",
            },
            {
                "system": "Gemini escalation",
                "evaluation": "Escalation",
                "metric": "Accuracy",
                "value": escalation["accuracy"],
                "unit": "fraction",
            },
            {
                "system": "Gemini escalation",
                "evaluation": "Escalation",
                "metric": "Escalation recall",
                "value": escalation["escalation_recall"],
                "unit": "fraction",
            },
            {
                "system": "Gemini escalation",
                "evaluation": "Escalation",
                "metric": "Unsafe auto-handle rate",
                "value": escalation[
                    "unsafe_auto_handle_rate"
                ],
                "unit": "fraction",
            },
        ]
    )

    summary_df = pd.DataFrame(rows)

    summary_path = (
        root
        / "data"
        / "golden"
        / "final_evaluation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------

    report_path = (
        root
        / "data"
        / "golden"
        / "final_evaluation_report.md"
    )

    lines = []

    lines.append(
        "# Final Evaluation Report"
    )

    lines.append("")

    lines.append(
        "## Evaluation setup"
    )

    lines.append("")

    lines.append(
        "- Brand: AmazonHelp"
    )

    lines.append(
        "- Golden set: 200 hand-labeled customer messages"
    )

    lines.append(
        "- Intent taxonomy: 14 labels including OTHER"
    )

    lines.append(
        "- Retrieval: FAISS IndexFlatIP over normalized "
        "all-MiniLM-L6-v2 embeddings"
    )

    lines.append(
        "- Response model: Gemini 3.5 Flash-Lite"
    )

    lines.append(
        "- Evaluation uses a leakage-safe retrieval corpus "
        "with golden conversation components removed."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Intent
    # ---------------------------------------------------------

    lines.append(
        "## Intent classification"
    )

    lines.append("")

    lines.append(
        "| System | Accuracy | Macro F1 |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    lines.append(
        f"| Gemini 3.5 Flash-Lite | "
        f"{pct(intent['accuracy'])} | "
        f"{pct(intent['macro_f1'])} |"
    )

    if baselines:

        for name in [
            "Semantic baseline",
            "Rule baseline",
        ]:

            if name in baselines:

                value = baselines[name]

                lines.append(
                    f"| {name} | "
                    f"{pct(value['accuracy'])} | "
                    f"{pct(value['macro_f1'])} |"
                )

    lines.append("")

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    lines.append(
        "## Retrieval"
    )

    lines.append("")

    lines.append(
        f"- Top-1 mean cosine similarity: "
        f"{retrieval['top1_mean']:.4f}"
    )

    lines.append(
        f"- Top-5 mean cosine similarity: "
        f"{retrieval['top5_mean']:.4f}"
    )

    lines.append("")

    lines.append(
        "**Important:** these are retrieval similarity "
        "diagnostics, not retrieval accuracy. The corpus "
        "does not contain gold intent labels for the "
        "retrieved cases."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    lines.append(
        "## Response quality"
    )

    lines.append("")

    lines.append(
        "| Dimension | Mean / 5 | % >= 4 |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    for dimension in [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "historical_grounding",
    ]:

        mean_value = response.get(
            f"{dimension}_mean",
            float("nan"),
        )

        ge4 = response.get(
            f"{dimension}_ge4",
            float("nan"),
        )

        lines.append(
            f"| {dimension.replace('_', ' ').title()} | "
            f"{fmt(mean_value)} | "
            f"{pct(ge4) if not pd.isna(ge4) else 'N/A'} |"
        )

    lines.append(
        f"| **Overall** | "
        f"**{fmt(response.get('overall_mean', float('nan')))}** | - |"
    )

    lines.append("")

    lines.append(
        f"Safety scores <=2: "
        f"{response.get('safety_le2', 'N/A')}/200"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Escalation
    # ---------------------------------------------------------

    lines.append(
        "## Escalation"
    )

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Accuracy | {pct(escalation['accuracy'])} |"
    )

    lines.append(
        f"| Escalation recall | "
        f"{pct(escalation['escalation_recall'])} |"
    )

    lines.append(
        f"| Unsafe auto-handle rate | "
        f"{pct(escalation['unsafe_auto_handle_rate'])} |"
    )

    lines.append(
        f"| Unsafe auto-handles | "
        f"{escalation['unsafe_auto_handles']}/200 |"
    )

    lines.append(
        f"| Unnecessary escalations | "
        f"{escalation['unnecessary_escalations']}/200 |"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Failure analysis
    # ---------------------------------------------------------

    lines.append(
        "## Failure analysis"
    )

    lines.append("")

    lines.append(
        f"- Intent errors: "
        f"{failures['intent_errors']}/200 "
        f"({pct(failures['intent_errors'] / 200)})"
    )

    lines.append(
        f"- Unsafe auto-handles: "
        f"{failures['unsafe_auto_handles']}/200 "
        f"({pct(failures['unsafe_auto_handles'] / 200)})"
    )

    lines.append(
        f"- Unnecessary escalations: "
        f"{failures['unnecessary_escalations']}/200 "
        f"({pct(failures['unnecessary_escalations'] / 200)})"
    )

    lines.append(
        f"- Low helpfulness (<=2): "
        f"{failures['low_helpfulness']}/200 "
        f"({pct(failures['low_helpfulness'] / 200)})"
    )

    lines.append(
        f"- Low relevance (<=2): "
        f"{failures['low_relevance']}/200 "
        f"({pct(failures['low_relevance'] / 200)})"
    )

    lines.append(
        f"- Low groundedness (<=2): "
        f"{failures['low_groundedness']}/200 "
        f"({pct(failures['low_groundedness'] / 200)})"
    )

    lines.append("")

    lines.append(
        "The five concrete examples selected by the failure "
        "analysis should be documented separately with the "
        "customer message, generated reply, failure type, "
        "and proposed mitigation."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Human agreement
    # ---------------------------------------------------------

    lines.append(
        "## Human vs LLM-judge agreement"
    )

    lines.append("")

    if human is None:

        lines.append(
            "Human agreement results were not found."
        )

    else:

        lines.append(
            "| Dimension | Judge mean | Human mean | "
            "Exact agreement | Within-1 agreement |"
        )

        lines.append(
            "|---|---:|---:|---:|---:|"
        )

        for _, row in human.iterrows():

            lines.append(
                f"| {row['dimension']} | "
                f"{row['judge_mean']:.3f} | "
                f"{row['human_mean']:.3f} | "
                f"{row['exact_agreement'] * 100:.1f}% | "
                f"{row['within_1_agreement'] * 100:.1f}% |"
            )

    lines.append("")

    # ---------------------------------------------------------
    # Misleading headline number
    # ---------------------------------------------------------

    lines.append(
        "## What is misleading about my headline number?"
    )

    lines.append("")

    lines.append(
        "The 66.5% intent accuracy should not be interpreted "
        "as production accuracy. It is measured on a 200-example "
        "golden set, the first version of the taxonomy, and an "
        "English-focused sample. The dataset itself is historical "
        "Twitter support data, not a live production stream."
    )

    lines.append("")

    lines.append(
        "The response-quality scores are also LLM-as-judge "
        "measurements rather than direct customer satisfaction. "
        "A 4.553/5 mean therefore does not mean 91% of real "
        "customers would be satisfied."
    )

    lines.append("")

    lines.append(
        "Retrieval cosine similarity is a relevance diagnostic, "
        "not retrieval accuracy. High similarity can still "
        "correspond to a historically similar but operationally "
        "incorrect case."
    )

    lines.append("")

    lines.append(
        "Finally, escalation accuracy hides an asymmetric risk: "
        "an unsafe auto-handle can be substantially worse than "
        "an unnecessary escalation. For that reason, escalation "
        "recall and unsafe auto-handle rate are reported separately."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    lines.append(
        "## Reproducibility"
    )

    lines.append("")

    lines.append(
        "The headline evaluation should reuse the persisted "
        "FAISS index and cached model outputs rather than "
        "rebuilding embeddings. The one-time CPU embedding "
        "construction is not included in the headline evaluation "
        "runtime."
    )

    lines.append("")

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("FINAL EVALUATION COMPLETE")
    print("========================================")

    print()
    print("INTENT")
    print(
        f"  Gemini accuracy: "
        f"{pct(intent['accuracy'])}"
    )
    print(
        f"  Gemini macro F1: "
        f"{pct(intent['macro_f1'])}"
    )

    for name, values in baselines.items():
        print(
            f"  {name} accuracy: "
            f"{pct(values['accuracy'])}"
        )
        print(
            f"  {name} macro F1: "
            f"{pct(values['macro_f1'])}"
        )

    print()
    print("RETRIEVAL")
    print(
        f"  Top-1 mean similarity: "
        f"{retrieval['top1_mean']:.4f}"
    )
    print(
        f"  Top-5 mean similarity: "
        f"{retrieval['top5_mean']:.4f}"
    )

    print()
    print("RESPONSE")
    print(
        f"  Overall judge mean: "
        f"{response.get('overall_mean', float('nan')):.3f}/5"
    )
    print(
        f"  Helpfulness: "
        f"{response.get('helpfulness_mean', float('nan')):.3f}/5"
    )
    print(
        f"  Safety: "
        f"{response.get('safety_mean', float('nan')):.3f}/5"
    )

    print()
    print("ESCALATION")
    print(
        f"  Accuracy: "
        f"{pct(escalation['accuracy'])}"
    )
    print(
        f"  Escalation recall: "
        f"{pct(escalation['escalation_recall'])}"
    )
    print(
        f"  Unsafe auto-handle rate: "
        f"{pct(escalation['unsafe_auto_handle_rate'])}"
    )

    print()
    print("FILES")
    print(
        f"  {summary_path}"
    )
    print(
        f"  {report_path}"
    )


if __name__ == "__main__":
    main()