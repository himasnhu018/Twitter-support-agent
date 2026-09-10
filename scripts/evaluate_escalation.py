import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MODEL_NAME = "gemini-3.5-flash-lite"

VALID_DECISIONS = {
    "AUTO_HANDLE",
    "ESCALATE",
}


class EscalationItem(BaseModel):
    tweet_id: str
    decision: str
    confidence: float
    reason: str


class EscalationBatch(BaseModel):
    results: list[EscalationItem]


def normalize_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def load_golden():
    path = Path("data/golden/golden_set.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "tweet_id",
        "text",
        "gold_escalation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in golden set: {sorted(missing)}"
        )

    df["tweet_id"] = df["tweet_id"].map(normalize_id)
    df["text"] = df["text"].fillna("").astype(str)
    df["gold_escalation"] = (
        df["gold_escalation"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


def load_cache(cache_path):
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            return {}

        return cache

    except Exception:
        print("Warning: cache could not be loaded. Starting fresh.")
        return {}


def save_cache(cache, cache_path):
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = cache_path.with_suffix(".tmp")

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(cache_path)


def build_prompt(batch):
    examples = []

    for i, row in enumerate(batch, start=1):
        examples.append(
            f"""
CASE {i}
Tweet ID: {row["tweet_id"]}

Customer message:
{row["text"]}
""".strip()
        )

    cases = "\n\n".join(examples)

    return f"""
You are evaluating whether Amazon customer-support messages
should be automatically handled or escalated to a human support agent.

Classify EVERY case into exactly one:

AUTO_HANDLE
ESCALATE

Definitions:

AUTO_HANDLE:
The customer's immediate need can be safely addressed with general
information or safe troubleshooting, without accessing or changing
their specific account, order, shipment, payment, refund, or transaction.

ESCALATE:
The customer requires account/order-specific investigation, a
consequential action, financial investigation, fraud investigation,
delivery investigation, sensitive-information handling, or human
judgment.

Escalate when:
- The customer wants Amazon to inspect their specific order/shipment.
- The customer asks where their specific package/refund/payment is.
- A delayed/missing delivery needs investigation.
- A refund/payment/charge needs transaction investigation.
- The customer asks to cancel, modify, or change something.
- Fraud, scams, counterfeit products, or suspicious seller activity
  are reported.
- Reasonable troubleshooting has already failed.
- Sensitive personal information is involved.
- Available context is insufficient to safely resolve the issue.
- The situation is ambiguous enough that automation could mislead.

AUTO_HANDLE examples:
- General how-to questions.
- General product/service information.
- Basic troubleshooting that does not require account access.
- General information about how to use a feature.
- Safe, non-consequential guidance.

Important:
- Do NOT escalate merely because the customer is angry.
- Do NOT auto-handle merely because similar historical responses exist.
- Historical support responses are not authoritative current policy.
- A generic instruction such as "check your tracking" does NOT make
  a customer-specific delivery problem AUTO_HANDLE.
- When uncertain, prefer ESCALATE.

Return exactly one result for every case.
Preserve the Tweet ID exactly.

CASES:

{cases}
""".strip()


def classify_batch(client, batch):
    prompt = build_prompt(batch)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=EscalationBatch,
        ),
    )

    result = EscalationBatch.model_validate_json(
        response.text
    )

    output = {}

    for item in result.results:

        tweet_id = normalize_id(item.tweet_id)

        decision = item.decision.strip().upper()

        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision for {tweet_id}: {decision}"
            )

        confidence = max(
            0.0,
            min(1.0, float(item.confidence)),
        )

        output[tweet_id] = {
            "decision": decision,
            "confidence": confidence,
            "reason": item.reason.strip(),
        }

    expected_ids = {
        normalize_id(row["tweet_id"])
        for row in batch
    }

    returned_ids = set(output.keys())

    missing = expected_ids - returned_ids

    if missing:
        raise ValueError(
            f"Gemini did not return results for: {sorted(missing)}"
        )

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="Seconds between Gemini requests.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    golden = load_golden()

    cache_path = Path(
        "data/golden/escalation_classifier_cache.json"
    )

    output_path = Path(
        "data/golden/escalation_predictions.csv"
    )

    cache = load_cache(cache_path)

    print("=" * 80)
    print("GEMINI ESCALATION EVALUATION")
    print("=" * 80)

    print()
    print(f"Model:             {MODEL_NAME}")
    print(f"Golden examples:   {len(golden)}")
    print(f"Cached predictions: {len(cache)}")
    print(f"Batch size:        {args.batch_size}")
    print()

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    rows = golden.to_dict("records")

    batches = [
        rows[i:i + args.batch_size]
        for i in range(
            0,
            len(rows),
            args.batch_size,
        )
    ]

    api_calls = 0

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        uncached = [
            row
            for row in batch
            if normalize_id(row["tweet_id"]) not in cache
        ]

        if not uncached:
            print(
                f"[{batch_number}/{len(batches)}] "
                f"Fully cached — skipping API call."
            )
            continue

        print(
            f"[{batch_number}/{len(batches)}] "
            f"Classifying {len(uncached)} cases..."
        )

        # Respect the user's 20 RPM quota.
        if api_calls > 0:
            time.sleep(args.sleep)

        success = False

        for attempt in range(3):

            try:

                predictions = classify_batch(
                    client,
                    uncached,
                )

                for tweet_id, prediction in predictions.items():
                    cache[tweet_id] = prediction

                save_cache(
                    cache,
                    cache_path,
                )

                api_calls += 1
                success = True

                break

            except Exception as e:

                print(
                    f"  Attempt {attempt + 1}/3 failed: {e}"
                )

                if attempt < 2:
                    print(
                        "  Waiting 15 seconds before retry..."
                    )
                    time.sleep(15)

        if not success:
            raise RuntimeError(
                f"Batch {batch_number} failed after 3 attempts."
            )

    # ------------------------------------------------------------------
    # Build evaluation dataframe
    # ------------------------------------------------------------------

    predictions = []

    for _, row in golden.iterrows():

        tweet_id = normalize_id(
            row["tweet_id"]
        )

        if tweet_id not in cache:
            raise RuntimeError(
                f"No prediction for tweet {tweet_id}"
            )

        prediction = cache[tweet_id]

        predictions.append(
            {
                "tweet_id": tweet_id,
                "text": row["text"],
                "gold_escalation": row["gold_escalation"],
                "predicted_escalation": prediction["decision"],
                "confidence": prediction["confidence"],
                "reason": prediction["reason"],
            }
        )

    result_df = pd.DataFrame(predictions)

    result_df.to_csv(
        output_path,
        index=False,
    )

    y_true = result_df[
        "gold_escalation"
    ]

    y_pred = result_df[
        "predicted_escalation"
    ]

    labels = [
        "AUTO_HANDLE",
        "ESCALATE",
    ]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    # Here:
    # gold AUTO_HANDLE + predicted ESCALATE = false escalation
    # gold ESCALATE + predicted AUTO_HANDLE = unsafe auto-handle

    unsafe_auto_handle_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    escalation_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    auto_handle_precision = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("HEADLINE ESCALATION METRICS")
    print("=" * 80)

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print()
    print("=" * 80)
    print("PER-DECISION PERFORMANCE")
    print("=" * 80)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    print(report)

    print("=" * 80)
    print("SAFETY-RELEVANT METRICS")
    print("=" * 80)

    print(
        f"Escalation recall:       "
        f"{escalation_recall:.4f}"
    )

    print(
        f"Unsafe AUTO_HANDLE rate: "
        f"{unsafe_auto_handle_rate:.4f}"
    )

    print(
        f"AUTO_HANDLE precision:    "
        f"{auto_handle_precision:.4f}"
    )

    print()
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        "                 Predicted"
    )

    print(
        "                 AUTO   ESCALATE"
    )

    print(
        f"Gold AUTO       {tn:4d}   {fp:4d}"
    )

    print(
        f"Gold ESCALATE   {fn:4d}   {tp:4d}"
    )

    print()
    print("=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"Predictions: {output_path}"
    )

    print(
        f"Cache:       {cache_path}"
    )

    print(
        f"API calls this run: {api_calls}"
    )


if __name__ == "__main__":
    main()