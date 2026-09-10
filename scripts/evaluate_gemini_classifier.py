import json
import os
import time

import pandas as pd
from google import genai
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


GOLDEN_PATH = "data/golden/golden_set.csv"
CACHE_PATH = "data/golden/gemini_classifier_cache.json"
OUTPUT_PATH = "data/golden/gemini_classifier_predictions.csv"

MODEL_NAME = "gemini-3.5-flash-lite"


INTENTS = [
    "DELIVERY_DELAY",
    "DELIVERY_TRACKING",
    "DELIVERY_FAILURE",
    "ORDER_CHANGE",
    "RETURN",
    "REFUND",
    "PAYMENT",
    "PRIME",
    "TECHNICAL",
    "DIGITAL_CONTENT",
    "SELLER_FRAUD",
    "ORDER_PRODUCT",
    "SUPPORT_COMPLAINT",
    "OTHER",
]


INTENT_DEFINITIONS = """
DELIVERY_DELAY:
The customer says an order or package is late, delayed, overdue,
or has not arrived by the promised delivery date.

DELIVERY_TRACKING:
The customer primarily wants to know where an order/package is,
or asks about tracking information, carrier status, or delivery status.

DELIVERY_FAILURE:
A delivery attempt failed, was missed, was unsuccessful,
was delivered to the wrong place, or the carrier could not complete delivery.

ORDER_CHANGE:
The customer wants to change, cancel, modify, or update an order,
delivery address, shipping option, or other order detail.

RETURN:
The customer wants to return an item or asks about return eligibility,
return procedure, or return shipping.

REFUND:
The customer asks about a refund, missing refund, refund status,
or refund timing.

PAYMENT:
The customer has a payment, billing, charge, card, or checkout payment problem.

PRIME:
The customer asks about Amazon Prime membership, subscription,
Prime benefits, or Prime eligibility.

TECHNICAL:
The customer reports a technical malfunction, error, broken feature,
website/application/device problem, or playback malfunction.

DIGITAL_CONTENT:
The customer asks an informational question about Prime Video, movies,
shows, Kindle, ebooks, digital content, availability, language, or access
without reporting a technical malfunction.

SELLER_FRAUD:
The customer reports fraud, a scam, suspicious seller behavior,
impersonation, or a potentially fraudulent transaction/contact.

ORDER_PRODUCT:
The customer asks about a specific product or order item,
including product availability, pricing, specifications, or product information.

SUPPORT_COMPLAINT:
The primary issue is dissatisfaction with Amazon customer support,
a previous agent, lack of help, or repeated failure of support to resolve the issue.

OTHER:
The message does not contain a clear actionable support intent,
is casual conversation, praise, thanks, irrelevant content,
or has insufficient information to determine an intent.
"""


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reason: str


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )


def classify(client, text):
    prompt = f"""
You are an intent classifier for an Amazon customer support system.

Classify the customer message into EXACTLY ONE of these intents:

{", ".join(INTENTS)}

Use these definitions:

{INTENT_DEFINITIONS}

Important rules:

1. Return exactly one intent.
2. Classify the customer's PRIMARY actionable issue.
3. Emotion, anger, profanity, or politeness does not determine the intent.
4. Account-specific information is not itself an intent.
5. If the customer has multiple issues, choose the primary actionable issue.
6. If the message is resolved, casual, praise/thanks, irrelevant, or lacks
   enough information for a support intent, use OTHER.
7. Do not invent information that is not in the customer message.

Customer message:

{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": IntentResult,
            "temperature": 0,
        },
    )

    result = IntentResult.model_validate_json(
        response.text
    )

    # Safety validation
    if result.intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned: {result.intent}"
        )

    return result


def main():

    print("=" * 80)
    print("GEMINI INTENT CLASSIFIER EVALUATION")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    client = genai.Client(api_key=api_key)

    df = pd.read_csv(GOLDEN_PATH)

    print(f"\nGolden examples: {len(df)}")

    cache = load_cache()

    print(f"Cached predictions: {len(cache)}")

    predictions = []

    for position, row in df.iterrows():

        tweet_id = str(row["tweet_id"])
        text = str(row["text"])

        if tweet_id in cache:

            result = cache[tweet_id]

        else:

            print(
                f"\n[{position + 1}/{len(df)}] "
                f"Classifying tweet {tweet_id}"
            )

            print(f"Text: {text[:120]}")

            # Retry a few times for transient API errors.
            last_error = None

            for attempt in range(3):

                try:
                    result_obj = classify(
                        client,
                        text,
                    )

                    result = {
                        "intent": result_obj.intent,
                        "confidence": result_obj.confidence,
                        "reason": result_obj.reason,
                    }

                    cache[tweet_id] = result
                    save_cache(cache)

                    break

                except Exception as exc:

                    last_error = exc

                    print(
                        f"Attempt {attempt + 1}/3 failed: "
                        f"{exc}"
                    )

                    time.sleep(
                        2 ** attempt
                    )

            else:
                raise RuntimeError(
                    f"Failed to classify {tweet_id}"
                ) from last_error

            # Small delay to be polite to the API.
            time.sleep(0.2)

        predictions.append(result)

        print(
            f"Prediction: {result['intent']} "
            f"(confidence={result['confidence']:.2f})"
        )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    y_true = (
        df["gold_intent"]
        .astype(str)
        .tolist()
    )

    y_pred = [
        result["intent"]
        for result in predictions
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 80)
    print("HEADLINE METRICS")
    print("=" * 80)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print()
    print("=" * 80)
    print("PER-INTENT PERFORMANCE")
    print("=" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=INTENTS,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output = df.copy()

    output["predicted_intent"] = [
        result["intent"]
        for result in predictions
    ]

    output["gemini_confidence"] = [
        result["confidence"]
        for result in predictions
    ]

    output["gemini_reason"] = [
        result["reason"]
        for result in predictions
    ]

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()