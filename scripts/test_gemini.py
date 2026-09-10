import os
from google import genai
from pydantic import BaseModel


MODEL_NAME = "gemini-3.5-flash-lite"


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reason: str


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = """
You are a customer support intent classifier for AmazonHelp.

Classify this customer message into exactly one of these intents:

DELIVERY_DELAY
DELIVERY_TRACKING
DELIVERY_FAILURE
ORDER_CHANGE
RETURN
REFUND
PAYMENT
PRIME
TECHNICAL
DIGITAL_CONTENT
SELLER_FRAUD
ORDER_PRODUCT
SUPPORT_COMPLAINT
OTHER

Definitions:

DELIVERY_DELAY:
Order/package is late, delayed, overdue, or has not arrived by the expected date.

DELIVERY_TRACKING:
Customer primarily asks where an order/package is or asks about tracking/carrier status.

DELIVERY_FAILURE:
Delivery attempt failed, package was not successfully delivered, wrong delivery location,
or carrier could not complete delivery.

ORDER_CHANGE:
Customer wants to change, cancel, modify, or update an order or delivery detail.

RETURN:
Customer wants to return an item or asks about the return process/eligibility.

REFUND:
Customer asks about a refund, missing refund, refund status, or refund timing.

PAYMENT:
Payment, billing, card, charge, or checkout payment problem.

PRIME:
Amazon Prime membership, subscription, benefits, or Prime eligibility.

TECHNICAL:
A technical malfunction, error, broken feature, application/website/device problem,
or playback malfunction.

DIGITAL_CONTENT:
Informational question about Prime Video, movies, shows, Kindle, ebooks,
digital content, availability, language, or access without a technical malfunction.

SELLER_FRAUD:
Fraud, scam, suspicious seller behavior, impersonation, or fraudulent contact/transaction.

ORDER_PRODUCT:
Question about a specific product/order item, availability, pricing, specifications,
or product information.

SUPPORT_COMPLAINT:
The primary issue is dissatisfaction with customer support, an agent, or repeated failure
of support to resolve the issue.

OTHER:
No clear actionable support intent, casual conversation, praise, thanks, irrelevant content,
or insufficient information.

Customer message:
"My package is two days late and still hasn't arrived"
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": IntentResult,
        },
    )

    result = IntentResult.model_validate_json(
        response.text
    )

    print("=" * 80)
    print("GEMINI TEST")
    print("=" * 80)

    print()
    print("Intent:", result.intent)
    print("Confidence:", result.confidence)
    print("Reason:", result.reason)


if __name__ == "__main__":
    main()