import argparse
import os

from pydantic import BaseModel
from google import genai
from google.genai import types


class EscalationDecision(BaseModel):
    decision: str
    reason: str
    confidence: float


VALID_DECISIONS = {
    "AUTO_HANDLE",
    "ESCALATE",
}


def decide_escalation(
    client,
    customer_message,
    intent,
    draft_response,
    retrieved_cases,
):
    evidence = []

    for case in retrieved_cases:
        evidence.append(
            f"""
Historical case {case["rank"]}:
Customer: {case["customer_text"]}
Historical response: {case["agent_response"]}
""".strip()
        )

    evidence_text = "\n\n".join(evidence)

    prompt = f"""
You are the escalation decision component of an Amazon customer-support
AI assistant.

CUSTOMER MESSAGE:
{customer_message}

INTENT:
{intent}

DRAFT RESPONSE:
{draft_response}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

Decide whether this customer message can safely receive an automated
response or should be escalated to a human support agent.

Choose exactly one:

AUTO_HANDLE
ESCALATE

AUTO_HANDLE means:
- The issue can be addressed safely with general information or
  non-consequential troubleshooting.
- No account-specific lookup is required.
- No financial investigation is required.
- No order-specific action is required.
- No sensitive information needs to be handled.
- The response does not need to claim an action was performed.

ESCALATE means:
- Account/order-specific investigation is required.
- A refund, payment, charge, or financial issue requires investigation.
- Fraud or suspected fraudulent activity requires investigation.
- A delivery problem needs shipment/account investigation.
- The customer is asking the company to perform a consequential action.
- Sensitive personal information is involved.
- The customer reports repeated failure after reasonable troubleshooting.
- The historical evidence is insufficient to safely answer the question.
- The situation is ambiguous enough that an automated answer could cause
  harm or mislead the customer.

IMPORTANT:

1. Do not escalate merely because the customer is angry.
2. Do not auto-handle merely because similar historical responses exist.
3. Historical responses are evidence, NOT authoritative current policy.
4. Never assume access to the customer's Amazon account.
5. When uncertain, prefer ESCALATE.
6. Give a short, concrete reason for the decision.
7. Confidence must be between 0 and 1.

Return JSON with:
- decision
- reason
- confidence
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=EscalationDecision,
        ),
    )

    result = EscalationDecision.model_validate_json(
        response.text
    )

    result.decision = result.decision.strip().upper()

    if result.decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision returned: {result.decision}"
        )

    result.confidence = max(
        0.0,
        min(1.0, result.confidence),
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message",
        required=True,
    )

    parser.add_argument(
        "--intent",
        required=True,
    )

    parser.add_argument(
        "--draft",
        required=True,
    )

    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    # For this standalone test, no retrieval evidence is required.
    # The full agent will pass the actual retrieved cases.
    result = decide_escalation(
        client=client,
        customer_message=args.message,
        intent=args.intent,
        draft_response=args.draft,
        retrieved_cases=[],
    )

    print("=" * 80)
    print("ESCALATION DECISION")
    print("=" * 80)

    print()
    print(f"Decision:   {result.decision}")
    print(f"Reason:     {result.reason}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    main()