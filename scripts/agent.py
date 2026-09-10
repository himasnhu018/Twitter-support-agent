import argparse
import os
from pathlib import Path

import faiss
import pandas as pd
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


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


INTENT_DEFINITIONS = {
    "DELIVERY_DELAY":
        "Package/order is late or has missed an expected delivery time.",

    "DELIVERY_TRACKING":
        "Customer wants tracking or shipment status information.",

    "DELIVERY_FAILURE":
        "Package was not delivered, was lost, or was marked delivered incorrectly.",

    "ORDER_CHANGE":
        "Customer wants to cancel, modify, edit, or change an order/address.",

    "RETURN":
        "Customer wants to return or exchange a product.",

    "REFUND":
        "Customer is asking about a refund or missing refund.",

    "PAYMENT":
        "Payment, charge, billing, card, or payment authorization issue.",

    "PRIME":
        "Amazon Prime membership or subscription issue.",

    "TECHNICAL":
        "Amazon app, website, device, or service malfunction.",

    "DIGITAL_CONTENT":
        "Informational question about Prime Video, Kindle, ebooks, movies, or digital content.",

    "SELLER_FRAUD":
        "Suspected seller fraud, scam, counterfeit, or fraudulent seller behavior.",

    "ORDER_PRODUCT":
        "Question about a product, availability, selection, or ordering a product.",

    "SUPPORT_COMPLAINT":
        "Complaint primarily about Amazon customer support/service.",

    "OTHER":
        "Irrelevant, social, already-resolved, or insufficient-context message.",
}


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reason: str


class SupportResponse(BaseModel):
    reply: str
    grounding: str
    confidence: float


class EscalationDecision(BaseModel):
    decision: str
    reason: str
    confidence: float


class AmazonSupportAgent:

    def __init__(self, k=5):

        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.k = k

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            "data/processed/amazon_support.faiss"
        )

        self.metadata = pd.read_parquet(
            "data/processed/amazon_support_metadata.parquet"
        )

        print(
            f"FAISS vectors: {self.index.ntotal:,}"
        )

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

    # --------------------------------------------------------------
    # INTENT CLASSIFICATION
    # --------------------------------------------------------------

    def classify_intent(self, message):

        definitions = "\n".join(
            f"- {intent}: {description}"
            for intent, description
            in INTENT_DEFINITIONS.items()
        )

        prompt = f"""
Classify this Amazon customer-support message into exactly one intent.

CUSTOMER MESSAGE:
{message}

INTENTS:
{definitions}

Rules:

- Choose the customer's primary actionable intent.
- Emotion does not determine intent.
- Account-specific lookup is not itself an intent.
- If the underlying problem is clear, choose that problem instead of
  SUPPORT_COMPLAINT.
- Use DELIVERY_DELAY for late packages.
- Use DELIVERY_TRACKING for tracking/status requests.
- Use DELIVERY_FAILURE for lost, missing, or incorrectly delivered packages.
- Use TECHNICAL for actual technical malfunction.
- Use DIGITAL_CONTENT for informational digital-content questions.
- Use OTHER when there is insufficient context or no meaningful support intent.

Return:
intent
confidence from 0 to 1
short reason
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=IntentResult,
            ),
        )

        result = IntentResult.model_validate_json(
            response.text
        )

        result.intent = result.intent.strip().upper()

        if result.intent not in INTENTS:
            result.intent = "OTHER"

        result.confidence = max(
            0,
            min(1, result.confidence),
        )

        return result

    # --------------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------------

    def retrieve(self, message):

        embedding = self.embedding_model.encode(
            [message],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores, indices = self.index.search(
            embedding.astype("float32"),
            self.k,
        )

        cases = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if idx < 0:
                continue

            row = self.metadata.iloc[int(idx)]

            cases.append(
                {
                    "rank": rank,
                    "similarity": float(score),
                    "customer_text": str(
                        row["customer_text"]
                    ),
                    "agent_response": str(
                        row["agent_response"]
                    ),
                }
            )

        return cases

    # --------------------------------------------------------------
    # RESPONSE GENERATION
    # --------------------------------------------------------------

    def generate_response(
        self,
        message,
        intent,
        cases,
    ):

        evidence = []

        for case in cases:

            evidence.append(
                f"""
Historical case {case["rank"]}
Similarity: {case["similarity"]:.4f}

Customer:
{case["customer_text"]}

AmazonHelp response:
{case["agent_response"]}
""".strip()
            )

        evidence_text = "\n\n".join(evidence)

        prompt = f"""
You are an AI customer-support drafting assistant for AmazonHelp.

CUSTOMER MESSAGE:
{message}

INTENT:
{intent}

INTENT DESCRIPTION:
{INTENT_DEFINITIONS[intent]}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

Generate a concise customer-facing response.

Rules:

1. Historical cases are evidence for response patterns, NOT authoritative
   current policy.

2. Use only historical evidence that is clearly relevant.

3. Never blindly copy a historical response.

4. Never invent order information.

5. Never invent tracking numbers, delivery dates, refund amounts,
   account details, or policy guarantees.

6. Never claim that you checked the customer's account.

7. Never claim that you performed an action.

8. Never copy URLs, ticket IDs, order IDs, tracking IDs, usernames,
   email addresses, or other identifiers from historical examples.

9. If a historical response contains a URL, describe the underlying
   action generically instead.

10. If account-specific investigation is required, do not pretend
    to resolve it.

11. Keep the answer concise and natural.

12. Do not mention AI, FAISS, embeddings, retrieval, or historical
    examples.

Return:
reply
grounding
confidence from 0 to 1
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=SupportResponse,
            ),
        )

        result = SupportResponse.model_validate_json(
            response.text
        )

        result.confidence = max(
            0,
            min(1, result.confidence),
        )

        return result

    # --------------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------------

    def decide_escalation(
        self,
        message,
        intent,
        draft,
        cases,
    ):

        evidence = "\n\n".join(
            f"""
Historical case {case["rank"]}:
Customer: {case["customer_text"]}
Response: {case["agent_response"]}
"""
            for case in cases
        )

        prompt = f"""
You are the escalation decision component of an Amazon support AI agent.

CUSTOMER:
{message}

INTENT:
{intent}

DRAFT RESPONSE:
{draft}

HISTORICAL EVIDENCE:
{evidence}

Choose exactly one:

AUTO_HANDLE
ESCALATE

AUTO_HANDLE when:
- The response can fully address the customer's immediate need using
  general information or safe troubleshooting.
- No account-specific lookup is required.
- No financial investigation is required.
- No order-specific investigation is required.
- No consequential action is required.
- No sensitive information is involved.

ESCALATE when:
- The customer needs Amazon to inspect their specific order, shipment,
  account, payment, refund, or transaction.
- The customer is asking where their specific package/refund/payment is.
- A delivery problem requires shipment investigation.
- A refund or payment issue requires transaction investigation.
- The customer requests cancellation, modification, or another
  consequential action.
- Fraud, counterfeit goods, or suspected scams are reported.
- The customer has already tried reasonable troubleshooting without
  resolving the issue.
- Sensitive information is involved.
- The available evidence is insufficient to safely resolve the issue.

IMPORTANT:
A generic instruction such as "check your tracking" does NOT by itself
make a customer-specific delivery problem AUTO_HANDLE. If the customer's
actual problem requires Amazon to investigate their specific shipment,
choose ESCALATE.

Return:
decision
reason
confidence from 0 to 1
"""

        response = self.client.models.generate_content(
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

        if result.decision not in {
            "AUTO_HANDLE",
            "ESCALATE",
        }:
            result.decision = "ESCALATE"

        result.confidence = max(
            0,
            min(1, result.confidence),
        )

        return result

    # --------------------------------------------------------------
    # END-TO-END PIPELINE
    # --------------------------------------------------------------

    def run(self, message):

        print()
        print("=" * 80)
        print("AMAZONHELP AI SUPPORT AGENT")
        print("=" * 80)

        print()
        print("Customer:")
        print(message)

        # 1. Intent
        print()
        print("[1/4] Classifying intent...")

        intent_result = self.classify_intent(
            message
        )

        print(
            f"Intent: {intent_result.intent}"
        )

        print(
            f"Confidence: {intent_result.confidence:.2f}"
        )

        print(
            f"Reason: {intent_result.reason}"
        )

        # 2. Retrieval
        print()
        print("[2/4] Retrieving historical evidence...")

        cases = self.retrieve(message)

        for case in cases:
            print(
                f"Rank {case['rank']} | "
                f"similarity={case['similarity']:.4f}"
            )

        # 3. Response
        print()
        print("[3/4] Generating grounded response...")

        response = self.generate_response(
            message,
            intent_result.intent,
            cases,
        )

        print()
        print("Draft response:")
        print(response.reply)

        print()
        print(
            f"Response confidence: "
            f"{response.confidence:.2f}"
        )

        # 4. Escalation
        print()
        print("[4/4] Deciding escalation...")

        escalation = self.decide_escalation(
            message,
            intent_result.intent,
            response.reply,
            cases,
        )

        print()
        print(
            f"Decision: {escalation.decision}"
        )

        print(
            f"Reason: {escalation.reason}"
        )

        print(
            f"Confidence: {escalation.confidence:.2f}"
        )

        print()
        print("=" * 80)

        return {
            "message": message,
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "intent_reason": intent_result.reason,
            "reply": response.reply,
            "grounding": response.grounding,
            "response_confidence": response.confidence,
            "decision": escalation.decision,
            "escalation_reason": escalation.reason,
            "escalation_confidence": escalation.confidence,
            "retrieved_cases": cases,
        }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message",
        required=True,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    agent = AmazonSupportAgent(
        k=args.k
    )

    agent.run(args.message)


if __name__ == "__main__":
    main()