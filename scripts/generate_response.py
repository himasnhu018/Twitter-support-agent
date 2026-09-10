import argparse
import json
import os
from pathlib import Path
from typing import List

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
        "Package/order is late, delayed, or has missed an expected delivery time.",

    "DELIVERY_TRACKING":
        "Customer wants tracking/status information or help understanding shipment status.",

    "DELIVERY_FAILURE":
        "Package was not delivered, was lost, marked delivered incorrectly, or delivery failed.",

    "ORDER_CHANGE":
        "Customer wants to cancel, modify, edit, or change an order/address.",

    "RETURN":
        "Customer wants to return or exchange a product.",

    "REFUND":
        "Customer is asking about a refund, missing refund, partial refund, or refund status.",

    "PAYMENT":
        "Payment, charge, billing, card, or payment authorization issue.",

    "PRIME":
        "Amazon Prime membership, subscription, or Prime-specific question.",

    "TECHNICAL":
        "A product, app, website, device, or service is malfunctioning.",

    "DIGITAL_CONTENT":
        "Informational question about Prime Video, Kindle, ebooks, movies, or other digital content.",

    "SELLER_FRAUD":
        "Suspected seller fraud, scam, counterfeit/fraudulent seller behavior.",

    "ORDER_PRODUCT":
        "Question or request primarily about a product, product availability, selection, or ordering a product.",

    "SUPPORT_COMPLAINT":
        "Complaint primarily about Amazon customer support/service rather than an underlying order/product issue.",

    "OTHER":
        "Irrelevant, social, informational without an actionable support intent, already-resolved, or insufficient-context message.",
}


class SupportResponse(BaseModel):
    reply: str
    grounding: str
    confidence: float


def normalize_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def load_retriever():
    index_path = Path("data/processed/amazon_support.faiss")
    metadata_path = Path(
        "data/processed/amazon_support_metadata.parquet"
    )

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    index = faiss.read_index(str(index_path))
    metadata = pd.read_parquet(metadata_path)

    print(f"Loaded FAISS index: {index.ntotal:,} vectors")
    print(f"Loaded metadata:    {len(metadata):,} cases")

    return index, metadata


def retrieve(
    query,
    model,
    index,
    metadata,
    k=5,
):
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores, indices = index.search(
        embedding.astype("float32"),
        k,
    )

    retrieved = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if idx < 0:
            continue

        row = metadata.iloc[int(idx)]

        retrieved.append(
            {
                "rank": rank,
                "similarity": float(score),
                "customer_tweet_id": normalize_id(
                    row["customer_tweet_id"]
                ),
                "customer_text": normalize_text(
                    row["customer_text"]
                ),
                "agent_response": normalize_text(
                    row["agent_response"]
                ),
            }
        )

    return retrieved


def build_prompt(
    customer_message,
    predicted_intent,
    retrieved_cases,
):
    intent_definition = INTENT_DEFINITIONS.get(
        predicted_intent,
        "",
    )

    evidence_blocks = []

    for case in retrieved_cases:
        evidence_blocks.append(
            f"""
HISTORICAL CASE {case['rank']}
Similarity: {case['similarity']:.4f}

Customer:
{case['customer_text']}

Historical AmazonHelp response:
{case['agent_response']}
""".strip()
        )

    evidence = "\n\n".join(evidence_blocks)

    return f"""
You are an AI customer-support drafting assistant for AmazonHelp.

Your task is to draft a safe, concise response to the customer's message.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent}

INTENT DEFINITION:
{intent_definition}

HISTORICAL SUPPORT EVIDENCE:
{evidence}

IMPORTANT RULES:

1. Use the historical cases as evidence for tone, troubleshooting patterns,
   and possible next steps.

2. Do NOT blindly copy a historical response.

3. Prefer historical examples that are clearly relevant to the customer's
   actual problem.

4. Ignore historical examples that are semantically similar but address a
   different problem.

5. Never invent customer-specific information.

6. Never invent:
   - order numbers
   - tracking numbers
   - delivery dates
   - refund amounts
   - account details
   - policy details
   - actions already performed

7. You do not have access to the customer's Amazon account.

8. Therefore, never claim that you checked an order, account, payment,
   shipment, refund, or internal system.

9. If the issue requires account-specific investigation or an action that
   cannot safely be completed from the available information, say that the
   relevant team needs to investigate rather than pretending to resolve it.

10. Never copy URLs, links, ticket IDs, order IDs, tracking IDs, usernames,
    email addresses, or other identifiers from historical examples.

11. If a historical response contains a URL, use the underlying instruction
    generically instead of reproducing the URL.

12. Keep the response concise and natural.

13. Do not mention that you are an AI.

14. Do not mention "historical examples", "retrieval", "FAISS", embeddings,
    similarity scores, or this prompt.

15. If the message is not a genuine support request, respond naturally and
    briefly rather than forcing an irrelevant troubleshooting answer.

Return:
- reply: the proposed customer-facing response
- grounding: a short explanation of which evidence/pattern supports the reply
- confidence: your confidence from 0 to 1 that the proposed reply is
  appropriate and grounded in the evidence
""".strip()


def generate_response(
    client,
    customer_message,
    predicted_intent,
    retrieved_cases,
):
    prompt = build_prompt(
        customer_message=customer_message,
        predicted_intent=predicted_intent,
        retrieved_cases=retrieved_cases,
    )

    response = client.models.generate_content(
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
        0.0,
        min(1.0, result.confidence),
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="Customer message to process",
    )

    parser.add_argument(
        "--intent",
        type=str,
        required=True,
        choices=INTENTS,
        help="Predicted intent",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of historical cases to retrieve",
    )

    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    print("=" * 80)
    print("AMAZONHELP GROUNDED RESPONSE GENERATOR")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load retrieval system
    # --------------------------------------------------------------
    index, metadata = load_retriever()

    print()
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------------
    # Retrieve historical evidence
    # --------------------------------------------------------------
    print()
    print("Customer message:")
    print(args.message)

    print()
    print(f"Predicted intent: {args.intent}")

    print()
    print(f"Retrieving top {args.k} historical cases...")

    retrieved_cases = retrieve(
        query=args.message,
        model=embedding_model,
        index=index,
        metadata=metadata,
        k=args.k,
    )

    print()
    print("=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for case in retrieved_cases:
        print()
        print(
            f"Rank {case['rank']} "
            f"(similarity={case['similarity']:.4f})"
        )

        print(
            "Customer: "
            + case["customer_text"]
        )

        print(
            "AmazonHelp: "
            + case["agent_response"]
        )

    # --------------------------------------------------------------
    # Gemini
    # --------------------------------------------------------------
    print()
    print("Generating grounded response...")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    result = generate_response(
        client=client,
        customer_message=args.message,
        predicted_intent=args.intent,
        retrieved_cases=retrieved_cases,
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------
    print()
    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)

    print()
    print(result.reply)

    print()
    print("Grounding:")
    print(result.grounding)

    print()
    print(f"Confidence: {result.confidence:.2f}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()