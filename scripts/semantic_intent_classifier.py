import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


INTENT_DESCRIPTIONS = {
    "DELIVERY_DELAY":
        "The customer says an order or package is late, delayed, overdue, "
        "or has not arrived by the promised delivery date.",

    "DELIVERY_TRACKING":
        "The customer wants to track an order, asks where the package is, "
        "asks about tracking information, carrier status, or delivery status.",

    "DELIVERY_FAILURE":
        "The customer reports that delivery failed, was missed, was attempted "
        "unsuccessfully, was delivered to the wrong place, or the carrier could "
        "not complete delivery.",

    "ORDER_CHANGE":
        "The customer wants to change, cancel, modify, or update an order, "
        "delivery address, shipping option, or other order detail.",

    "RETURN":
        "The customer wants to return an item, asks about return eligibility, "
        "return procedure, return shipping, or returning a product.",

    "REFUND":
        "The customer asks about a refund, missing refund, refund timing, "
        "refund status, or money being returned after a purchase or return.",

    "PAYMENT":
        "The customer has a payment, billing, charge, card, checkout payment, "
        "or payment authorization problem.",

    "PRIME":
        "The customer asks about Amazon Prime membership, Prime benefits, "
        "Prime eligibility, Prime shipping benefits, or Prime subscription.",

    "TECHNICAL":
        "The customer reports a technical malfunction, error, broken feature, "
        "website problem, application problem, device problem, or playback issue.",

    "DIGITAL_CONTENT":
        "The customer asks an informational question about digital content, "
        "Prime Video, movies, shows, Kindle, digital purchases, availability, "
        "language, access, or content-related information without reporting "
        "a technical malfunction.",

    "SELLER_FRAUD":
        "The customer reports fraud, a scam, suspicious seller behavior, "
        "a fraudulent transaction, impersonation, or a potentially fraudulent "
        "Amazon seller or contact.",

    "ORDER_PRODUCT":
        "The customer asks about a specific product or order item, including "
        "product availability, product information, order item details, "
        "pricing, specifications, or product-related questions.",

    "SUPPORT_COMPLAINT":
        "The main problem is dissatisfaction with Amazon customer support, "
        "a previous agent, lack of help, poor support experience, or repeated "
        "failure to resolve the customer's issue.",

    "OTHER":
        "The message does not contain a clear actionable Amazon support intent, "
        "is casual conversation, praise, thanks, irrelevant content, or has "
        "insufficient information to determine a support issue.",
}


class SemanticIntentClassifier:

    def __init__(self):
        print(f"Loading model: {MODEL_NAME}")

        self.model = SentenceTransformer(MODEL_NAME)

        self.intents = list(INTENT_DESCRIPTIONS.keys())

        descriptions = [
            INTENT_DESCRIPTIONS[intent]
            for intent in self.intents
        ]

        self.intent_embeddings = self.model.encode(
            descriptions,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def predict(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores = embeddings @ self.intent_embeddings.T

        predictions = []

        for row in scores:
            best_idx = int(np.argmax(row))

            predictions.append(
                {
                    "intent": self.intents[best_idx],
                    "score": float(row[best_idx]),
                }
            )

        return predictions


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    classifier = SemanticIntentClassifier()

    result = classifier.predict([args.query])[0]

    print()
    print("=" * 80)
    print("SEMANTIC INTENT CLASSIFICATION")
    print("=" * 80)

    print()
    print("MESSAGE:")
    print(args.query)

    print()
    print(f"PREDICTED INTENT: {result['intent']}")
    print(f"SIMILARITY SCORE: {result['score']:.4f}")


if __name__ == "__main__":
    main()