import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sentence_transformers import SentenceTransformer
import numpy as np


GOLDEN_PATH = "data/golden/golden_set.csv"

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


def main():

    print("=" * 80)
    print("SEMANTIC INTENT CLASSIFIER EVALUATION")
    print("=" * 80)

    print("\nLoading golden set...")

    df = pd.read_csv(GOLDEN_PATH)

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y_true = (
        df["gold_intent"]
        .astype(str)
        .tolist()
    )

    print(f"Golden examples: {len(df)}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    intents = list(INTENT_DESCRIPTIONS.keys())

    descriptions = [
        INTENT_DESCRIPTIONS[intent]
        for intent in intents
    ]

    print("Encoding intent descriptions...")

    intent_embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("Encoding golden examples...")

    text_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ---------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------

    similarities = text_embeddings @ intent_embeddings.T

    predicted_indices = np.argmax(
        similarities,
        axis=1,
    )

    y_pred = [
        intents[i]
        for i in predicted_indices
    ]

    confidence = similarities[
        np.arange(len(similarities)),
        predicted_indices,
    ]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("PER-INTENT PERFORMANCE")
    print("=" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=intents,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=intents,
    )

    cm_df = pd.DataFrame(
        cm,
        index=intents,
        columns=intents,
    )

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(cm_df.to_string())

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output = df.copy()

    output["predicted_intent"] = y_pred
    output["semantic_score"] = confidence

    output.to_csv(
        "data/golden/semantic_classifier_predictions.csv",
        index=False,
    )

    print()
    print(
        "Saved: "
        "data/golden/semantic_classifier_predictions.csv"
    )


if __name__ == "__main__":
    main()