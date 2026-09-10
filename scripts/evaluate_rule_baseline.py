import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


GOLDEN_PATH = "data/golden/golden_set.csv"


RULES = {
    "SELLER_FRAUD": [
        r"\bfraud\b",
        r"\bscam\b",
        r"\bscammer\b",
        r"\bfake seller\b",
        r"\bphishing\b",
        r"\bimpersonat",
    ],

    "REFUND": [
        r"\brefund\b",
        r"\brefunded\b",
        r"\bmoney back\b",
        r"\brefund.*return\b",
    ],

    "RETURN": [
        r"\breturn\b",
        r"\breturning\b",
        r"\bsend.*back\b",
    ],

    "PAYMENT": [
        r"\bpayment\b",
        r"\bpaid\b",
        r"\bpay\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bcard\b",
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bcheckout\b",
    ],

    "ORDER_CHANGE": [
        r"\bcancel.*order\b",
        r"\bcancel.*purchase\b",
        r"\bchange.*order\b",
        r"\bchange.*address\b",
        r"\bchange.*delivery\b",
        r"\bmodify.*order\b",
    ],

    "DELIVERY_TRACKING": [
        r"\btracking\b",
        r"\btrack.*order\b",
        r"\btrack.*package\b",
        r"\btracking number\b",
        r"\bwhere.*package\b",
        r"\bwhere.*order\b",
    ],

    "DELIVERY_FAILURE": [
        r"\bfailed delivery\b",
        r"\bdelivery failed\b",
        r"\bcouldn't deliver\b",
        r"\bcould not deliver\b",
        r"\bunable to deliver\b",
        r"\bwrong address\b",
        r"\bwrong location\b",
        r"\bmissed delivery\b",
    ],

    "DELIVERY_DELAY": [
        r"\bdelayed\b",
        r"\bdelay\b",
        r"\blate\b",
        r"\boverdue\b",
        r"\bhasn't arrived\b",
        r"\bhas not arrived\b",
        r"\bnot arrived\b",
        r"\bstill waiting\b",
        r"\bpast.*delivery\b",
    ],

    "PRIME": [
        r"\bprime\b",
    ],

    "TECHNICAL": [
        r"\berror\b",
        r"\bbug\b",
        r"\bcrash\b",
        r"\bnot working\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b",
        r"\bwon't work\b",
        r"\bcan't connect\b",
        r"\bcannot connect\b",
        r"\bcan't play\b",
        r"\bcannot play\b",
        r"\bplayback\b",
    ],

    "DIGITAL_CONTENT": [
        r"\bprime video\b",
        r"\bkindle\b",
        r"\bmovie\b",
        r"\bmovies\b",
        r"\bshow\b",
        r"\bseries\b",
        r"\bebook\b",
        r"\be-book\b",
        r"\bdigital content\b",
    ],

    "SUPPORT_COMPLAINT": [
        r"\bcustomer service\b",
        r"\bcustomer support\b",
        r"\bsupport team\b",
        r"\bsupport agent\b",
        r"\bterrible service\b",
        r"\bpoor service\b",
        r"\bno help\b",
        r"\bnot helping\b",
        r"\bno one.*help\b",
        r"\bprevious agent\b",
    ],

    "ORDER_PRODUCT": [
        r"\bproduct\b",
        r"\bitem\b",
        r"\bavailable\b",
        r"\bavailability\b",
        r"\bin stock\b",
        r"\bprice\b",
        r"\bpricing\b",
        r"\bhow much\b",
    ],
}


def classify(text):
    text = str(text).lower()

    matches = []

    for intent, patterns in RULES.items():

        count = 0

        for pattern in patterns:
            if re.search(pattern, text):
                count += 1

        if count > 0:
            matches.append((intent, count))

    if not matches:
        return "OTHER"

    # Most matching rules wins.
    # Explicit ordering resolves ties.
    priority = [
        "SELLER_FRAUD",
        "REFUND",
        "RETURN",
        "PAYMENT",
        "ORDER_CHANGE",
        "DELIVERY_FAILURE",
        "DELIVERY_TRACKING",
        "DELIVERY_DELAY",
        "PRIME",
        "TECHNICAL",
        "DIGITAL_CONTENT",
        "SUPPORT_COMPLAINT",
        "ORDER_PRODUCT",
    ]

    priority_rank = {
        intent: i
        for i, intent in enumerate(priority)
    }

    matches.sort(
        key=lambda x: (
            -x[1],
            priority_rank[x[0]],
        )
    )

    return matches[0][0]


def main():

    print("=" * 80)
    print("RULE-BASED INTENT BASELINE")
    print("=" * 80)

    df = pd.read_csv(GOLDEN_PATH)

    y_true = df["gold_intent"].astype(str).tolist()

    y_pred = [
        classify(text)
        for text in df["text"]
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
    print("HEADLINE METRICS")
    print("-" * 80)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    intents = list(RULES.keys()) + ["OTHER"]

    print()
    print("PER-INTENT PERFORMANCE")
    print("-" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=intents,
            zero_division=0,
        )
    )

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 80)

    print(
        pd.Series(y_pred)
        .value_counts()
        .to_string()
    )

    output = df.copy()

    output["predicted_intent"] = y_pred

    output.to_csv(
        "data/golden/rule_baseline_predictions.csv",
        index=False,
    )

    print()
    print(
        "Saved: "
        "data/golden/rule_baseline_predictions.csv"
    )


if __name__ == "__main__":
    main()