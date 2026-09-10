import pandas as pd

INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set.csv"


labels = {
    1: ("OTHER", "AUTO_HANDLE",
        "Non-actionable digital-content comment; no support request.",
        ""),

    2: ("TECHNICAL", "AUTO_HANDLE",
        "Prime Video streaming appears not to work.",
        ""),

    3: ("OTHER", "ESCALATE",
        "Insufficient context to determine the underlying issue.",
        ""),

    4: ("TECHNICAL", "AUTO_HANDLE",
        "Video unavailable/location restriction is a digital-service access problem.",
        ""),

    5: ("DELIVERY_DELAY", "ESCALATE",
        "Urgent replacement/delivery issue requires order-specific action.",
        ""),

    6: ("DELIVERY_DELAY", "ESCALATE",
        "Paid expedited shipping but delivery is substantially delayed.",
        ""),

    7: ("DELIVERY_FAILURE", "ESCALATE",
        "Delivery provider is failing to properly deliver to the location.",
        ""),

    8: ("DELIVERY_FAILURE", "ESCALATE",
        "Orders are repeatedly being returned without explanation.",
        ""),

    9: ("TECHNICAL", "ESCALATE",
        "Geo-restriction issue remains unresolved after repeated support contacts.",
        ""),

    10: ("DELIVERY_FAILURE", "ESCALATE",
         "Package arrived damaged during delivery.",
         ""),

    11: ("DIGITAL_CONTENT", "AUTO_HANDLE",
         "Informational question about the viewing period for rented content.",
         ""),

    12: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Repeated unsuccessful support interaction; customer is exhausted and wants resolution.",
         ""),

    13: ("OTHER", "AUTO_HANDLE",
         "Issue has already been resolved and no further action is requested.",
         ""),

    14: ("OTHER", "AUTO_HANDLE",
         "Message indicates partial resolution but gives insufficient information about the underlying issue.",
         ""),

    15: ("ORDER_PRODUCT", "ESCALATE",
         "Pre-order/product availability problem requires customer/order-specific resolution.",
         ""),

    16: ("ORDER_CHANGE", "ESCALATE",
         "Customer wants to change the address/details of an existing order.",
         ""),

    17: ("ORDER_CHANGE", "AUTO_HANDLE",
         "Customer is asking about available delivery options rather than a specific shipment failure.",
         ""),

    18: ("DELIVERY_FAILURE", "ESCALATE",
         "Improper delivery-driver behavior requires investigation.",
         ""),

    19: ("DELIVERY_FAILURE", "ESCALATE",
         "Delivery is unavailable to the customer's location.",
         ""),

    20: ("OTHER", "AUTO_HANDLE",
         "Message is a brief resolved/commentary statement with no actionable support request.",
         ""),
}

def main():
    df = pd.read_csv(INPUT)

    # These columns contain text labels/reasons/notes.
    # Force them to object dtype so strings can be assigned safely.
    text_columns = [
        "gold_intent",
        "gold_escalation",
        "gold_reason",
        "annotator_notes",
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = None
        else:
            df[column] = df[column].astype("object")

    if len(df) < 20:
        raise ValueError("Golden set contains fewer than 20 rows.")

    for number, (intent, escalation, reason, notes) in labels.items():
        idx = number - 1

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_escalation"] = escalation
        df.loc[idx, "gold_reason"] = reason
        df.loc[idx, "annotator_notes"] = notes

    df.to_csv(OUTPUT, index=False)

    print("Applied reviewed labels to examples 1-20.")
    print()
    print(df.iloc[:20][
        [
            "tweet_id",
            "gold_intent",
            "gold_escalation",
            "gold_reason"
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()