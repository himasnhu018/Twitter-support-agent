import pandas as pd


INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set.csv"


labels = {
    21: ("DELIVERY_FAILURE", "ESCALATE",
         "Order was not received and courier responsibility is disputed; shipment investigation is required."),

    22: ("OTHER", "AUTO_HANDLE",
         "Pure reaction to a previous response; no actionable issue is stated."),

    23: ("PRIME", "AUTO_HANDLE",
         "Positive/commentary message about the value of the Prime service."),

    24: ("TECHNICAL", "AUTO_HANDLE",
         "Customer cannot stream in HD despite sufficient bandwidth and quality settings."),

    25: ("DELIVERY_DELAY", "AUTO_HANDLE",
         "Complaint about shipping-time expectations, particularly for pre-orders."),

    26: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Explicit legal threat concerning a specific order; consequential and high-risk escalation."),

    27: ("ORDER_PRODUCT", "AUTO_HANDLE",
         "Customer requests a change to how pantry products can be purchased."),

    28: ("DIGITAL_CONTENT", "AUTO_HANDLE",
         "Commentary about watching content in different language versions; no support action requested."),

    29: ("REFUND", "ESCALATE",
         "Only part of the order was refunded; financial and order-specific investigation is required."),

    30: ("DELIVERY_DELAY", "ESCALATE",
         "Customer paid for faster shipping but the order still shows a later delivery date."),

    31: ("ORDER_PRODUCT", "AUTO_HANDLE",
         "Provides seller and fulfillment information in response to a previous support interaction; no standalone issue."),

    32: ("OTHER", "AUTO_HANDLE",
         "Non-support social/commentary message."),

    33: ("OTHER", "AUTO_HANDLE",
         "Promotional/commentary message with no support request."),

    34: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer reports repeated support failure and asks for query logs."),

    35: ("OTHER", "AUTO_HANDLE",
         "Customer confirms receiving the requested information and expresses appreciation; issue is effectively resolved."),

    36: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Complaint about seller and delivery-service quality requiring support attention."),

    37: ("DELIVERY_DELAY", "ESCALATE",
         "Account is blocked while an order is pending, creating uncertainty about expected delivery."),

    38: ("ORDER_PRODUCT", "ESCALATE",
         "Prime delivery status for a specific product appears to have changed and requires product/order-specific investigation."),

    39: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Previous steps were followed but there was no response or acknowledgment; customer asks about the support SLA."),

    40: ("OTHER", "ESCALATE",
         "A lock is mentioned but the underlying issue cannot be confidently determined from this message alone."),
}


def main():
    df = pd.read_csv(INPUT)

    for example_num, (intent, escalation, reason) in labels.items():
        idx = example_num - 1

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_escalation"] = escalation
        df.loc[idx, "gold_reason"] = reason

    df.to_csv(OUTPUT, index=False)

    print("Applied reviewed labels to examples 21-40.")
    print()

    print(
        df.iloc[20:40][
            ["tweet_id", "gold_intent", "gold_escalation", "gold_reason"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()