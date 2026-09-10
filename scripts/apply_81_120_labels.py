import pandas as pd


INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set.csv"


labels = {
    81: ("OTHER", "AUTO_HANDLE",
         "Casual commentary with no support request."),

    82: ("OTHER", "ESCALATE",
         "Customer requests urgent action on specific orders, but the underlying issue is not stated."),

    83: ("DIGITAL_CONTENT", "AUTO_HANDLE",
         "Customer wants to watch a specific movie; content availability or access question."),

    84: ("DELIVERY_TRACKING", "ESCALATE",
         "Shipment location appears inconsistent with the expected route and delivery is already late."),

    85: ("DELIVERY_DELAY", "AUTO_HANDLE",
         "Delivery date changed to earlier than expected and the customer is pleased with the update."),

    86: ("OTHER", "ESCALATE",
         "Customer asks to expedite an ongoing process, but the underlying support issue is not stated."),

    87: ("DELIVERY_DELAY", "ESCALATE",
         "Customer paid for Prime and one-day shipping but wants the order delivered on time."),

    88: ("OTHER", "AUTO_HANDLE",
         "Positive acknowledgment of a fast support response."),

    89: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Existing investigation has a 3–5 day response period and customer requests expedited handling."),

    90: ("OTHER", "AUTO_HANDLE",
         "Positive commentary about fast delivery with no support request."),

    91: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer explicitly asks for a response, but the underlying issue is not stated."),

    92: ("DELIVERY_FAILURE", "ESCALATE",
         "Customer attributes the need to reorder and additional cost to a delivery driver's mistake."),

    93: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer has waited 22 days and is demanding a response or resolution."),

    94: ("DELIVERY_FAILURE", "AUTO_HANDLE",
         "Sarcastic comment about poor delivery with no explicit request for intervention."),

    95: ("DIGITAL_CONTENT", "AUTO_HANDLE",
         "Informational question about whether Amazon Video is included with Prime."),

    96: ("RETURN", "ESCALATE",
         "Customer complains about having to return orders and references a specific order requiring investigation."),

    97: ("OTHER", "ESCALATE",
         "Account/password error is reported, but the taxonomy does not include a dedicated account-access intent."),

    98: ("PAYMENT", "ESCALATE",
         "Cash-on-delivery payment method remains blocked and requires investigation."),

    99: ("TECHNICAL", "AUTO_HANDLE",
         "Customer reports that an issue persists while noting that it works on their phone; technical troubleshooting context."),

    100: ("DELIVERY_DELAY", "ESCALATE",
         "Delivery has taken six days and was not completed within the expected period."),

    101: ("PRIME", "ESCALATE",
         "Prime trial converted to paid membership and the customer was charged; billing/account-specific resolution required."),

    102: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer followed the instructed process but received no response."),

    103: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Promised callback never occurred and emails only generated automated responses."),

    104: ("DELIVERY_DELAY", "ESCALATE",
         "Order remains pending shipment and has repeatedly been delayed."),

    105: ("DELIVERY_DELAY", "AUTO_HANDLE",
         "Customer is hoping delivery occurs as promised without making a specific intervention request."),

    106: ("RETURN", "ESCALATE",
         "Return is in progress and customer expects a refund; return/order-specific processing is involved."),

    107: ("DELIVERY_FAILURE", "ESCALATE",
         "Customer says a Super Famicom Mini has not arrived; delivery problem requires investigation."),

    108: ("TECHNICAL", "AUTO_HANDLE",
         "A book's page or functionality has apparently been broken for a long time; technical/product-page issue."),

    109: ("PAYMENT", "ESCALATE",
         "Customer appears unable to complete an order because of a payment/security process issue."),

    110: ("ORDER_CHANGE", "ESCALATE",
         "Previous cancellation request apparently failed and the order was still delivered; order-state problem."),

    111: ("OTHER", "AUTO_HANDLE",
         "Customer is worried about possible package theft but reports no actual theft or delivery failure."),

    112: ("OTHER", "AUTO_HANDLE",
         "Casual commentary with no support request."),

    113: ("REFUND", "ESCALATE",
         "Customer explicitly requests a refund after service denial and threatens legal action."),

    114: ("ORDER_CHANGE", "ESCALATE",
         "Customer requests an address/postcode update relevant to delivery."),

    115: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Explicit complaint about customer service and insurance/service quality."),

    116: ("DELIVERY_DELAY", "ESCALATE",
         "Customer complains about delivery taking more than two days and questions the value of Prime."),

    117: ("DELIVERY_DELAY", "ESCALATE",
         "Pre-order is expected to ship on time but customer is concerned about delay."),

    118: ("OTHER", "AUTO_HANDLE",
         "Casual entertainment/content commentary with no support request."),

    119: ("REFUND", "AUTO_HANDLE",
         "Customer confirms receiving the items and asks Amazon to stop an already-initiated refund."),

    120: ("OTHER", "AUTO_HANDLE",
         "Positive commentary about fast Prime Now delivery."),
}


def main():
    df = pd.read_csv(INPUT)

    for example_num, (intent, escalation, reason) in labels.items():
        idx = example_num - 1

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_escalation"] = escalation
        df.loc[idx, "gold_reason"] = reason

    df.to_csv(OUTPUT, index=False)

    print("Applied reviewed labels to examples 81-120.")
    print()

    print(
        df.iloc[80:120][
            ["tweet_id", "gold_intent", "gold_escalation", "gold_reason"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()