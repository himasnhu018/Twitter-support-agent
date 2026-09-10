import pandas as pd


INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set.csv"


labels = {
    41: ("DELIVERY_FAILURE", "ESCALATE",
         "Unusual package handoff arrangement indicates a delivery problem requiring investigation."),

    42: ("DELIVERY_FAILURE", "ESCALATE",
         "Product arrived damaged and unsealed, with uncertainty about whether it is new or refurbished; replacement investigation required."),

    43: ("ORDER_CHANGE", "ESCALATE",
         "Customer accidentally cancelled an order and wants it recovered or reinstated."),

    44: ("DELIVERY_DELAY", "ESCALATE",
         "Customer says the package has not arrived and asks how long they must wait."),

    45: ("PAYMENT", "ESCALATE",
         "Gift-card loading and order payment are being rejected; account/payment-specific investigation is required."),

    46: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer reports an extremely long and frustrating support process for a replacement."),

    47: ("OTHER", "AUTO_HANDLE",
         "Customer says the previous help worked; issue is resolved."),

    48: ("DELIVERY_TRACKING", "AUTO_HANDLE",
         "Customer provides the carrier name in response to support; no new actionable problem is stated."),

    49: ("ORDER_PRODUCT", "ESCALATE",
         "Customer feels cheated because product pricing and promotions changed immediately after purchase."),

    50: ("DELIVERY_DELAY", "ESCALATE",
         "Prime customer complains that deliveries are repeatedly late."),

    51: ("DELIVERY_TRACKING", "ESCALATE",
         "Customer is waiting for a parcel and requests an immediate delivery update."),

    52: ("ORDER_CHANGE", "ESCALATE",
         "Order was cancelled and customer asks what to do to retain the advantageous price; order-specific resolution is required."),

    53: ("TECHNICAL", "ESCALATE",
         "Amazon's website appears unusually slow and the customer raises a possible security concern."),

    54: ("DELIVERY_DELAY", "ESCALATE",
         "Delivery is still marked for today near the end of the day and the customer expects a no-show."),

    55: ("PRIME", "AUTO_HANDLE",
         "Informational question about whether Amazon Video is included with Prime or requires another subscription."),

    56: ("ORDER_PRODUCT", "AUTO_HANDLE",
         "Customer cannot check out items in the basket; product purchase issue without an explicit payment failure."),

    57: ("DELIVERY_TRACKING", "ESCALATE",
         "Package is marked delivered to a locker but the customer has not received the collection code."),

    58: ("TECHNICAL", "AUTO_HANDLE",
         "Customer reports difficulty rewinding and fast-forwarding in the PS4 app."),

    59: ("DELIVERY_DELAY", "AUTO_HANDLE",
         "Customer acknowledges general delivery difficulties and chooses to wait until Friday."),

    60: ("PAYMENT", "ESCALATE",
         "Customer is concerned the unresolved issue could affect their credit rating; financial consequences require escalation."),

    61: ("DELIVERY_FAILURE", "ESCALATE",
         "Customer questions unexpected logistics surrounding an order; order-specific investigation is required."),

    62: ("DELIVERY_FAILURE", "ESCALATE",
         "Customer complains about difficulty recovering a package."),

    63: ("DELIVERY_FAILURE", "ESCALATE",
         "Delivery was marked unsuccessful despite the customer being home; package location is unclear."),

    64: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer is still waiting for a response from support."),

    65: ("OTHER", "AUTO_HANDLE",
         "Short contextual statement with no actionable support request."),

    66: ("DELIVERY_FAILURE", "ESCALATE",
         "Second delivery attempt has failed; customer wants to ensure the driver can reach them."),

    67: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Customer has waited a week for promised phone/email contact and asks what to do."),

    68: ("ORDER_PRODUCT", "AUTO_HANDLE",
         "Customer comments on a product's high price and shipping cost without requesting a specific support action."),

    69: ("ORDER_PRODUCT", "ESCALATE",
         "Customer disputes marketplace/seller pricing responsibility and requests accountability."),

    70: ("OTHER", "ESCALATE",
         "Customer reports receiving a selling-related message despite saying they do not have a seller account; underlying issue is unclear and requires investigation."),

    71: ("OTHER", "AUTO_HANDLE",
         "Informational question about when results of a game or contest will be announced."),

    72: ("OTHER", "AUTO_HANDLE",
         "Feature/language request for the Amazon site and app; no defined taxonomy category applies."),

    73: ("SUPPORT_COMPLAINT", "ESCALATE",
         "Strong complaint about support and failure to replace a defective phone."),

    74: ("OTHER", "AUTO_HANDLE",
         "Simple conversational response with no actionable issue."),

    75: ("DELIVERY_DELAY", "ESCALATE",
         "Delivery is expected today but there is a concern it cannot physically fit in the mailbox."),

    76: ("ORDER_PRODUCT", "ESCALATE",
         "Customer alleges a seller is charging above MRP and requests investigation."),

    77: ("OTHER", "AUTO_HANDLE",
         "Customer merely provides contact information; no explicit issue is stated."),

    78: ("DELIVERY_DELAY", "ESCALATE",
         "Customer paid for next-day delivery but received it late and complains about lack of follow-up."),

    79: ("OTHER", "AUTO_HANDLE",
         "Positive customer-service compliment with no support request."),

    80: ("DELIVERY_FAILURE", "ESCALATE",
         "Birthday package arrived empty; missing contents require order/delivery investigation."),
}


def main():
    df = pd.read_csv(INPUT)

    for example_num, (intent, escalation, reason) in labels.items():
        idx = example_num - 1

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_escalation"] = escalation
        df.loc[idx, "gold_reason"] = reason

    df.to_csv(OUTPUT, index=False)

    print("Applied reviewed labels to examples 41-80.")
    print()

    print(
        df.iloc[40:80][
            ["tweet_id", "gold_intent", "gold_escalation", "gold_reason"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()