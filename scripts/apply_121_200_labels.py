import pandas as pd


INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set.csv"


labels = {
    # =========================
    # 121-160
    # =========================

    121: ("OTHER", "AUTO_HANDLE",
          "Reaction to a previous response with no actionable issue stated."),

    122: ("DELIVERY_DELAY", "ESCALATE",
          "Customer paid for next-day delivery but says the order is not arriving next day."),

    123: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer cannot access the provided support link and requests direct contact."),

    124: ("DELIVERY_FAILURE", "AUTO_HANDLE",
          "Customer is waiting to see whether the parcel arrives and plans to visit the post office if it does not."),

    125: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer says Amazon has not provided a solution."),

    126: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Promised callback has not arrived and the customer requests an urgent response."),

    127: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer complains about seller review solicitation and Amazon's handling of the review."),

    128: ("DELIVERY_FAILURE", "AUTO_HANDLE",
          "Commentary about differences in delivery behavior between Amazon and FedEx; no explicit support request."),

    129: ("DELIVERY_DELAY", "ESCALATE",
          "Order has not arrived within the expected timeframe."),

    130: ("PAYMENT", "AUTO_HANDLE",
          "Customer confirms that a monetary amount has been activated or credited to the account."),

    131: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Promised agent contact did not happen and the customer complains about lack of communication."),

    132: ("ORDER_PRODUCT", "AUTO_HANDLE",
          "Simple statement about shopping through Prime Now with no support issue."),

    133: ("DELIVERY_DELAY", "ESCALATE",
          "Customer has paid for a package and is waiting for it to arrive."),

    134: ("OTHER", "ESCALATE",
          "Customer suspects an email may be a scam because it is not associated with their Amazon account; underlying issue is unclear."),

    135: ("ORDER_PRODUCT", "AUTO_HANDLE",
          "Feature suggestion for packaging with purchases; no existing order problem."),

    136: ("ORDER_PRODUCT", "ESCALATE",
          "Customer has repeatedly received the wrong product color and requires product/order-specific resolution."),

    137: ("OTHER", "AUTO_HANDLE",
          "Customer confirms completion and thanks support."),

    138: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer says two orders had their deliveries seriously mishandled."),

    139: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Direct complaint about poor customer-service competence."),

    140: ("REFUND", "ESCALATE",
          "Returned package was picked up 15 days ago but the refund has not arrived and support is not helping."),

    141: ("DELIVERY_TRACKING", "ESCALATE",
          "Customer cannot identify the carrier and has paid for furniture that has not arrived."),

    142: ("ORDER_PRODUCT", "ESCALATE",
          "Pre-ordered signed edition became unavailable due to stock shortage and requires resolution."),

    143: ("ORDER_PRODUCT", "AUTO_HANDLE",
          "Commentary listing books or content purchased with no support request."),

    144: ("DELIVERY_TRACKING", "AUTO_HANDLE",
          "Automated delivery-status notification with no customer support request."),

    145: ("DELIVERY_TRACKING", "ESCALATE",
          "Tracking link does not work and customer asks where the parcel is and when it will arrive."),

    146: ("DELIVERY_DELAY", "AUTO_HANDLE",
          "Customer merely states the original delivery date; insufficient context for escalation."),

    147: ("DELIVERY_DELAY", "ESCALATE",
          "Customer urgently asks for a book that has not arrived."),

    148: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer asks Amazon to use the registered account to resolve an ongoing delivery problem."),

    149: ("DELIVERY_DELAY", "ESCALATE",
          "Prime member complains that products are not arriving quickly enough."),

    150: ("PAYMENT", "ESCALATE",
          "Bank-account confirmation is involved while the customer's Amazon account is closed; account and financial handling is required."),

    151: ("DELIVERY_TRACKING", "ESCALATE",
          "Prime order is 11 business days late and tracking provides no useful information."),

    152: ("DELIVERY_FAILURE", "AUTO_HANDLE",
          "Delivery-person behavior caused concern but the package was successfully delivered."),

    153: ("DELIVERY_TRACKING", "ESCALATE",
          "Customer says the wrong delivered order is being referenced and asks Amazon to locate the specified order."),

    154: ("PAYMENT", "ESCALATE",
          "Gift-card balance exists but the customer cannot use the payment flow to complete a purchase."),

    155: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer says multiple agents have repeatedly made assurances without resolving the issue."),

    156: ("OTHER", "AUTO_HANDLE",
          "Customer acknowledges the response and ends the interaction positively."),

    157: ("DELIVERY_DELAY", "ESCALATE",
          "Prime member is waiting several additional days for a product that should have arrived sooner."),

    158: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer alleges a previous package was effectively stolen or mishandled and is concerned about delivery handling."),

    159: ("ORDER_PRODUCT", "ESCALATE",
          "Purchased physical product has a defective charger and requires product-specific support or replacement."),

    160: ("OTHER", "AUTO_HANDLE",
          "Informational question about winners of a quiz with no support issue."),


    # =========================
    # 161-200
    # =========================

    161: ("DELIVERY_FAILURE", "ESCALATE",
          "Package was left exposed in the rain, creating potential damage and a delivery-handling issue."),

    162: ("DELIVERY_DELAY", "ESCALATE",
          "Third Amazon delivery in a few months has failed to arrive, indicating a repeated delivery problem."),

    163: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer already followed the suggested steps multiple times without getting useful information."),

    164: ("DELIVERY_DELAY", "ESCALATE",
          "Customer paid for next-day service but the promised delivery timing is not being met."),

    165: ("TECHNICAL", "AUTO_HANDLE",
          "Website is inaccessible only through a specific Firefox version or browser."),

    166: ("OTHER", "AUTO_HANDLE",
          "Complaint about a poorly synchronized visual layer in an advertisement; no defined support intent applies."),

    167: ("REFUND", "ESCALATE",
          "Customer asks why their money has not been returned."),

    168: ("DELIVERY_TRACKING", "ESCALATE",
          "Customer wants to know where the shoes are and is confused by the delivery email."),

    169: ("DELIVERY_DELAY", "ESCALATE",
          "Multiple Prime shipments have experienced delays and the customer questions the service."),

    170: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer asks for a different support contact method after the offered method did not meet their needs."),

    171: ("ORDER_CHANGE", "ESCALATE",
          "Customer explicitly complains about the order cancellation process."),

    172: ("ORDER_PRODUCT", "AUTO_HANDLE",
          "Customer provides fulfillment information with no standalone actionable request."),

    173: ("OTHER", "AUTO_HANDLE",
          "General question about how long delivery takes without reference to a specific delayed shipment."),

    174: ("DELIVERY_DELAY", "ESCALATE",
          "Customer is waiting for a product and asks Amazon to take action."),

    175: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer already contacted the carrier for a week with poor service and still needs delivery assistance."),

    176: ("PAYMENT", "ESCALATE",
          "Customer cannot use bank debit as a payment method and explicitly asks why."),

    177: ("DELIVERY_FAILURE", "ESCALATE",
          "Product arrived in unacceptable condition and cannot be used as intended."),

    178: ("OTHER", "AUTO_HANDLE",
          "Short conversational response with no actionable issue."),

    179: ("ORDER_PRODUCT", "ESCALATE",
          "Customer cannot complete checkout because no delivery times are available for the order."),

    180: ("DELIVERY_FAILURE", "ESCALATE",
          "Shipping partner failed a one-day delivery and the package was subsequently misrouted."),

    181: ("OTHER", "AUTO_HANDLE",
          "Casual acknowledgment with no support request."),

    182: ("DELIVERY_DELAY", "ESCALATE",
          "Package was expected Friday and still has not arrived."),

    183: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer observes driver behavior suggesting the package was not delivered despite being nearby."),

    184: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer says delivery was not rescheduled and reports suspected manipulation of the delivery system."),

    185: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer alleges the shipping department is mishandling or destroying orders."),

    186: ("OTHER", "ESCALATE",
          "Account is inaccessible, but the taxonomy has no dedicated account-access intent; direct account intervention is required."),

    187: ("DELIVERY_FAILURE", "ESCALATE",
          "Package appears to have been delivered to the wrong address or falsely marked as delivered."),

    188: ("DELIVERY_DELAY", "ESCALATE",
          "Product is more than a week late relative to its promised delivery date."),

    189: ("DELIVERY_FAILURE", "ESCALATE",
          "Customer suggests their shipment may have been taken or misappropriated."),

    190: ("DELIVERY_DELAY", "ESCALATE",
          "Prime orders repeatedly fail to arrive through a particular carrier."),

    191: ("DIGITAL_CONTENT", "AUTO_HANDLE",
          "Informational question about pricing of additional Prime Video channels."),

    192: ("DELIVERY_TRACKING", "ESCALATE",
          "Conflicting tracking and delivery-date information appears between the app and support section."),

    193: ("OTHER", "ESCALATE",
          "Customer cannot access a support link and the underlying issue is unknown."),

    194: ("DELIVERY_TRACKING", "ESCALATE",
          "Parcel appears to be in conflicting locations, indicating a tracking discrepancy."),

    195: ("OTHER", "AUTO_HANDLE",
          "Customer provides a shipment-status update with no explicit support problem."),

    196: ("OTHER", "AUTO_HANDLE",
          "Customer indicates that the feared delivery problem did not actually occur."),

    197: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Customer followed the suggested process but nobody called or provided help."),

    198: ("DELIVERY_DELAY", "ESCALATE",
          "Package was rescheduled despite being expected by 8pm that day."),

    199: ("SUPPORT_COMPLAINT", "ESCALATE",
          "Repeated issue remains unresolved, promised refund or fulfillment has not occurred, and support cannot explain the problem."),

    200: ("DELIVERY_DELAY", "ESCALATE",
          "Delivery was promised by 8pm but the customer is still waiting shortly before the deadline."),
}


def main():
    df = pd.read_csv(INPUT)

    for example_num, (intent, escalation, reason) in labels.items():
        idx = example_num - 1

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_escalation"] = escalation
        df.loc[idx, "gold_reason"] = reason

    df.to_csv(OUTPUT, index=False)

    print("Applied reviewed labels to examples 121-200.")
    print()

    print(
        df.iloc[120:200][
            ["tweet_id", "gold_intent", "gold_escalation", "gold_reason"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()