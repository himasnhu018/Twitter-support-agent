import pandas as pd
from pathlib import Path

INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/annotation_queue.txt"

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

ESCALATION = [
    "AUTO_HANDLE",
    "ESCALATE",
]


def main():
    df = pd.read_csv(INPUT)

    with open(OUTPUT, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("AMAZONHELP GOLDEN SET ANNOTATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("INTENTS:\n")
        for i, intent in enumerate(INTENTS, 1):
            f.write(f"{i:2}. {intent}\n")

        f.write("\nESCALATION:\n")
        f.write("  1. AUTO_HANDLE\n")
        f.write("  2. ESCALATE\n")

        f.write("""
ANNOTATION RULES:

- Label the PRIMARY actionable customer intent.
- Emotion does not determine intent.
- Use the most specific applicable intent.
- Account information is not automatically an intent.
- Intent and escalation are separate labels.
- Use OTHER when the issue is genuinely outside the taxonomy,
  already resolved with no remaining request, or lacks enough
  context to determine the intent.
- Escalate when resolution requires account-specific information,
  sensitive information, investigation, consequential action,
  or human intervention.
- Do not escalate merely because the customer is angry.
- Do not infer fraud/seller problems without evidence.

For ambiguous cases, explain the ambiguity in notes.

""" + "=" * 80 + "\n\n")

        for i, row in df.iterrows():

            number = i + 1

            f.write(f"EXAMPLE {number}\n")
            f.write("-" * 80 + "\n")

            f.write(f"Tweet ID: {row['tweet_id']}\n\n")

            f.write("CUSTOMER MESSAGE:\n")
            f.write(str(row["text"]).strip() + "\n\n")

            f.write("YOUR LABEL:\n")
            f.write("Intent: ______________________________\n")
            f.write("Escalation: ___________________________\n")
            f.write("Reason: _______________________________\n")
            f.write("Notes: ________________________________\n\n")

            f.write("=" * 80 + "\n\n")

    print(f"Created annotation queue with {len(df)} examples.")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()