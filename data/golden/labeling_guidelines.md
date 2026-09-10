# AmazonHelp Intent Labeling Guidelines

## Purpose

Each golden-set example represents a single customer message.

Annotate the customer's **primary actionable intent**.

Do not label based only on keywords or emotional tone.

The intent describes **what the customer needs help with**.

The escalation label separately describes whether the AI agent
can safely handle the request automatically or should escalate it.

---

# Intent Definitions

## 1. DELIVERY_DELAY

Use when the customer's primary issue is that a delivery is
late, delayed, or expected later than promised.

Examples:

- "My package was supposed to arrive yesterday."
- "Why has my delivery been delayed?"
- "It keeps getting pushed back."
- "I paid for one day shipping but it will take a week."

Do NOT use for:

- asking where the package currently is -> DELIVERY_TRACKING
- failed/incorrect delivery -> DELIVERY_FAILURE
- changing delivery speed/date/address -> ORDER_CHANGE

---

## 2. DELIVERY_TRACKING

Use when the primary request is to locate or understand the
current tracking status of a shipment.

Examples:

- "Where is my package?"
- "Tracking hasn't updated."
- "Can you tell me where my parcel is?"
- "Can I pick it up from the local depot?"
- "It says out for delivery but I don't know where it is."

Use this when the customer's main question is about the
**current location/status of the shipment**.

If the expected delivery date has already passed and the
main complaint is lateness, prefer DELIVERY_DELAY.

---

## 3. DELIVERY_FAILURE

Use when the delivery failed or was completed incorrectly.

Examples:

- package marked undeliverable
- courier failed to deliver
- package delivered to the wrong place
- package was damaged during delivery
- courier behaved improperly
- delivery attempt failed
- delivery is unavailable to the customer's location

Examples:

- "The driver threw my parcel at the door."
- "My package says undeliverable."
- "The courier delivered it to the wrong house."

---

## 4. ORDER_CHANGE

Use when the customer wants to modify an existing order.

Examples:

- change delivery address
- change delivery speed
- change delivery date
- change an order detail
- modify an existing order
- request expedited delivery for an existing order

Examples:

- "I ordered it to the wrong address."
- "Can I change the delivery method?"
- "Can you change my order?"

Do NOT use for:

- simply asking where an order is -> DELIVERY_TRACKING
- asking why an order is late -> DELIVERY_DELAY
- asking to return an item -> RETURN
- asking for money back -> REFUND

---

## 5. RETURN

Use for return eligibility, return process, or questions
about sending an item back.

Examples:

- "Can I return this?"
- "Why isn't this product eligible for return?"
- "How do I send this back?"
- "I want to return this product."

Use RETURN when the main issue concerns the **return itself**.

If the item has already been returned and the customer is
waiting for money, use REFUND.

---

## 6. REFUND

Use when the primary issue concerns money being returned
after a return, cancellation, or other eligible transaction.

Examples:

- "Where is my refund?"
- "When will I get my money back?"
- "My refund hasn't arrived."
- "I returned the item but haven't received the refund."

Prefer REFUND over RETURN when the return has already happened
and the remaining issue is receiving the money.

---

## 7. PAYMENT

Use for payment or billing problems.

Examples:

- payment failed
- card charged/not charged
- payment method issue
- billing problem
- EMI issue
- payment authorization issue
- unexpected charge related to payment

Examples:

- "My card wasn't charged."
- "Why is my payment failing?"
- "It keeps asking me to update my payment method."

If the primary issue is an unauthorized/fraudulent transaction,
consider SELLER_FRAUD and apply escalation rules.

---

## 8. PRIME

Use when the primary issue specifically concerns Amazon Prime
membership or Prime benefits.

Examples:

- Prime membership problem
- Prime subscription
- Prime membership cancellation
- Prime-specific benefit
- Prime-specific delivery entitlement
- Prime membership charge/problem

Examples:

- "Why am I being charged for Prime?"
- "My Prime membership isn't working."
- "Why don't I get next-day delivery as a Prime member?"

If the customer happens to be a Prime member but the actual
issue is an ordinary delivery problem, use the relevant
DELIVERY intent instead.

---

## 9. TECHNICAL

Use for a technical malfunction of an Amazon application,
website, supported device, or digital service.

Examples:

- app error
- website malfunction
- application not working
- video playback malfunction
- device malfunction
- repeated technical error
- feature not functioning correctly

Examples:

- "The app keeps crashing."
- "The video won't play."
- "I've tried two devices and it still says unavailable."

Use TECHNICAL when there is an actual **technical problem**.

Do NOT automatically use TECHNICAL simply because the issue
involves a digital product.

For informational questions about digital content, use
DIGITAL_CONTENT.

---

## 10. DIGITAL_CONTENT

Use for informational or access questions involving Amazon
digital content or digital services, especially when there is
no technical malfunction.

Examples:

- Prime Video content questions
- movie rental duration
- digital content availability
- questions about watching rented movies
- content access rules
- questions about availability by location

Examples:

- "How long can I watch a rented movie?"
- "Is this movie available in my country?"
- "How long does a rental last?"

If the customer reports that the digital service is
malfunctioning, use TECHNICAL instead.

---

## 11. SELLER_FRAUD

Use for marketplace seller problems, fraudulent sellers,
scams, counterfeit products, suspicious transactions,
or serious seller-related misconduct.

Examples:

- fraudulent third-party seller
- counterfeit/fake product
- seller disappeared
- seller refuses return/refund
- seller does not respond
- suspicious marketplace transaction
- customer believes they were scammed by a seller

Examples:

- "The seller disappeared with my money."
- "I received a fake product."
- "This third-party seller is fraudulent."

Use this category when the **seller/fraud aspect is the
primary problem**.

---

## 12. ORDER_PRODUCT

Use for general order or product questions that do not fit
a more specific category.

Examples:

- product availability
- product information
- general order information
- pre-order questions
- product availability after pre-order
- order cancellation/status when no more specific intent applies

Examples:

- "Why was my pre-order cancelled?"
- "When will this product be available?"
- "What happened to my order?"

Prefer RETURN, REFUND, PAYMENT, DELIVERY, or ORDER_CHANGE
when one of those is clearly the customer's primary issue.

---

## 13. SUPPORT_COMPLAINT

Use when the primary issue is failure of customer support
rather than the underlying product/order problem.

Examples:

- repeated unsuccessful support contacts
- requesting a supervisor
- complaint about support response
- requesting direct human contact
- customer says previous agents failed to resolve the issue
- customer is primarily asking for better support

Examples:

- "I've contacted customer service five times and nobody helped."
- "I need to speak to a supervisor."
- "Your support team keeps giving me the same answer."

If there is a clear underlying actionable issue, label that
underlying issue instead.

Example:

"I've contacted you five times and my refund still hasn't arrived."

Label:

REFUND

Not:

SUPPORT_COMPLAINT

---

## 14. OTHER

Use only when none of the above intents reasonably fit.

Use OTHER for:

- irrelevant/non-support messages
- messages that are purely conversational
- already-resolved messages with no remaining actionable issue
- messages whose context is genuinely insufficient to determine
  an intent
- topics outside the supported taxonomy

Do not use OTHER simply because the message is difficult.

When using OTHER, explain the reason in the annotation notes.

---

# Annotation Rules

## Rule 1: Label the primary actionable intent

Each message receives exactly one intent.

Choose the problem that represents the customer's primary
support need.

Example:

"I've contacted you five times and my refund still hasn't arrived."

Label:

REFUND

Not:

SUPPORT_COMPLAINT

---

## Rule 2: Emotion does not determine intent

Anger, frustration, sarcasm, or insults do not automatically
make something SUPPORT_COMPLAINT.

Example:

"Your delivery service is terrible and my package is still late."

Label:

DELIVERY_DELAY

Not:

SUPPORT_COMPLAINT.

---

## Rule 3: Account information is not automatically an intent

A customer's need for account-specific assistance does not
change the underlying intent.

Example:

"My package hasn't arrived and you need to check my account."

Intent:

DELIVERY_DELAY

Escalation:

ESCALATE

---

## Rule 4: Use the most specific applicable intent

Prefer:

DELIVERY_TRACKING

over:

DELIVERY_DELAY

when the customer specifically asks where the package is.

Prefer:

REFUND

over:

RETURN

when the return has already happened and the problem is
receiving the money.

Prefer:

TECHNICAL

over:

DIGITAL_CONTENT

when the customer reports an actual technical malfunction.

---

## Rule 5: If two intents appear, choose the primary problem

Example:

"My package is late and I want to cancel the order."

If cancellation/modification is the main requested action:

ORDER_CHANGE

If the customer is primarily complaining about the delay:

DELIVERY_DELAY.

Use the customer's explicit requested action when it is clear.

---

## Rule 6: Resolved messages

If the customer says the issue has already been resolved and
does not request further help:

OTHER

Example:

"Already sorted with the help desk."

Intent:

OTHER

Escalation:

AUTO_HANDLE

---

## Rule 7: Insufficient context

If a message cannot reasonably be classified because the
actual problem is missing from the message:

OTHER

Example:

"I have some now after 5 1/2 hrs trying!"

If the context required to understand the issue is absent,
do not guess the intent.

Use:

OTHER

and explain:

"Insufficient context to determine the underlying issue."

---

# Escalation Annotation

Intent and escalation are separate labels.

The intent describes:

> WHAT does the customer need?

The escalation label describes:

> CAN our AI safely handle this without account access
> or human intervention?

---

## AUTO_HANDLE

Use AUTO_HANDLE when a safe, evidence-grounded response can
reasonably be provided without accessing customer-specific
information or performing a consequential action.

Examples:

- general informational questions
- general delivery-option questions
- general digital-content information
- basic product information
- general policy/process explanation

---

## ESCALATE

Use ESCALATE when the request requires account-specific
information, verification, investigation, sensitive information,
or human intervention.

Common escalation triggers:

### 1. Account-specific action

Examples:

- checking a specific order
- changing an existing order
- accessing account details
- checking account status

### 2. Payment or financial investigation

Examples:

- disputed charge
- missing refund requiring account lookup
- unauthorized transaction
- payment investigation

### 3. Fraud or suspicious activity

Examples:

- suspected scam
- fraudulent seller
- counterfeit product
- unauthorized account activity

### 4. Delivery investigation

Examples:

- missing package requiring carrier investigation
- repeated delivery failure
- damaged package
- incorrect delivery

### 5. Sensitive/private information

If the customer needs to provide or verify private
account/order information, escalate to an appropriate
secure support channel.

### 6. Repeated unresolved support

If the customer has already tried support repeatedly
without resolution and requests human intervention,
escalate.

### 7. Consequential action

Escalate when the customer asks for an action that the
AI cannot safely perform, such as modifying an order,
issuing a refund, or changing account information.

---

# Important Escalation Principle

Do NOT escalate simply because the customer's message
contains an order number.

Escalate when resolving the request actually requires
accessing or acting on customer-specific information.

Similarly, do not automatically escalate every angry
customer.

The decision should be based on **operational risk and
required capability**, not emotional tone.

---

# Annotation Format

For every example record:

1. Intent
2. Escalation
3. Short reason
4. Optional notes for ambiguity

Example:

Intent:
DELIVERY_TRACKING

Escalation:
ESCALATE

Reason:
Requires checking the customer's shipment status.

Notes:
Customer asks where the package currently is.

---

# Quality Rules

Before finalizing a label, ask:

1. What is the customer actually asking for?
2. Is there a more specific intent?
3. Am I being influenced by emotional language?
4. Does the request require customer/account-specific data?
5. Could the AI safely answer without taking an external action?
6. If I selected OTHER, can I clearly explain why?

When uncertain between two intents, prefer the more specific
intent and document the ambiguity in notes.