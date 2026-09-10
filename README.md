# AmazonHelp AI Support Agent

A retrieval-augmented customer-support agent built for the Hiver SDE Intern take-home assignment.

The system analyzes an incoming Amazon customer-support message, classifies its intent, retrieves relevant historical AmazonHelp support cases, drafts a grounded response, and decides whether the case can be auto-handled or should be escalated.

---

## 1. Problem

Customer-support agents repeatedly handle similar issues such as:

- delayed deliveries
- tracking requests
- failed deliveries
- order changes
- returns
- refunds
- payments
- Prime
- technical problems
- digital content
- seller fraud
- product questions
- support complaints

The goal is to use historical AmazonHelp conversations to assist support agents while avoiding unsupported claims and unsafe automatic handling.

---

## 2. System Architecture

```text
                         Customer Message
                                |
                                v
                    +----------------------+
                    | Gemini Intent         |
                    | Classification        |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Sentence Transformer |
                    | all-MiniLM-L6-v2     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | FAISS Vector Search   |
                    | Top-K Historical      |
                    | Support Cases         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Gemini Response       |
                    | Generation             |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Gemini Escalation     |
                    | Decision               |
                    +----------+-----------+
                               |
                               v
                  +--------------------------+
                  | AUTO_HANDLE / ESCALATE   |
                  +--------------------------+


The end-to-end pipeline is implemented in:

scripts/agent.py

The interactive UI is implemented in:

app.py
4. Tech Stack
Component	Technology
Language	Python 3.12
LLM	Gemini gemini-3.5-flash-lite
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector database	FAISS
Data processing	pandas
Data format	CSV / Parquet
Validation	Pydantic
Web UI	Streamlit
Evaluation	Python + Gemini LLM judge
Platform	Windows / PowerShell
5. Dataset

Dataset:

thoughtvector/customer-support-on-twitter

Selected brand:

AmazonHelp

The original dataset contains Twitter customer-support conversations involving customers and support brands.

Relevant columns include:

tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id

Interpretation:

inbound = True
    -> customer message

inbound = False
    -> brand/support response
6. Why AmazonHelp?

AmazonHelp was selected because it provides:

high interaction volume
many customer-support scenarios
diverse support intents
sufficient customer-agent interactions for retrieval
enough examples to construct a meaningful evaluation set

The direct AmazonHelp extraction contained:

AmazonHelp brand tweets:       169,840
Customer messages:             100,503
Direct interactions:           270,343

Additional conversation relationships were reconstructed to improve contextual coverage.

7. Conversation Reconstruction

Tweet relationships were reconstructed using:

response_tweet_id
in_response_to_tweet_id

Because response_tweet_id can contain multiple comma-separated IDs, a single tweet can have multiple response relationships.

The reconstructed conversation dataset contains approximately:

Total conversation tweets:     358,973
AmazonHelp tweets:             169,840
Customer tweets:               189,133
Conversation components:        85,087

Conversation statistics:

Average tweets/component:        4.22
Median tweets/component:         3
Maximum component size:        206

The reconstruction should be interpreted as conversation components, not guaranteed complete support cases.

The available tweet relationships do not necessarily provide the full transitive conversation graph.

8. Support Case Construction

For retrieval, the system constructs customer-agent support cases.

Each case contains:

customer_tweet_id
agent_tweet_id
customer_text
agent_response
component_id
customer_created_at
agent_created_at

Final leakage-safe support corpus:

Support cases:                  168,141
Unique customer messages:       154,361
Unique conversations:            84,883

The retrieval index contains:

168,141

customer-agent support cases.

9. Intent Taxonomy

The system uses 14 intents.

1. DELIVERY_DELAY

Package/order is late or has missed an expected delivery time.

2. DELIVERY_TRACKING

Customer wants tracking or shipment status information.

3. DELIVERY_FAILURE

Package was not delivered, was lost, or was marked delivered incorrectly.

4. ORDER_CHANGE

Customer wants to cancel, modify, edit, or change an order/address.

5. RETURN

Customer wants to return or exchange a product.

6. REFUND

Customer is asking about a refund or missing refund.

7. PAYMENT

Payment, charge, billing, card, or payment authorization issue.

8. PRIME

Amazon Prime membership or subscription issue.

9. TECHNICAL

Amazon app, website, device, or service malfunction.

10. DIGITAL_CONTENT

Informational question about Prime Video, Kindle, ebooks, movies, or digital content.

11. SELLER_FRAUD

Suspected seller fraud, scam, counterfeit, or fraudulent seller behavior.

12. ORDER_PRODUCT

Question about a product, availability, selection, or ordering a product.

13. SUPPORT_COMPLAINT

Complaint primarily about Amazon customer support/service.

14. OTHER

Irrelevant, social, already-resolved, or insufficient-context message.

10. Important Taxonomy Decisions
Digital content vs technical

A question such as:

Is this movie available on Prime Video?

is:

DIGITAL_CONTENT

A malfunction such as:

Prime Video keeps crashing.

is:

TECHNICAL
Delivery delay vs delivery failure

A package that is simply late is:

DELIVERY_DELAY

A package that is missing, lost, or incorrectly marked delivered is:

DELIVERY_FAILURE
Order change

Cancellation, modification, address change, etc. are grouped into:

ORDER_CHANGE

rather than creating separate change intents.

Support complaint

If the customer complains about support but has a clearly identifiable underlying issue, the underlying issue takes priority.

For example:

Your support is terrible and my package still hasn't arrived.

is primarily a delivery problem rather than merely:

SUPPORT_COMPLAINT
11. Golden Evaluation Set

A manually labeled golden set of:

200 examples

was created.

The set was designed to cover:

common intents
less frequent intents
different message lengths
different support situations
ambiguous cases
irrelevant/insufficient-context messages

Each example contains an intent label and escalation label.

Escalation distribution:

AUTO_HANDLE: 68
ESCALATE:    132

Intent distribution:

OTHER                41
DELIVERY_DELAY       34
DELIVERY_FAILURE     31
SUPPORT_COMPLAINT    26
ORDER_PRODUCT        17
DELIVERY_TRACKING    12
TECHNICAL             9
PAYMENT               8
ORDER_CHANGE          7
DIGITAL_CONTENT       5
REFUND                5
PRIME                 3
RETURN                2
SELLER_FRAUD          0

The absence of SELLER_FRAUD examples in this 200-example sample means its per-class performance cannot be meaningfully estimated from this evaluation.

12. Leakage Prevention

Retrieval evaluation must not retrieve the exact historical example being evaluated.

Therefore the project creates a leakage-safe retrieval corpus.

All conversation components containing golden examples were removed from the retrieval corpus.

Results:

Original conversation rows:              358,973
Golden examples:                             200
Components containing golden examples:       200
Rows removed:                              1,475
Remaining conversation rows:             357,498

The resulting support-case corpus contains:

168,141

customer-agent cases.

This prevents evaluation examples from being directly retrieved from the knowledge base.

13. Embeddings

The embedding model is:

sentence-transformers/all-MiniLM-L6-v2

Embeddings are normalized before indexing.

The resulting FAISS index uses:

IndexFlatIP

Because the vectors are normalized, inner product corresponds to cosine similarity.

The final index contains:

168,141 vectors
384 dimensions

Persisted files:

data/processed/amazon_support.faiss
data/processed/amazon_support_metadata.parquet

The embedding/index build is intentionally performed only once.

14. Retrieval

For each incoming customer message:

The message is embedded.
The embedding is normalized.
FAISS searches the support corpus.
The top 5 historical cases are returned.
Their customer messages and historical responses are passed as evidence to the response generator.

Default:

k = 5

Retrieval diagnostics:

Top-1 mean cosine similarity:   0.7847
Top-5 mean similarity:          0.7642

These similarity scores are diagnostics only.

They should not be interpreted as retrieval accuracy.

A high similarity score does not guarantee that the retrieved historical response is correct for the current customer.

15. Response Generation

Gemini:

gemini-3.5-flash-lite

The response generator receives:

Customer message
Predicted intent
Intent definition
Top retrieved historical cases

Historical responses are treated as:

evidence

rather than:

authoritative current policy

The generator is instructed not to:

invent order information
invent tracking numbers
invent delivery dates
invent refund amounts
invent account details
invent policy guarantees
claim that it accessed the customer's account
claim that it performed an action
copy identifiers from historical cases
copy URLs from historical cases
pretend to resolve account-specific issues
16. Escalation Policy

The system has two possible decisions:

AUTO_HANDLE
ESCALATE
AUTO_HANDLE

Used when the customer's immediate need can be safely addressed using:

general information
safe troubleshooting
no account-specific lookup
no financial investigation
no order-specific investigation
no consequential action
no sensitive information
ESCALATE

Used when the customer needs:

specific order investigation
shipment investigation
account investigation
payment investigation
refund investigation
specific package status
specific refund status
specific payment status
cancellation
order modification
consequential action
fraud investigation
counterfeit investigation
scam investigation
investigation after failed troubleshooting
sensitive information handling
resolution requiring unavailable evidence

Important rule:

A generic instruction such as "check your tracking" does not make a customer-specific delivery problem safe to auto-handle.

If Amazon must inspect the customer's actual shipment, the case is escalated.

17. End-to-End Agent

The complete implementation is:

scripts/agent.py

Main class:

AmazonSupportAgent

The main pipeline is:

agent = AmazonSupportAgent(k=5)

result = agent.run(
    "My package is two days late"
)

The returned result contains:

message
intent
intent_confidence
intent_reason
reply
grounding
response_confidence
decision
escalation_reason
escalation_confidence
retrieved_cases
18. Example

Input:

My package was supposed to arrive yesterday but it still hasn't arrived.

Example behavior:

Intent:
DELIVERY_DELAY

Intent confidence:
0.99

Draft:
I'm sorry for the delay in receiving your package!
What does your most recent tracking information show?

Decision:
ESCALATE

Reason:
The customer is asking about the specific delivery status
of their delayed package, which requires inspecting the
specific order/shipment.

The exact LLM wording may vary slightly.

19. Evaluation Results

Evaluation was performed on:

200 golden examples
19.1 Intent Classification
System	Accuracy	Macro F1	Weighted F1
Gemini	66.5%	60.9%	65.9%
Semantic baseline	27.0%	16.6%	27.1%
Rule baseline	26.5%	15.8%	19.5%

Gemini substantially outperformed both baselines.

The semantic baseline and rule baseline demonstrate that simple similarity/rule-based approaches are considerably weaker on the 14-class taxonomy.

20. Baselines

Two baselines were evaluated.

Semantic baseline

The customer message is compared semantically against intent descriptions/examples and the closest intent is selected.

Results:

Accuracy: 27.0%
Macro F1: 16.6%
Rule baseline

A keyword/rule-based classifier was used.

Results:

Accuracy: 26.5%
Macro F1: 15.8%

The rule baseline strongly over-predicted:

OTHER

This demonstrates the brittleness of simple keyword rules for noisy support messages.

21. Response Quality Evaluation

An LLM-as-judge evaluation was performed on all:

200

generated responses.

Dimensions:

relevance
groundedness
helpfulness
safety
historical grounding
overall quality

Results:

Dimension	Mean
Relevance	4.425 / 5
Groundedness	4.855 / 5
Helpfulness	4.070 / 5
Safety	4.940 / 5
Historical grounding	4.475 / 5
Overall	4.553 / 5

Percentage scoring at least 4:

Relevance:           88.5%
Groundedness:        98.0%
Helpfulness:         79.0%
Safety:              98.0%
Historical grounding:92.0%

No response received a safety score of 2 or below.

22. Human Evaluation

A separate:

40-example

human response review was created.

Human reviewers scored response dimensions on a 1–5 scale.

The human review is used to compare human judgments against the LLM judge rather than treating the LLM judge as ground truth.

The agreement analysis is generated by:

scripts/analyze_human_agreement.py

Output:

data/golden/human_agreement_results.csv
data/golden/human_agreement_enriched.csv

The agreement analysis includes:

judge mean
human mean
exact agreement
within-1 agreement
mean absolute error
overall agreement
judge/human bias
largest disagreements
23. Escalation Evaluation

Results on the 200-example golden set:

Metric	Result
Accuracy	85.0%
Escalation recall	89.4%
AUTO_HANDLE precision	78.8%
Unsafe AUTO_HANDLE rate	10.6%

Confusion matrix:

                     PREDICTED
                  AUTO      ESCALATE

GOLD AUTO           52          16

GOLD ESCALATE      14         118

Escalation recall:

118 / 132 = 89.4%

Unsafe auto-handle rate:

14 / 132 = 10.6%
24. Why Unsafe Auto-Handling Matters

For a support agent, overall accuracy is not sufficient.

An unnecessary escalation usually costs time.

An unsafe auto-handle can instead:

incorrectly imply resolution
fail to investigate an actual customer issue
cause customer frustration
create operational risk

Therefore this project explicitly tracks:

Unsafe AUTO_HANDLE rate

as a key safety metric.

25. Failure Analysis

The evaluation identified several important failure modes.

1. Unsupported investigation claims

Some responses incorrectly implied that:

a specialist team is currently investigating the issue

even though the system cannot actually perform such an investigation.

This is a grounding failure.

2. Intent confusion

Common confusion pairs included:

DELIVERY_FAILURE → DELIVERY_DELAY
ORDER_PRODUCT → OTHER
DELIVERY_TRACKING → DELIVERY_DELAY
OTHER → TECHNICAL
DELIVERY_FAILURE → SUPPORT_COMPLAINT

These are areas for taxonomy and classifier improvement.

3. Unsafe auto-handling

Some specific delivery problems were incorrectly classified as safe to auto-handle.

These are particularly important because the customer may require an actual shipment investigation.

4. Language mismatch

Some messages were answered in a language that did not match the customer message.

This can make an otherwise grounded answer unusable.

5. Low helpfulness

Some responses were safe and grounded but too generic to provide a useful next step.

26. Failure Statistics

From the 200-example evaluation:

Intent errors:                    67
Unsafe auto-handles:              14
Unnecessary escalations:          16

Low relevance (<=2):               7
Low groundedness (<=2):            3
Low helpfulness (<=2):            10
Low safety (<=2):                  0
Low historical grounding (<=2):    3

Overall response score <3.5:      10

The failure analysis artifact is:

data/golden/failure_analysis.csv
27. What Is Misleading About My Headline Number?

The headline intent accuracy is:

66.5%

This number should not be interpreted as:

"The production support agent is 66.5% correct."

There are several reasons.

1. Small evaluation set

The intent accuracy is measured on only:

200

hand-labeled examples.

This is useful for a prototype but not sufficient to establish production-level accuracy.

2. Historical dataset

The data comes from historical Twitter support interactions.

It may not represent:

current Amazon policies
current customer behavior
current product catalog
current support workflows
modern support channels
3. Intent accuracy is only one component

The agent also performs:

retrieval
response generation
safety handling
escalation

A correct intent does not automatically imply a correct response.

4. LLM judge score is not customer satisfaction

The:

4.553 / 5

response score is an LLM evaluation score.

It is not:

customer satisfaction

or:

real-world resolution rate
5. Retrieval similarity is not retrieval accuracy

The:

0.7847

top-1 mean similarity is a diagnostic.

It does not mean:

78.47% retrieval accuracy
6. Escalation accuracy hides asymmetric risk

A model can have good overall accuracy while still making unsafe AUTO_HANDLE decisions.

For this reason:

Unsafe auto-handle rate

is tracked explicitly.

28. Reproducibility

The expensive embedding/index construction is performed once and persisted.

The following artifacts are reused:

data/processed/amazon_support.faiss
data/processed/amazon_support_metadata.parquet

Evaluation caches are also persisted under:

data/golden/

The final evaluation was measured using:

Measure-Command { python scripts\build_final_evaluation.py }

Measured runtime:

1.1917503 seconds

Therefore the reproducible evaluation is comfortably below the assignment requirement of:

15 minutes

Important:

Do not rebuild the FAISS embeddings when running the final evaluation.

The original embedding/index construction was a one-time preprocessing step.

29. Installation
Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd hiver-support-agent
Create virtual environment
python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

If PowerShell blocks activation, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then:

.venv\Scripts\Activate.ps1
30. Install Dependencies

If requirements.txt is present:

pip install -r requirements.txt

Otherwise install the required packages:

pip install pandas
pip install pyarrow
pip install numpy
pip install sentence-transformers
pip install faiss-cpu
pip install google-genai
pip install pydantic
pip install streamlit
31. Gemini API Key

The system requires a Gemini API key.

Set it as an environment variable.

PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"

Verify that the variable exists:

if ($env:GEMINI_API_KEY) {
    Write-Host "GEMINI_API_KEY is set"
} else {
    Write-Host "GEMINI_API_KEY is NOT set"
}

Do not commit the API key to Git.

32. Running the CLI Agent

From the project root:

python scripts\agent.py --message "My package is two days late"

Example:

python scripts\agent.py --message "The Amazon app keeps crashing"

Example:

python scripts\agent.py --message "I want to know where my package is"

Optional retrieval depth:

python scripts\agent.py --message "My package is late" --k 5
33. Running the Streamlit Demo

Start Streamlit:

streamlit run app.py

The terminal should display:

Local URL: http://localhost:8501

Open:

http://localhost:8501

The interface provides:

customer message input
predicted intent
intent confidence
intent reasoning
AUTO_HANDLE / ESCALATE
escalation reasoning
response confidence
generated draft response
retrieved historical support cases
similarity scores
historical customer messages
historical AmazonHelp responses
34. Example Streamlit Input

Try:

My package was supposed to arrive yesterday but it still hasn't arrived.

The UI should produce an output broadly similar to:

Intent:
DELIVERY_DELAY

Decision:
ESCALATE

Draft response:
I'm sorry for the delay in receiving your package!
What does your most recent tracking information show?

The exact response can vary because Gemini generates the final text.

35. Data Processing Pipeline

The main preprocessing scripts are:

scripts/inspect_data.py
scripts/analyze_brands.py
scripts/test_relationships.py
scripts/inspect_conversation.py
scripts/extract_brand_data.py
scripts/build_amazon_threads.py
scripts/analyze_amazon_threads.py
scripts/sample_amazon_conversations.py
scripts/build_leakage_safe_corpus.py
scripts/build_support_cases.py
scripts/analyze_support_cases.py

A typical preprocessing flow is:

Raw Kaggle dataset
        |
        v
Brand analysis
        |
        v
AmazonHelp extraction
        |
        v
Conversation reconstruction
        |
        v
Leakage-safe corpus
        |
        v
Customer-agent support cases
        |
        v
Embeddings
        |
        v
FAISS index
36. Important Processing Commands
Extract AmazonHelp data
python scripts\extract_brand_data.py
Build conversation components
python scripts\build_amazon_threads.py
Analyze conversations
python scripts\analyze_amazon_threads.py
Build leakage-safe corpus
python scripts\build_leakage_safe_corpus.py
Build support cases
python scripts\build_support_cases.py
Analyze support cases
python scripts\analyze_support_cases.py
37. Evaluation Commands
Semantic baseline
python scripts\evaluate_semantic_classifier.py
Rule baseline
python scripts\evaluate_rule_baseline.py
Response generation evaluation
python scripts\evaluate_responses.py
LLM response judge
python scripts\judge_responses.py
Escalation evaluation
python scripts\evaluate_escalation.py
Human review generation
python scripts\create_human_response_review.py
Human agreement analysis
python scripts\analyze_human_agreement.py
Failure analysis
python scripts\analyze_failures.py
Final evaluation
python scripts\build_final_evaluation.py
38. Final Evaluation Runtime

To reproduce the final evaluation runtime:

Measure-Command { python scripts\build_final_evaluation.py }

Expected behavior:

FINAL EVALUATION COMPLETE

and approximately:

1–2 seconds

when using the persisted artifacts and cached results.

39. Final Evaluation Artifacts

The main final outputs are:

data/golden/final_evaluation_summary.csv
data/golden/final_evaluation_report.md

Other important evaluation artifacts include:

data/golden/golden_set.csv
data/golden/gemini_classifier_predictions.csv
data/golden/semantic_classifier_predictions.csv
data/golden/rule_baseline_predictions.csv
data/golden/retrieval_evaluation_results.csv
data/golden/retrieval_top1.csv
data/golden/end_to_end_responses.csv
data/golden/response_judge_predictions.csv
data/golden/escalation_predictions.csv
data/golden/human_response_review.csv
data/golden/human_agreement_results.csv
data/golden/human_agreement_enriched.csv
data/golden/failure_analysis.csv
40. Project Structure
hiver-support-agent/
│
├── app.py
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── agent.py
│   ├── inspect_data.py
│   ├── analyze_brands.py
│   ├── test_relationships.py
│   ├── inspect_conversation.py
│   ├── extract_brand_data.py
│   ├── build_amazon_threads.py
│   ├── analyze_amazon_threads.py
│   ├── sample_amazon_conversations.py
│   ├── build_leakage_safe_corpus.py
│   ├── build_support_cases.py
│   ├── analyze_support_cases.py
│   ├── semantic_intent_classifier.py
│   ├── evaluate_semantic_classifier.py
│   ├── evaluate_rule_baseline.py
│   ├── generate_response.py
│   ├── evaluate_responses.py
│   ├── judge_responses.py
│   ├── evaluate_escalation.py
│   ├── create_human_response_review.py
│   ├── analyze_human_agreement.py
│   ├── analyze_failures.py
│   └── build_final_evaluation.py
│
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   │
│   ├── processed/
│   │   ├── amazonhelp_tweets.parquet
│   │   ├── amazonhelp_conversation_tweets.parquet
│   │   ├── amazon_retrieval_corpus.parquet
│   │   ├── amazon_support_cases.parquet
│   │   ├── amazon_support.faiss
│   │   └── amazon_support_metadata.parquet
│   │
│   └── golden/
│       ├── golden_set.csv
│       ├── labeling_guidelines.md
│       ├── final_evaluation_summary.csv
│       ├── final_evaluation_report.md
│       ├── failure_analysis.csv
│       └── evaluation caches
│
└── .venv/
41. Security

Never commit:

GEMINI_API_KEY

to Git.

Do not create a .env file containing the API key unless .env is included in .gitignore.

Recommended .gitignore:

.venv/
__pycache__/
*.pyc
.env
*.log

If the raw dataset is too large for the repository, keep it outside Git and document how to obtain it.

42. Production Safety Considerations

This prototype should not automatically send every generated response to customers.

A production system should include:

Customer message
        |
        v
Intent
        |
        v
Retrieval
        |
        v
Response draft
        |
        v
Safety validation
        |
        +------ unsafe ------> Human agent
        |
        v
Escalation policy
        |
        +------ ESCALATE ---> Human agent
        |
        v
AUTO_HANDLE

Recommended production safeguards:

deterministic policy checks
human approval for sensitive actions
current-policy knowledge source
account/order-system integration
audit logging
model/version tracking
rate limiting
prompt injection protection
PII filtering
regression tests
monitoring
rollback capability
43. Limitations
Historical data

Historical Twitter conversations may contain outdated:

policies
URLs
product information
workflows
support processes

Therefore historical responses should not be blindly copied.

No operational system access

The prototype cannot access:

customer accounts
orders
shipment systems
payment systems
refund systems
transaction records

Therefore account-specific operational problems should generally be escalated.

Conversation reconstruction

The reconstructed conversations are best treated as conversation components.

The available tweet relationships do not guarantee complete transitive conversation reconstruction.

Language coverage

The initial taxonomy and evaluation are primarily designed around English-language support messages.

A production system should explicitly detect and handle language before generating a response.

Limited rare-intent evaluation

Some rare intents have few or zero examples in the 200-example golden set.

Therefore per-class metrics for these intents have high uncertainty or are not estimable.

LLM variability

Gemini is used for:

intent classification
response generation
escalation

Production deployment should therefore use:

pinned model versions
regression testing
monitoring
deterministic safety checks
44. One-Week Production Roadmap
Day 1 — Taxonomy and evaluation
Expand the golden set from 200 to 500–1,000 examples.
Increase coverage of rare intents.
Add multiple human reviewers.
Validate the taxonomy with support-agent feedback.
Day 2 — Retrieval improvements

Evaluate:

alternative embedding models
hybrid BM25 + semantic retrieval
metadata filtering
reranking
intent-aware retrieval

Add explicit retrieval relevance labels.

Day 3 — Response quality

Improve:

grounding constraints
unsupported-claim detection
language matching
response style
actionable next steps

Add deterministic validation before displaying or sending a response.

Day 4 — Escalation safety

Focus on reducing:

unsafe AUTO_HANDLE

Add:

deterministic escalation rules
sensitive-information detection
order/payment/refund detection
adversarial test cases
failed-troubleshooting detection
Day 5 — Evaluation

Build:

larger golden set
multi-rater evaluation
inter-rater agreement
regression suite
adversarial test suite

Track separate metrics for:

intent
retrieval
response quality
safety
escalation
Day 6 — Support integration

Integrate with a support inbox.

Add:

conversation history
ticket metadata
agent approval
feedback buttons
correction workflow
response editing
Day 7 — Deployment and monitoring

Add:

authentication
logging
latency monitoring
token/cost monitoring
escalation rate monitoring
unsafe-response monitoring
retrieval failure monitoring
model regression monitoring

Deploy behind appropriate access controls.

45. Non-Obvious Design Decisions
Decision 1 — Select AmazonHelp

AmazonHelp was selected because it provides high interaction volume and diverse support scenarios.

Decision 2 — Message-level intent

Intent is assigned to the current customer message rather than the entire conversation because customer issues can evolve over multiple turns.

Decision 3 — 14-intent taxonomy

The taxonomy intentionally balances coverage with usability.

Too few categories collapse distinct operational problems.

Too many categories make classification and evaluation unnecessarily fragmented.

Decision 4 — ORDER_CHANGE

Cancellation, modification, editing, and address changes are grouped into ORDER_CHANGE because they share a common operational requirement: changing an existing order.

Decision 5 — DIGITAL_CONTENT vs TECHNICAL

Informational digital-content questions are separated from actual service malfunction.

Decision 6 — Historical responses are evidence

Historical support responses are not treated as authoritative policy because policies and workflows can change.

Decision 7 — FAISS IndexFlatIP

Normalized embeddings allow inner product to represent cosine similarity.

The support corpus is small enough for an exact flat index.

Decision 8 — Top-5 retrieval

Five retrieved cases provide multiple examples without unnecessarily increasing the generation context.

Decision 9 — Leakage-safe evaluation

Entire conversation components containing golden examples are removed rather than removing only individual tweets.

This provides stronger leakage protection.

Decision 10 — Specific operational issues escalate

The model cannot access Amazon's operational systems.

Therefore it should not pretend to inspect an order, shipment, payment, refund, or account.

Decision 11 — Safety over raw escalation accuracy

Unsafe auto-handling is treated as a high-priority failure because it can be more harmful than unnecessary escalation.

Decision 12 — LLM judge + human review

LLM-as-judge provides scalable evaluation, while human review provides an independent quality signal.

Decision 13 — Persist embeddings

Embedding construction is expensive compared with evaluation.

Persisting the FAISS index makes evaluation and demo startup practical.

Decision 14 — Streamlit demo

Streamlit provides a lightweight way to demonstrate the complete workflow without building a separate frontend/backend system.

46. Final Results Summary
Evaluation set:
200 examples

Intent classification:
66.5% accuracy
60.9% Macro F1

Semantic baseline:
27.0% accuracy
16.6% Macro F1

Rule baseline:
26.5% accuracy
15.8% Macro F1

Retrieval:
0.7847 mean Top-1 similarity
0.7642 mean Top-5 similarity

Response quality:
4.553 / 5 overall

Response helpfulness:
4.070 / 5

Response safety:
4.940 / 5

Escalation:
85.0% accuracy
89.4% escalation recall
10.6% unsafe auto-handle rate

Reproducibility:
~1.19 seconds
47. Final Takeaway

This project demonstrates a complete RAG-based customer-support workflow:

Customer message
        |
        v
Intent classification
        |
        v
Semantic retrieval
        |
        v
Historical support evidence
        |
        v
Grounded response generation
        |
        v
Safety-aware escalation
        |
        v
Human-reviewable support decision

The prototype shows meaningful improvement over simple baselines while explicitly measuring response quality and escalation safety.

The most important remaining work for production would be:

larger and more representative evaluation data
current-policy grounding
stronger retrieval evaluation
stronger safety validation
lower unsafe auto-handle rate
human-in-the-loop support integration
production monitoring

### One correction before you paste it

Don't blindly include the `requirements.txt` claim unless you actually have that file. We should create it next so the README's installation instructions are **fully reproducible**.

After saving the README, run:

```powershell
Test-Path requirements.txt