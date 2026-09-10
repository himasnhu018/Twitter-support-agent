# Final Evaluation Report

## Evaluation setup

- Brand: AmazonHelp
- Golden set: 200 hand-labeled customer messages
- Intent taxonomy: 14 labels including OTHER
- Retrieval: FAISS IndexFlatIP over normalized all-MiniLM-L6-v2 embeddings
- Response model: Gemini 3.5 Flash-Lite
- Evaluation uses a leakage-safe retrieval corpus with golden conversation components removed.

## Intent classification

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Gemini 3.5 Flash-Lite | 66.5% | 60.9% |
| Semantic baseline | 27.0% | 16.6% |
| Rule baseline | 26.5% | 15.8% |

## Retrieval

- Top-1 mean cosine similarity: 0.7847
- Top-5 mean cosine similarity: 0.7642

**Important:** these are retrieval similarity diagnostics, not retrieval accuracy. The corpus does not contain gold intent labels for the retrieved cases.

## Response quality

| Dimension | Mean / 5 | % >= 4 |
|---|---:|---:|
| Relevance | 4.425 | 88.5% |
| Groundedness | 4.855 | 98.0% |
| Helpfulness | 4.070 | 79.0% |
| Safety | 4.940 | 98.0% |
| Historical Grounding | 4.475 | 92.0% |
| **Overall** | **4.553** | - |

Safety scores <=2: 0/200

## Escalation

| Metric | Result |
|---|---:|
| Accuracy | 85.0% |
| Escalation recall | 89.4% |
| Unsafe auto-handle rate | 10.6% |
| Unsafe auto-handles | 14/200 |
| Unnecessary escalations | 16/200 |

## Failure analysis

- Intent errors: 67/200 (33.5%)
- Unsafe auto-handles: 14/200 (7.0%)
- Unnecessary escalations: 16/200 (8.0%)
- Low helpfulness (<=2): 10/200 (5.0%)
- Low relevance (<=2): 7/200 (3.5%)
- Low groundedness (<=2): 3/200 (1.5%)

The five concrete examples selected by the failure analysis should be documented separately with the customer message, generated reply, failure type, and proposed mitigation.

## Human vs LLM-judge agreement

Human agreement results were not found.

## What is misleading about my headline number?

The 66.5% intent accuracy should not be interpreted as production accuracy. It is measured on a 200-example golden set, the first version of the taxonomy, and an English-focused sample. The dataset itself is historical Twitter support data, not a live production stream.

The response-quality scores are also LLM-as-judge measurements rather than direct customer satisfaction. A 4.553/5 mean therefore does not mean 91% of real customers would be satisfied.

Retrieval cosine similarity is a relevance diagnostic, not retrieval accuracy. High similarity can still correspond to a historically similar but operationally incorrect case.

Finally, escalation accuracy hides an asymmetric risk: an unsafe auto-handle can be substantially worse than an unnecessary escalation. For that reason, escalation recall and unsafe auto-handle rate are reported separately.

## Reproducibility

The headline evaluation should reuse the persisted FAISS index and cached model outputs rather than rebuilding embeddings. The one-time CPU embedding construction is not included in the headline evaluation runtime.
