import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL = "gemini-3.5-flash-lite"
DEFAULT_BATCH_SIZE = 5
DEFAULT_SLEEP_SECONDS = 3.3


class ResponseItem(BaseModel):
    tweet_id: str
    reply: str
    grounding: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResponseBatch(BaseModel):
    results: list[ResponseItem]


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def load_inputs(root):
    golden_path = root / "data" / "golden" / "golden_set.csv"
    predictions_path = (
        root / "data" / "golden" / "gemini_classifier_predictions.csv"
    )
    retrieval_path = (
        root / "data" / "golden" / "retrieval_evaluation_results.csv"
    )

    print("Loading:")
    print(f"  Golden set: {golden_path}")
    print(f"  Intent predictions: {predictions_path}")
    print(f"  Retrieval results: {retrieval_path}")

    golden = pd.read_csv(golden_path)
    predictions = pd.read_csv(predictions_path)
    retrieval = pd.read_csv(retrieval_path)

    # ---------------------------------------------------------
    # Golden set columns
    # ---------------------------------------------------------

    golden_id = find_column(
        golden,
        ["tweet_id", "id"],
        "golden tweet ID",
    )

    text_column = find_column(
        golden,
        ["text", "customer_text", "query_text"],
        "golden customer text",
    )

    gold_intent_column = find_column(
        golden,
        ["gold_intent", "intent"],
        "gold intent",
    )

    golden = golden[
        [golden_id, text_column, gold_intent_column]
    ].copy()

    golden = golden.rename(
        columns={
            golden_id: "tweet_id",
            text_column: "customer_text",
            gold_intent_column: "gold_intent",
        }
    )

    golden["tweet_id"] = golden["tweet_id"].astype(str)

    # ---------------------------------------------------------
    # Gemini intent prediction columns
    # ---------------------------------------------------------

    prediction_id = find_column(
        predictions,
        ["tweet_id", "query_tweet_id", "id"],
        "prediction tweet ID",
    )

    predicted_intent_column = find_column(
        predictions,
        ["predicted_intent", "intent"],
        "predicted intent",
    )

    confidence_column = None

    for candidate in [
        "confidence",
        "intent_confidence",
        "predicted_confidence",
    ]:
        if candidate in predictions.columns:
            confidence_column = candidate
            break

    prediction_columns = [
        prediction_id,
        predicted_intent_column,
    ]

    if confidence_column:
        prediction_columns.append(confidence_column)

    predictions = predictions[prediction_columns].copy()

    rename_map = {
        prediction_id: "tweet_id",
        predicted_intent_column: "predicted_intent",
    }

    if confidence_column:
        rename_map[confidence_column] = "intent_confidence"

    predictions = predictions.rename(columns=rename_map)

    predictions["tweet_id"] = predictions["tweet_id"].astype(str)

    # ---------------------------------------------------------
    # Retrieval columns
    # ---------------------------------------------------------

    retrieval_id = find_column(
        retrieval,
        ["query_tweet_id", "tweet_id"],
        "retrieval query tweet ID",
    )

    rank_column = find_column(
        retrieval,
        ["rank"],
        "retrieval rank",
    )

    similarity_column = find_column(
        retrieval,
        ["similarity", "score"],
        "retrieval similarity",
    )

    retrieval = retrieval.rename(
        columns={
            retrieval_id: "tweet_id",
            rank_column: "rank",
            similarity_column: "similarity",
        }
    )

    retrieval["tweet_id"] = retrieval["tweet_id"].astype(str)
    retrieval["rank"] = pd.to_numeric(
        retrieval["rank"],
        errors="coerce",
    )
    retrieval["similarity"] = pd.to_numeric(
        retrieval["similarity"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Merge golden + predicted intent
    # ---------------------------------------------------------

    df = golden.merge(
        predictions,
        on="tweet_id",
        how="inner",
    )

    if len(df) != len(golden):
        print(
            f"WARNING: golden examples = {len(golden)}, "
            f"merged examples = {len(df)}"
        )

    return df, retrieval


def build_case_prompt(row, evidence):
    evidence_sections = []

    evidence = evidence.sort_values("rank").head(5)

    for _, item in evidence.iterrows():
        customer_example = str(
            item.get("retrieved_customer_text", "")
        )

        historical_reply = str(
            item.get("retrieved_agent_response", "")
        )

        evidence_sections.append(
            f"""
EVIDENCE {int(item["rank"])}

Historical customer:
{customer_example}

Historical AmazonHelp response:
{historical_reply}
""".strip()
        )

    evidence_text = "\n\n".join(evidence_sections)

    return f"""
You are the response-drafting component of an Amazon customer-support AI agent.

CUSTOMER MESSAGE:
{row["customer_text"]}

PREDICTED INTENT:
{row["predicted_intent"]}

RETRIEVED HISTORICAL SUPPORT CASES:
{evidence_text}

TASK:
Draft one concise customer-facing response to the customer's message.

IMPORTANT RULES:

1. Historical AmazonHelp replies are evidence only.
   They are NOT authoritative current Amazon policy.

2. Do not blindly copy historical replies.

3. Never invent:
   - order numbers
   - tracking numbers
   - refund status
   - payment status
   - account information
   - delivery dates
   - eligibility
   - current policies
   - actions already taken

4. Never claim that you accessed the customer's account.

5. Never claim that you changed an order, issued a refund,
   contacted a carrier, checked an account, or performed another
   action unless the customer message itself establishes that fact.

6. Do not copy:
   - URLs
   - ticket IDs
   - order IDs
   - tracking IDs
   - usernames
   - email addresses
   - phone numbers
   - other identifying information
   from historical examples.

7. If a historical response contains a URL or identifier,
   describe the underlying action generically instead.

8. If the issue requires account-specific investigation,
   order-specific investigation, shipment investigation,
   payment investigation, refund investigation, fraud investigation,
   or another consequential action, do not pretend the AI can perform it.

9. Give a useful and safe next step whenever possible.

10. Keep the response natural and concise.

11. Do not mention:
   - AI
   - Gemini
   - FAISS
   - embeddings
   - retrieval
   - this evaluation
   - historical examples

12. The grounding field should briefly explain which retrieved
   evidence influenced the response.

13. Confidence is confidence in the quality and safety of the draft.
   It is NOT a probability of correctness.

Return exactly one result for tweet_id:
{row["tweet_id"]}
""".strip()


def generate_batch(client, batch_rows, retrieval):
    case_prompts = []

    for index, row in enumerate(batch_rows, start=1):
        tweet_id = str(row["tweet_id"])

        evidence = retrieval[
            retrieval["tweet_id"] == tweet_id
        ].sort_values("rank").head(5)

        if len(evidence) == 0:
            print(
                f"WARNING: no retrieval evidence found for "
                f"tweet_id={tweet_id}"
            )

        case_prompts.append(
            f"""
========================
CASE {index}
========================

{build_case_prompt(row, evidence)}
""".strip()
        )

    combined_prompt = """
Process all cases independently.

Return exactly one structured response for every case.
Preserve every tweet_id exactly.

Do not omit cases.

""" + "\n\n".join(case_prompts)

    result = client.models.generate_content(
        model=MODEL,
        contents=combined_prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ResponseBatch,
        ),
    )

    if result.parsed is None:
        raise RuntimeError(
            "Gemini returned no structured parsed response."
        )

    return result.parsed.results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of golden examples per Gemini request",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds between Gemini requests",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    output_path = (
        root
        / "data"
        / "golden"
        / "end_to_end_responses.csv"
    )

    cache_path = (
        root
        / "data"
        / "golden"
        / "response_generation_cache.json"
    )

    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    df, retrieval = load_inputs(root)

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    if cache_path.exists():
        print(f"\nLoading cache: {cache_path}")

        with open(
            cache_path,
            "r",
            encoding="utf-8",
        ) as f:
            cache = json.load(f)
    else:
        cache = {}

    # ---------------------------------------------------------
    # Determine pending examples
    # ---------------------------------------------------------

    pending = []

    for _, row in df.iterrows():
        tweet_id = str(row["tweet_id"])

        if tweet_id not in cache:
            pending.append(row)

    print("\n========================================")
    print("RESPONSE GENERATION EVALUATION")
    print("========================================")
    print(f"Golden examples: {len(df)}")
    print(f"Cached responses: {len(cache)}")
    print(f"Pending responses: {len(pending)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {MODEL}")
    print("========================================\n")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    for start in range(
        0,
        len(pending),
        args.batch_size,
    ):
        batch = pending[
            start:start + args.batch_size
        ]

        print(
            f"Generating "
            f"{start + 1}-"
            f"{min(start + len(batch), len(pending))}"
            f"/{len(pending)}..."
        )

        results = generate_batch(
            client,
            batch,
            retrieval,
        )

        returned = {
            str(item.tweet_id): item
            for item in results
        }

        for row in batch:
            tweet_id = str(row["tweet_id"])

            if tweet_id not in returned:
                raise RuntimeError(
                    f"Gemini did not return result for "
                    f"tweet_id={tweet_id}"
                )

            item = returned[tweet_id]

            cache[tweet_id] = {
                "reply": item.reply,
                "grounding": item.grounding,
                "response_confidence": float(
                    item.confidence
                ),
            }

        with open(
            cache_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                cache,
                f,
                indent=2,
                ensure_ascii=False,
            )

        completed = min(
            start + len(batch),
            len(pending),
        )

        print(
            f"  Completed: {completed}/{len(pending)}"
        )

        if completed < len(pending):
            time.sleep(args.sleep)

    # ---------------------------------------------------------
    # Build final evaluation table
    # ---------------------------------------------------------

    output_rows = []

    for _, row in df.iterrows():
        tweet_id = str(row["tweet_id"])

        evidence = retrieval[
            retrieval["tweet_id"] == tweet_id
        ].sort_values("rank").head(5)

        if len(evidence) > 0:
            top1_similarity = float(
                evidence.iloc[0]["similarity"]
            )
        else:
            top1_similarity = None

        generated = cache[tweet_id]

        output_rows.append(
            {
                "tweet_id": tweet_id,
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": row["predicted_intent"],
                "intent_confidence": row.get(
                    "intent_confidence",
                    None,
                ),
                "top1_similarity": top1_similarity,
                "generated_reply": generated["reply"],
                "grounding": generated["grounding"],
                "response_confidence": generated[
                    "response_confidence"
                ],
            }
        )

    result_df = pd.DataFrame(output_rows)

    result_df.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Basic diagnostics
    # ---------------------------------------------------------

    print("\n========================================")
    print("RESPONSE GENERATION COMPLETE")
    print("========================================")

    print(f"Rows written: {len(result_df)}")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_path}")

    print("\nMissing replies:")

    missing_reply = (
        result_df["generated_reply"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"  {missing_reply}")

    print("\nResponse confidence:")

    print(
        result_df["response_confidence"]
        .describe()
        .to_string()
    )

    print("\nPredicted intent distribution:")

    print(
        result_df["predicted_intent"]
        .value_counts()
        .to_string()
    )

    print("\nFirst 5 generated responses:")

    for _, row in result_df.head(5).iterrows():
        print("\n----------------------------------------")
        print(f"Tweet ID: {row['tweet_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Intent: {row['predicted_intent']}")
        print(f"Reply: {row['generated_reply']}")
        print(
            f"Confidence: "
            f"{row['response_confidence']:.2f}"
        )


if __name__ == "__main__":
    main()