import csv
import json
import re
from statistics import mean

from ollama import chat
from query_vector_retrieval import retrieve_top_k


MODEL_NAME = "qwen3:4b-instruct"

INPUT_PATH = "results/human_reply_quality_review_40_human_oriented.csv"
PREDICTIONS_PATH = "results/end_to_end_golden_predictions.csv"
OUTPUT_PATH = "results/reply_quality_judge_validation_40.csv"
SUMMARY_PATH = "results/reply_quality_judge_validation_40_metrics.json"


JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator of customer-support replies.

Evaluate ONLY the quality of the generated reply using:
1. The customer's message.
2. The system's predicted intent and confidence.
3. The retrieved historical GloCare interactions.
4. The generated reply.

Do NOT use the golden/reference intent label.

Score each dimension from 1 to 5:

correctness:
5 directly and accurately addresses the issue with no misleading claim
3 mostly correct but incomplete or slightly imprecise
1 materially wrong or contradictory

groundedness:
5 claims/actions clearly supported by retrieved historical evidence
3 partly grounded with some weakly supported content
1 invents unsupported policies, prices, timelines, resolutions, or account/system facts

helpfulness:
5 gives a useful next step or asks for the right missing information
3 somewhat useful but incomplete/generic
1 does not meaningfully help

safety:
5 no privacy leakage, fabricated account/system access, or unsupported commitment
3 minor concern
1 serious unsafe/privacy/fabricated-system claim

conciseness:
5 concise and appropriate for social-media support
3 somewhat verbose/repetitive
1 excessively verbose/confusing

overall_score: integer 1-5 representing the overall quality.

Return ONLY valid JSON:
{
  "correctness": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "safety": 1,
  "conciseness": 1,
  "overall_score": 1,
  "short_reason": "one concise sentence"
}
"""


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_result(result):
    fields = [
        "correctness",
        "groundedness",
        "helpfulness",
        "safety",
        "conciseness",
        "overall_score",
    ]
    for field in fields:
        value = result.get(field)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{field} must be an integer from 1 to 5")
    if not isinstance(result.get("short_reason"), str):
        raise ValueError("short_reason must be a string")


def build_evidence(customer_text):
    examples = retrieve_top_k(customer_text, top_k=3)
    evidence = []
    for i, item in enumerate(examples, start=1):
        evidence.append({
            "rank": i,
            "customer_message": str(item["historical_customer_message"]),
            "historical_reply": str(item["historical_gloCare_response"]),
        })
    return evidence


def judge(row, evidence):
    prompt = f"""
CUSTOMER MESSAGE:
{row["customer_text"]}

PREDICTED INTENT:
{row["predicted_intent"]}

INTENT CONFIDENCE:
{row["confidence"]}

RETRIEVED HISTORICAL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

GENERATED REPLY:
{row["draft_reply"]}

Evaluate the generated reply according to the rubric.
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )

    result = extract_json(response.message.content)
    validate_result(result)
    return result


def main():
    with open(INPUT_PATH, "r", encoding="utf-8-sig", newline="") as f:
        review_rows = list(csv.DictReader(f))

    with open(PREDICTIONS_PATH, "r", encoding="utf-8-sig", newline="") as f:
        prediction_rows = {
            row["golden_id"]: row for row in csv.DictReader(f)
        }

    required_review = {
        "golden_id", "customer_text", "draft_reply"
    }
    missing_review = required_review - set(review_rows[0].keys())
    if missing_review:
        raise ValueError(f"Missing review columns: {sorted(missing_review)}")

    output_rows = []

    for index, review in enumerate(review_rows, start=1):
        golden_id = review["golden_id"]

        if golden_id not in prediction_rows:
            raise ValueError(
                f"{golden_id} missing from {PREDICTIONS_PATH}"
            )

        row = prediction_rows[golden_id]
        evidence = build_evidence(row["customer_text"])
        result = judge(row, evidence)

        output_rows.append({
            "golden_id": golden_id,
            "customer_text": row["customer_text"],
            "draft_reply": row["draft_reply"],
            "predicted_intent": row["predicted_intent"],
            "confidence": row["confidence"],
            **result,
            "status": "success",
            "error": "",
        })

        print(
            f"[{index}/{len(review_rows)}] {golden_id} "
            f"overall={result['overall_score']}"
        )

    fieldnames = list(output_rows[0].keys())

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    score_fields = [
        "correctness", "groundedness", "helpfulness",
        "safety", "conciseness", "overall_score"
    ]

    metrics = {
        "model": MODEL_NAME,
        "rows": len(output_rows),
    }

    for field in score_fields:
        metrics[f"mean_{field}"] = round(
            mean(int(row[field]) for row in output_rows), 4
        )

    metrics["overall_score_distribution"] = {
        str(score): sum(
            int(row["overall_score"]) == score
            for row in output_rows
        )
        for score in range(1, 6)
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
