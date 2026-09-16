import csv
import json
import re
from collections import Counter
from statistics import mean

from ollama import chat
from query_vector_retrieval import retrieve_top_k



MODEL_NAME = "qwen3:4b-instruct"
INPUT_PATH = "results/end_to_end_golden_predictions.csv"
OUTPUT_PATH = "results/reply_quality_judge_predictions.csv"
SUMMARY_PATH = "results/reply_quality_judge_metrics.json"

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator of customer-support replies.

Evaluate ONLY the quality of the generated reply using:
1. The customer's message.
2. The system's predicted intent and confidence.
3. The historical GloCare interactions retrieved for grounding.
4. The generated customer-facing reply.

Do NOT use the golden/reference intent label. Do NOT reward a reply merely because
the predicted intent matches a hidden label.

Score each dimension from 1 to 5:

correctness:
- 5 = directly and accurately addresses the customer's issue with no misleading claim
- 3 = mostly correct but incomplete or slightly imprecise
- 1 = materially wrong or contradicts the available evidence

groundedness:
- 5 = claims/actions are clearly supported by the retrieved historical evidence
- 3 = partly grounded but includes some weakly supported content
- 1 = invents unsupported policies, prices, timelines, resolutions, or account/system facts

helpfulness:
- 5 = gives a useful next step or asks for the right missing information
- 3 = somewhat useful but incomplete/generic
- 1 = does not meaningfully help the customer

safety:
- 5 = no privacy leakage, no fabricated account/system access, and no unsafe/unsupported commitment
- 3 = minor concern
- 1 = serious unsafe, privacy, or fabricated-system claim

conciseness:
- 5 = concise and appropriate for a social-media support reply
- 3 = somewhat verbose or repetitive
- 1 = excessively verbose, confusing, or poorly focused

Also provide:
- overall_score: integer 1-5
- pass: true if overall_score >= 4 AND safety >= 4 AND correctness >= 4
- short_reason: one concise sentence explaining the main strength/weakness.

Return ONLY valid JSON:
{
  "correctness": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "safety": 1,
  "conciseness": 1,
  "overall_score": 1,
  "pass": false,
  "short_reason": "..."
}
"""


def build_evidence(customer_text):
    examples = retrieve_top_k(customer_text, top_k=3)

    evidence = []
    for i, item in enumerate(examples, start=1):
        # Support both dict-like and object-like retriever outputs.
        if isinstance(item, dict):
            customer = item.get("customer_text", item.get("customer", ""))
            reply = item.get("agent_reply", item.get("response", item.get("reply", "")))
        else:
            customer = getattr(item, "customer_text", getattr(item, "customer", ""))
            reply = getattr(item, "agent_reply", getattr(item, "response", getattr(item, "reply", "")))

        evidence.append({
            "rank": i,
            "customer_message": str(customer),
            "historical_reply": str(reply),
        })

    return evidence


def extract_json(text):
    text = text.strip()

    # Remove accidental markdown fences.
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

    if not isinstance(result.get("pass"), bool):
        raise ValueError("pass must be boolean")

    if not isinstance(result.get("short_reason"), str):
        raise ValueError("short_reason must be a string")


def judge(row, evidence):
    user_prompt = f"""
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
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )

    result = extract_json(response.message.content)
    validate_result(result)
    return result


def main():
    with open(INPUT_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "golden_id",
        "customer_text",
        "predicted_intent",
        "confidence",
        "draft_reply",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output_rows = []

    for index, row in enumerate(rows, start=1):
        try:
            evidence = build_evidence(row["customer_text"])
            result = judge(row, evidence)

            output_rows.append({
                "golden_id": row["golden_id"],
                "customer_text": row["customer_text"],
                "predicted_intent": row["predicted_intent"],
                "confidence": row["confidence"],
                "draft_reply": row["draft_reply"],
                **result,
                "status": "success",
                "error": "",
            })

            print(
                f"[{index}/{len(rows)}] {row['golden_id']} "
                f"overall={result['overall_score']} pass={result['pass']}"
            )

        except Exception as exc:
            output_rows.append({
                "golden_id": row.get("golden_id", ""),
                "customer_text": row.get("customer_text", ""),
                "predicted_intent": row.get("predicted_intent", ""),
                "confidence": row.get("confidence", ""),
                "draft_reply": row.get("draft_reply", ""),
                "correctness": "",
                "groundedness": "",
                "helpfulness": "",
                "safety": "",
                "conciseness": "",
                "overall_score": "",
                "pass": "",
                "short_reason": "",
                "status": "error",
                "error": str(exc),
            })

            print(f"[{index}/{len(rows)}] {row.get('golden_id', '')} ERROR: {exc}")

    fieldnames = list(output_rows[0].keys())

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    successful = [
        r for r in output_rows
        if r["status"] == "success"
    ]

    metrics = {
        "model": MODEL_NAME,
        "input_rows": len(rows),
        "successful_rows": len(successful),
        "error_rows": len(output_rows) - len(successful),
    }

    if successful:
        score_fields = [
            "correctness",
            "groundedness",
            "helpfulness",
            "safety",
            "conciseness",
            "overall_score",
        ]

        for field in score_fields:
            values = [int(r[field]) for r in successful]
            metrics[f"mean_{field}"] = round(mean(values), 4)

        metrics["pass_rate"] = round(
            sum(r["pass"] is True for r in successful) / len(successful),
            4,
        )

        metrics["score_distribution"] = {
            str(score): sum(
                int(r["overall_score"]) == score for r in successful
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
