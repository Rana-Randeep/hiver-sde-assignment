import argparse
import os

import pandas as pd

from support_agent import run_support_agent


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = "results/golden_set.csv"
DEFAULT_OUTPUT = "results/end_to_end_golden_predictions.csv"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_golden_set(input_path, output_path, limit=None):
    """
    Run the existing support agent on the golden set.

    Important:
    This function does not modify the support agent.
    It only records its actual outputs for evaluation.
    """

    df = pd.read_csv(input_path)

    required_columns = [
        "golden_id",
        "customer_text",
        "golden_intent",
        "escalation_expected",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if limit is not None:
        df = df.head(limit).copy()

    results = []

    total = len(df)

    print("=" * 80)
    print("END-TO-END GOLDEN SET EVALUATION")
    print("=" * 80)
    print(f"Input file : {input_path}")
    print(f"Examples   : {total}")
    print()

    for index, row in df.iterrows():

        golden_id = row["golden_id"]
        customer_text = row["customer_text"]

        print(
            f"[{len(results) + 1}/{total}] "
            f"Running golden example: {golden_id}"
        )

        try:
            agent_result = run_support_agent(
                customer_text
            )

            results.append(
                {
                    "golden_id": golden_id,
                    "customer_text": customer_text,
                    "golden_intent": row["golden_intent"],
                    "predicted_intent": agent_result["intent"],
                    "confidence": agent_result["confidence"],
                    "golden_escalation_expected": (
                        row["escalation_expected"]
                    ),
                    "actual_decision": (
                        agent_result["decision"]
                    ),
                    "escalation_reason": (
                        agent_result["reason"]
                    ),
                    "draft_reply": (
                        agent_result["draft_reply"]
                    ),
                    "status": "success",
                    "error": "",
                }
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            results.append(
                {
                    "golden_id": golden_id,
                    "customer_text": customer_text,
                    "golden_intent": row["golden_intent"],
                    "predicted_intent": "",
                    "confidence": "",
                    "golden_escalation_expected": (
                        row["escalation_expected"]
                    ),
                    "actual_decision": "",
                    "escalation_reason": "",
                    "draft_reply": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(results)

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Rows written: {len(results_df)}")
    print(f"Output file : {output_path}")

    print()
    print("Status counts:")
    print(
        results_df["status"]
        .value_counts()
        .to_string()
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the existing GloCare support agent "
            "against the golden set."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the golden-set CSV.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for the evaluation output CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of golden examples to run. "
            "Use this for a small smoke test before "
            "running the complete set."
        ),
    )

    args = parser.parse_args()

    evaluate_golden_set(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )