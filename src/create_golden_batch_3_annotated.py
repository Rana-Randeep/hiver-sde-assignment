from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "golden_batch_3_review.txt"
OUTPUT_FILE = RESULTS_DIR / "golden_batch_3_annotated.csv"


# Human-reviewed annotations from our Batch 3 review.
# Key = EXAMPLE number.
ANNOTATIONS = {
    1:  ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
         "Customer denies requesting Gamestore service; account/service billing issue."),
    2:  ("yes", "unauthorized_charge_or_vas", "high", "escalation_relevant", "yes",
         "Customer explicitly asks to resolve unexplained deductions."),
    3:  ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Complaint refers to prior support but underlying issue is not stated."),
    4:  ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "4G/data service stopped after subscription."),
    5:  ("yes", "unauthorized_charge_or_vas", "high", "escalation_relevant", "yes",
         "Unsolicited deductions/service issue requires account-specific investigation."),
    6:  ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
         "Customer wants caller-tune/VAS messages stopped."),
    7:  ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Customer refers to a previous issue without stating the underlying problem."),
    8:  ("no", "non_actionable_or_context_required", "high", "low_information", "no",
         "DM-only/context-dependent message with no actionable issue."),
    9:  ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Underlying support problem is not sufficiently specified."),
    10: ("yes", "account_or_general_support", "high", "routine", "no",
         "Asks for the meaning of DND."),
    11: ("yes", "data_plan_issue", "high", "routine", "no",
         "Requests deactivation of a specific data plan."),
    12: ("no", "non_actionable_or_context_required", "high", "low_information", "no",
         "Generic dissatisfaction without an actionable support goal."),
    13: ("yes", "data_plan_issue", "high", "routine", "no",
         "Requests removal from data gifting/sharing."),
    14: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Unspecified prior issue; message lacks actionable support goal."),
    15: ("yes", "sim_or_device_issue", "high", "routine", "no",
         "Asks where to retrieve a lost SIM."),
    16: ("yes", "recharge_or_airtime_issue", "high", "escalation_relevant", "yes",
         "Recharge was made but account was not credited."),
    17: ("yes", "sim_or_device_issue", "high", "routine", "no",
         "Asks requirements for retrieving a lost SIM."),
    18: ("yes", "recharge_or_airtime_issue", "medium", "ambiguous", "yes",
         "Short recharge code; historical response indicates recharge-card handling."),
    19: ("yes", "sim_or_device_issue", "high", "routine", "no",
         "SIM replacement requires affidavit."),
    20: ("yes", "sim_or_device_issue", "high", "routine", "no",
         "Missing SIM pack and asks about replacement requirements."),
    21: ("yes", "network_issue", "high", "routine", "yes",
         "Emergency-only service on 3G indicates network/connectivity issue."),
    22: ("yes", "voice_or_line_issue", "high", "escalation_relevant", "yes",
         "Customer reports unusually high voice-call charges."),
    23: ("yes", "data_plan_issue", "high", "routine", "no",
         "Requests deactivation of data auto-renewal."),
    24: ("yes", "voice_or_line_issue", "high", "routine", "no",
         "Calls cannot be completed; historical response attributes it to insufficient credit."),
    25: ("yes", "sim_or_device_issue", "high", "escalation_relevant", "yes",
         "SIM/line became blocked; historical response explicitly requires offline handling."),
    26: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Refers to an unresolved prior issue but does not clearly identify the service problem."),
    27: ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
         "Wants recurring promotional/service messages stopped."),
    28: ("yes", "recharge_or_airtime_issue", "high", "escalation_relevant", "yes",
         "Explicit difficulty recharging the line."),
    29: ("yes", "voice_or_line_issue", "high", "routine", "no",
         "Insufficient credit prevents calling."),
    30: ("yes", "voice_or_line_issue", "high", "routine", "yes",
         "Cannot receive SMS on the line."),
    31: ("yes", "recharge_or_airtime_issue", "high", "routine", "no",
         "Requests deactivation of Auto Borrow service."),
    32: ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
         "Requests unsubscribe from entertainment service."),
    33: ("yes", "network_issue", "high", "routine", "yes",
         "Reports poor network."),
    34: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Customer-service complaint without underlying support issue."),
    35: ("yes", "unauthorized_charge_or_vas", "high", "escalation_relevant", "yes",
         "Reports unexplained deductions/subscriptions."),
    36: ("yes", "sim_or_device_issue", "medium", "escalation_relevant", "yes",
         "New SIM has multiple service problems; SIM setup/account-specific handling."),
    37: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "Intermittent data service."),
    38: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "No browsing despite full signal bars."),
    39: ("yes", "network_issue", "high", "escalation_relevant", "yes",
         "Repeated signal/internet loss."),
    40: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "Cannot browse."),
    41: ("yes", "data_service_issue", "medium", "ambiguous", "yes",
         "Data exists but internet is charging the main account; needs account/service investigation."),
    42: ("yes", "data_service_issue", "high", "routine", "yes",
         "Persistent inability to browse."),
    43: ("no", "non_actionable_or_context_required", "medium", "low_information", "yes",
         "Provides diagnostic details but does not state the actual issue."),
    44: ("no", "non_actionable_or_context_required", "high", "low_information", "no",
         "Explicitly states there is no issue."),
    45: ("yes", "data_plan_issue", "high", "routine", "no",
         "Asks for the cheapest available data plan."),
    46: ("no", "non_actionable_or_context_required", "medium", "low_information", "yes",
         "Only diagnostic/device details; underlying issue is not stated."),
    47: ("yes", "network_issue", "high", "escalation_relevant", "yes",
         "Nine-day network outage affecting calls and browsing."),
    48: ("yes", "data_plan_issue", "medium", "ambiguous", "yes",
         "Questions money deduction associated with a data subscription."),
    49: ("yes", "network_issue", "high", "routine", "yes",
         "Location-specific browsing/network performance problem."),
    50: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Requests investigation but explicitly says data services are fine; actual issue is unclear."),
    51: ("yes", "data_plan_issue", "high", "routine", "no",
         "Subscription/data-plan access problem."),
    52: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "No browsing despite having a data subscription."),
    53: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "Has data but cannot use it."),
    54: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
         "Has data but cannot browse."),
    55: ("yes", "data_plan_issue", "high", "routine", "no",
         "Asks about data amount at renewal."),
    56: ("yes", "network_issue", "medium", "ambiguous", "yes",
         "Network problem appears to cause the perceived bundle/data issue."),
    57: ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
         "Requests stopping entertainment subscriptions."),
    58: ("yes", "data_plan_issue", "high", "routine", "no",
         "Asks about international calling combo balance."),
    59: ("yes", "data_plan_issue", "high", "routine", "no",
         "Requests/asks about a social data bundle/plan."),
    60: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
         "Vague complaint without a sufficiently clear underlying issue."),
}


def parse_review_file(path):
    text = path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"EXAMPLE\s+(\d+)\s*"
        r"-+\s*"
        r"Customer tweet ID:\s*(\d+)\s*"
        r"Customer message:\s*(.*?)\s*"
        r"Historical brand response:\s*(.*?)\s*"
        r"Sampling information:",
        re.DOTALL
    )

    rows = []

    for match in pattern.finditer(text):
        example_no = int(match.group(1))
        customer_tweet_id = match.group(2)
        customer_text = match.group(3).strip()
        brand_response = match.group(4).strip()

        rows.append({
            "review_example": example_no,
            "customer_tweet_id": customer_tweet_id,
            "customer_text": customer_text,
            "brand_response": brand_response,
        })

    return rows


def main():
    print("=== CREATE ANNOTATED GOLDEN BATCH 3 ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    rows = parse_review_file(INPUT_FILE)

    print(f"Examples parsed from review file: {len(rows)}")

    if len(rows) != 60:
        raise ValueError(
            f"Expected exactly 60 examples, found {len(rows)}. "
            "Do not continue until the review file format is checked."
        )

    output_rows = []

    for row in rows:
        example_no = row["review_example"]

        if example_no not in ANNOTATIONS:
            raise ValueError(
                f"Missing annotation for EXAMPLE {example_no}"
            )

        (
            selected,
            intent,
            confidence,
            case_type,
            escalation_expected,
            notes,
        ) = ANNOTATIONS[example_no]

        output_rows.append({
            **row,
            "golden_selected": selected,
            "golden_intent": intent,
            "confidence": confidence,
            "case_type": case_type,
            "escalation_expected": escalation_expected,
            "annotation_notes": notes,
        })

    df = pd.DataFrame(output_rows)

    selected = df["golden_selected"].eq("yes").sum()
    excluded = df["golden_selected"].eq("no").sum()

    print(f"Selected: {selected}")
    print(f"Excluded: {excluded}")

    if selected != 47 or excluded != 13:
        raise ValueError(
            f"Expected 47 selected and 13 excluded, "
            f"got {selected} selected and {excluded} excluded."
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nCreated: {OUTPUT_FILE}")

    print("\n=== SELECTED INTENT DISTRIBUTION ===")
    print(
        df.loc[df["golden_selected"] == "yes", "golden_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== CASE TYPE DISTRIBUTION ===")
    print(
        df.loc[df["golden_selected"] == "yes", "case_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== ESCALATION DISTRIBUTION ===")
    print(
        df.loc[df["golden_selected"] == "yes", "escalation_expected"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()