from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_batch_2.csv")
OUTPUT_FILE = Path("results/golden_batch_2_annotated.csv")


ANNOTATIONS = {
    1: ("yes", "data_plan_issue", "high", "escalation_relevant", "yes",
        "Subscription stopped before expected expiry; customer could not get support."),
    2: ("yes", "unauthorized_charge_or_vas", "high", "ambiguous", "yes",
        "Customer reports unexpected iFlix-related data charging and asks about the rule."),
    3: ("yes", "sim_or_device_issue", "high", "escalation_relevant", "yes",
        "New SIM was registered but activation did not occur as promised."),
    4: ("yes", "data_service_issue", "high", "routine", "yes",
        "Customer reports very poor data service."),
    5: ("yes", "network_issue", "high", "routine", "yes",
        "Customer reports severe data connectivity problem at a specific location."),
    6: ("yes", "network_issue", "high", "escalation_relevant", "yes",
        "Complete network outage affects both calls and data in a defined area."),
    7: ("yes", "bonus_or_promotion_issue", "high", "ambiguous", "yes",
        "Customer clarifies that the missing bonus is credit bonus rather than data bonus."),
    8: ("yes", "unauthorized_charge_or_vas", "high", "ambiguous", "yes",
        "Customer reports an unexpected deduction after recharge; historical response attributes it to loan repayment."),
    9: ("yes", "data_plan_issue", "high", "ambiguous", "yes",
        "Customer wants help activating a specific data plan after recharge."),
    10: ("yes", "unauthorized_charge_or_vas", "high", "escalation_relevant", "yes",
        "Customer reports unexplained recurring deductions and does not recall subscribing."),
    11: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer wants to check an international-call bundle balance."),
    12: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer cannot purchase an internet data package."),
    13: ("yes", "data_plan_issue", "medium", "ambiguous", "yes",
        "Customer reports receiving less data than expected after a subscription."),
    14: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer wants to switch from a weekend data plan to a normal data plan."),
    15: ("yes", "data_plan_issue", "high", "escalation_relevant", "yes",
        "Customer cannot renew a data bundle and reports repeated customer-care difficulty."),
    16: ("yes", "bonus_or_promotion_issue", "medium", "ambiguous", "yes",
        "Customer reports that their normal recharge bonus is no longer appearing."),
    17: ("yes", "recharge_or_airtime_issue", "high", "escalation_relevant", "yes",
        "Customer recharged but reports a much lower remaining balance than expected."),
    18: ("yes", "unauthorized_charge_or_vas", "high", "escalation_relevant", "yes",
        "Customer asks for unexplained deductions from airtime to be identified and removed."),
    19: ("yes", "recharge_or_airtime_issue", "high", "escalation_relevant", "yes",
        "Customer paid a BMC debt but was still notified that the debt remained."),
    20: ("yes", "voice_or_line_issue", "high", "escalation_relevant", "yes",
        "Customer cannot make calls or send messages and reports call-drop errors."),
    21: ("yes", "voice_or_line_issue", "high", "routine", "no",
        "Customer asks how to migrate to a different calling tariff."),
    22: ("yes", "bonus_or_promotion_issue", "medium", "ambiguous", "yes",
        "Customer asks about recharge requirements for unlocking additional bonus."),
    23: ("yes", "voice_or_line_issue", "high", "escalation_relevant", "yes",
        "Customer's calls cannot be completed despite trying customer care."),
    24: ("yes", "sim_or_device_issue", "high", "routine", "no",
        "Customer asks how to reactivate a deactivated line/SIM."),
    25: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
        "The message refers to an unspecified prior issue and does not identify the support goal."),
    26: ("no", "non_actionable_or_context_required", "high", "low_information", "yes",
        "The customer complains about prior assistance without stating the underlying issue."),
    27: ("yes", "unauthorized_charge_or_vas", "high", "routine", "no",
        "Customer reports receiving unwanted service messages despite activating DND."),
    28: ("yes", "account_or_general_support", "high", "escalation_relevant", "yes",
        "Customer reports unexpected deactivation from a package and requests an account check."),
    29: ("yes", "bonus_or_promotion_issue", "high", "routine", "no",
        "Customer reports missing bonus after purchasing a new Jumbo SIM and recharging."),
    30: ("no", "non_actionable_or_context_required", "high", "low_information", "no",
        "Message only confirms that browsing started working and thanks support."),
    31: ("yes", "voice_or_line_issue", "high", "routine", "no",
        "Customer complains about high per-minute calling charges and wants a better tariff."),
    32: ("yes", "recharge_or_airtime_issue", "high", "routine", "no",
        "Customer receives an invalid MMI error while attempting a recharge."),
    33: ("yes", "data_service_issue", "medium", "escalation_relevant", "yes",
        "Customer provides troubleshooting details for a data-service problem."),
    34: ("yes", "sim_or_device_issue", "high", "routine", "no",
        "Customer has difficulty retrieving/replacing a SIM because of identification requirements."),
    35: ("yes", "network_issue", "high", "routine", "yes",
        "Customer reports slow and unstable network quality in a specific location."),
    36: ("yes", "network_issue", "high", "escalation_relevant", "yes",
        "Customer reports a months-long local network outage affecting calls and data."),
    37: ("yes", "network_issue", "high", "routine", "no",
        "Customer asks about availability of a 4G site in a specific area."),
    38: ("yes", "data_service_issue", "medium", "escalation_relevant", "yes",
        "Customer reports data access problems after subscribing and expresses dissatisfaction."),
    39: ("yes", "data_service_issue", "high", "routine", "no",
        "Customer asks whether receiving shared data implies reciprocal sharing."),
    40: ("yes", "data_service_issue", "high", "escalation_relevant", "yes",
        "Customer reports unexpectedly rapid data consumption."),
    41: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer asks whether a specific streaming-data benefit is still available."),
    42: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer asks whether unused data carries over after a plan extension."),
    43: ("yes", "data_plan_issue", "high", "routine", "no",
        "Customer received less data than expected under a specific plan/renewal rule."),
    44: ("yes", "data_plan_issue", "medium", "ambiguous", "yes",
        "Customer asks whether a migration code would affect an existing data package."),
    45: ("yes", "network_issue", "high", "routine", "yes",
        "Customer reports complete loss of network service."),
    46: ("yes", "network_issue", "high", "escalation_relevant", "yes",
        "Customer reports emergency-only network status lasting several days."),
    47: ("yes", "data_plan_issue", "medium", "ambiguous", "yes",
        "Customer cannot subscribe to a data package and reports an ongoing network-related problem."),
    48: ("yes", "network_issue", "high", "escalation_relevant", "yes",
        "Customer reports severe network outage and asks whether service will remain unavailable."),
    49: ("yes", "network_issue", "high", "routine", "yes",
        "Customer reports very poor network quality and needs location-specific assistance."),
    50: ("yes", "bonus_or_promotion_issue", "high", "routine", "no",
        "Customer cannot access bonuses after recharge and asks why.")
}


def main():
    df = pd.read_csv(INPUT_FILE)

    required_orders = set(range(1, 51))
    actual_orders = set(df["review_order"].astype(int))

    if actual_orders != required_orders:
        raise ValueError(
            "Expected review_order values 1-50, "
            f"but found: {sorted(actual_orders)}"
        )

    annotation_columns = [
        "golden_selected",
        "golden_intent",
        "confidence",
        "case_type",
        "escalation_expected",
        "annotation_notes",
    ]

    for column in annotation_columns:
        df[column] = None

    for order, annotation in ANNOTATIONS.items():
        mask = df["review_order"].astype(int) == order

        (
            df.loc[mask, "golden_selected"],
            df.loc[mask, "golden_intent"],
            df.loc[mask, "confidence"],
            df.loc[mask, "case_type"],
            df.loc[mask, "escalation_expected"],
            df.loc[mask, "annotation_notes"],
        ) = annotation

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    selected_count = (df["golden_selected"] == "yes").sum()
    excluded_count = (df["golden_selected"] == "no").sum()

    print("Golden Batch 2 annotations saved.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total examples: {len(df)}")
    print(f"Selected: {selected_count}")
    print(f"Excluded: {excluded_count}")

    print("\nSelected intent distribution:")
    print(
        df[df["golden_selected"] == "yes"]["golden_intent"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()