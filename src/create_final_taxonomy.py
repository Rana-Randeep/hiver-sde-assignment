from pathlib import Path
import pandas as pd


OUTPUT_FILE = Path("results/final_taxonomy.csv")


TAXONOMY = [
    {
        "intent": "network_issue",
        "definition": "Customer reports poor, unavailable, unstable, or otherwise problematic network connectivity or coverage.",
        "include": "No signal, poor network, network unavailable, coverage/connectivity complaints.",
        "exclude": "Problems specifically caused by a data package, SIM/device, or voice/line issue when that is the primary goal.",
    },
    {
        "intent": "data_service_issue",
        "definition": "Customer reports that mobile data or browsing is unavailable, unreliable, slow, unexpectedly consumed, or otherwise not functioning as expected.",
        "include": "Cannot browse, cannot use data, slow browsing, download problems, unexpected data consumption.",
        "exclude": "Requests primarily about purchasing, activating, cancelling, changing, or understanding a data package or subscription.",
    },
    {
        "intent": "data_plan_issue",
        "definition": "Customer seeks information, activation, cancellation, modification, or resolution concerning a data package, subscription, bundle, renewal, allocation, or data-plan benefit.",
        "include": "Package size, subscription, activation, cancellation, renewal, allocation, plan information, data benefit.",
        "exclude": "Pure data connectivity or browsing-performance problems.",
    },
    {
        "intent": "unauthorized_charge_or_vas",
        "definition": "Customer reports an unexpected, unauthorized, or unwanted charge or value-added service.",
        "include": "Unauthorized deductions, unwanted VAS, caller-tune charges, unexplained service subscriptions.",
        "exclude": "Normal recharge or ordinary data-plan transactions without an unauthorized-charge complaint.",
    },
    {
        "intent": "recharge_or_airtime_issue",
        "definition": "Customer needs help with recharge, airtime balance, or an airtime/recharge transaction.",
        "include": "Recharge failed, recharge missing, airtime problems, recharge transaction issues.",
        "exclude": "Data-package or unauthorized-charge problems when those are the primary goal.",
    },
    {
        "intent": "sim_or_device_issue",
        "definition": "Customer's support goal concerns SIM functionality/replacement/credentials or device compatibility.",
        "include": "SIM replacement, SIM swap, PUK/SIM information, handset or SIM compatibility.",
        "exclude": "Problems that are primarily about data plans, network service, or voice service.",
    },
    {
        "intent": "voice_or_line_issue",
        "definition": "Customer's support goal concerns the status or functioning of their mobile line or voice service.",
        "include": "Line inactive, line access problems, voice/calling service problems.",
        "exclude": "Unauthorized VAS charges or SIM/device issues when those are the primary goal.",
    },
    {
        "intent": "bonus_or_promotion_issue",
        "definition": "Customer needs help with a bonus, promotion, promotional eligibility, or promotional entitlement.",
        "include": "Missing bonus, promotion eligibility, promotional activation or entitlement.",
        "exclude": "Ordinary purchased data or airtime plans.",
    },
    {
        "intent": "account_or_general_support",
        "definition": "Customer has an actionable support request that does not fit an established service-specific intent and requires general or account/customer-specific handling.",
        "include": "Miscellaneous actionable requests such as account-specific assistance or service requests without a better-fitting intent.",
        "exclude": "Messages that clearly belong to another specific intent or contain no actionable support request.",
    },
    {
        "intent": "non_actionable_or_context_required",
        "definition": "The message alone does not provide enough information to determine an actionable customer-support goal.",
        "include": "Phone number only, SIM/PUK details only, vague follow-ups, acknowledgements without a clear request.",
        "exclude": "Messages with a sufficiently clear support goal.",
    },
]


def main():
    df = pd.DataFrame(TAXONOMY)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Final taxonomy created.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Number of intents: {len(df)}")
    print()

    print(df[["intent", "definition"]].to_string(index=False))


if __name__ == "__main__":
    main()