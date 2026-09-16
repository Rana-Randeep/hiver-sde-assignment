# ============================================================
# ESCALATION POLICY
# ============================================================

# This is an initial development threshold only.
# It is NOT the final production threshold.
#
# Later, we will tune this using validation data.
CONFIDENCE_THRESHOLD = 0.50


# Intents that are potentially account-specific or sensitive.
#
# These should not be automatically handled purely from
# historical public conversations.
SENSITIVE_INTENTS = {
    "unauthorized_charge_or_vas",
    "account_or_general_support",
}


def decide_escalation(
    intent,
    confidence,
    retrieved_examples,
):
    """
    Decide whether a customer-support request should be
    auto-handled or escalated to a human.

    Args:
        intent: str
            Predicted intent.

        confidence: float
            Classifier confidence.

        retrieved_examples: list[dict]
            Retrieved historical conversations.

    Returns:
        dict:
            {
                "decision": "auto_handle" | "human",
                "reason": str
            }
    """

    # --------------------------------------------------------
    # Rule 1: Low classifier confidence
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "human",
            "reason": (
                f"Low intent confidence ({confidence:.2f}) "
                f"is below the threshold of "
                f"{CONFIDENCE_THRESHOLD:.2f}."
            ),
        }

    # --------------------------------------------------------
    # Rule 2: Sensitive/account-specific intent
    # --------------------------------------------------------

    if intent in SENSITIVE_INTENTS:
        return {
            "decision": "human",
            "reason": (
                f"The '{intent}' intent may require "
                "account-specific investigation that is "
                "not available to this system."
            ),
        }

    # --------------------------------------------------------
    # Rule 3: No historical evidence
    # --------------------------------------------------------

    if not retrieved_examples:
        return {
            "decision": "human",
            "reason": (
                "No relevant historical GloCare evidence "
                "was retrieved."
            ),
        }

    # --------------------------------------------------------
    # Otherwise: allow automatic handling
    # --------------------------------------------------------

    return {
        "decision": "auto_handle",
        "reason": (
            "The request has sufficient classifier confidence "
            "and relevant historical GloCare evidence, and it "
            "does not fall into a sensitive intent category."
        ),
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_examples = [
        {
            "rank": 1,
            "historical_customer_message": "internet not working",
            "historical_gloCare_response": "Please provide your number and location.",
        }
    ]

    tests = [
        {
            "name": "Low confidence network issue",
            "intent": "network_issue",
            "confidence": 0.1205,
        },
        {
            "name": "High confidence network issue",
            "intent": "network_issue",
            "confidence": 0.90,
        },
        {
            "name": "High confidence sensitive issue",
            "intent": "unauthorized_charge_or_vas",
            "confidence": 0.95,
        },
        {
            "name": "No evidence",
            "intent": "network_issue",
            "confidence": 0.90,
            "examples": [],
        },
    ]

    for test in tests:

        examples = test.get(
            "examples",
            test_examples,
        )

        result = decide_escalation(
            intent=test["intent"],
            confidence=test["confidence"],
            retrieved_examples=examples,
        )

        print()
        print("=" * 70)
        print(test["name"])
        print("=" * 70)
        print("Intent:", test["intent"])
        print("Confidence:", test["confidence"])
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])