from intent_classifier import predict_intent
from query_vector_retrieval import retrieve_top_k
from llm_generator import generate_reply
from escalation_policy import decide_escalation
from reply_sanitizer import sanitize_reply


def run_support_agent(customer_message):
    """
    Run the complete GloCare customer-support pipeline.

    Pipeline:
        customer message
            ↓
        intent classification
            ↓
        historical retrieval
            ↓
        grounded LLM generation
            ↓
        escalation decision
    """

    # --------------------------------------------------------
    # 1. Intent classification
    # --------------------------------------------------------

    intent_result = predict_intent(customer_message)

    intent = intent_result["intent"]
    confidence = intent_result["confidence"]

    # --------------------------------------------------------
    # 2. Historical retrieval
    # --------------------------------------------------------

    retrieved_examples = retrieve_top_k(
        customer_message,
        top_k=3,
    )

    # --------------------------------------------------------
    # 3. Grounded response generation
    # --------------------------------------------------------

    draft_reply = generate_reply(
        customer_message=customer_message,
        intent=intent,
        confidence=confidence,
        retrieved_examples=retrieved_examples,
    )
    draft_reply = sanitize_reply(draft_reply)

    # --------------------------------------------------------
    # 4. Escalation decision
    # --------------------------------------------------------

    escalation_result = decide_escalation(
        intent=intent,
        confidence=confidence,
        retrieved_examples=retrieved_examples,
    )

    # --------------------------------------------------------
    # 5. Final structured result
    # --------------------------------------------------------

    return {
        "intent": intent,
        "confidence": confidence,
        "decision": escalation_result["decision"],
        "reason": escalation_result["reason"],
        "draft_reply": draft_reply,
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_message = "my internet is not working"

    result = run_support_agent(test_message)

    print()
    print("=" * 80)
    print("FINAL SUPPORT AGENT OUTPUT")
    print("=" * 80)

    print()
    print("Customer message:")
    print(test_message)

    print()
    print("Intent:")
    print(result["intent"])

    print()
    print("Confidence:")
    print(round(result["confidence"], 4))

    print()
    print("Decision:")
    print(result["decision"])

    print()
    print("Reason:")
    print(result["reason"])

    print()
    print("Draft reply:")
    print(result["draft_reply"])