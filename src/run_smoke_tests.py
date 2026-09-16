from support_agent import run_support_agent
from query_vector_retrieval import retrieve_top_k


TEST_MESSAGES = [
    "@117627 @GloCare May be I should reactivate my glo line again?",

    "@GloCare I have bonus unlocked and available for use: Voice: N960.80; Data: 105MB. When am I entitle to use the bonus and is it for online calls only? ?? Please, revert back to me? Thanks.",

    "@GloCare No, I didn't send any message to 2442. But I have been doing USSD mobile banking successfully until about one week ago when it started giving me \"connection problem or invalid MMI code\" error, and with each attempt N5 service charge is deducted.",

    "@GloCare I subscribed for my phone 2 days but suddenly yesterday I saw a message notifying me that av used up my data",

    "@GloCare hi, I want to port from MTN to GLO network what and what would I need? Reply asap",
]


def main():
    for i, message in enumerate(TEST_MESSAGES, start=1):
        print("=" * 80)
        print(f"SMOKE TEST {i}")
        print("=" * 80)

        print(f"\nCustomer message:\n{message}")

        # Retrieve the same evidence used by the agent
        retrieved_examples = retrieve_top_k(
            message,
            top_k=3,
        )

        print("\nRetrieved historical evidence:")

        for rank, example in enumerate(retrieved_examples, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"Distance: {example['distance']:.4f}")
            print(
                f"Historical customer: "
                f"{example['historical_customer_message']}"
            )
            print(
                f"Historical response: "
                f"{example['historical_gloCare_response']}"
            )

        # Run complete agent
        result = run_support_agent(message)

        print(f"\nPredicted intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")

        print(f"\nDecision: {result['decision']}")
        print(f"Reason: {result['reason']}")

        print(f"\nDraft reply:\n{result['draft_reply']}")

        print()


if __name__ == "__main__":
    main()