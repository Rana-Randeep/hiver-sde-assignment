from support_agent import run_support_agent


def print_result(customer_message, result):
    print()
    print("=" * 80)
    print("GLOCARE AI CUSTOMER-SUPPORT AGENT")
    print("=" * 80)

    print()
    print("Customer message:")
    print(customer_message)

    print()
    print("-" * 80)
    print("INTENT")
    print("-" * 80)
    print(result["intent"])

    print()
    print("Confidence:")
    print(round(result["confidence"], 4))

    print()
    print("-" * 80)
    print("DECISION")
    print("-" * 80)
    print(result["decision"])

    print()
    print("Reason:")
    print(result["reason"])

    print()
    print("-" * 80)
    print("DRAFT REPLY")
    print("-" * 80)
    print(result["draft_reply"])

    print()
    print("=" * 80)


def main():
    print("=" * 80)
    print("GloCare AI Customer-Support Agent")
    print("=" * 80)
    print()
    print("Type a customer message and press Enter.")
    print("Type 'exit' to stop.")
    print()

    while True:
        customer_message = input("Customer: ").strip()

        if customer_message.lower() in {"exit", "quit"}:
            print("\nExiting agent demo.")
            break

        if not customer_message:
            print("Please enter a customer message.")
            continue

        try:
            result = run_support_agent(customer_message)
            print_result(customer_message, result)

        except Exception as exc:
            print()
            print("Agent error:")
            print(exc)
            print()


if __name__ == "__main__":
    main()