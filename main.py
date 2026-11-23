
from langchain_core.messages import HumanMessage, AIMessage

from src.rag_pipeline import rag_chain

ROLES = ["Student", "Trainer", "Mentor", "Admin"]


def choose_role():

    print("\n===== LearnAI Chatbot =====")

    for i, role in enumerate(ROLES, start=1):
        print(f"{i}. {role}")

    while True:

        choice = input("\nSelect your role (1-4): ")

        if choice in ["1", "2", "3", "4"]:
            return ROLES[int(choice) - 1]

        print("Please enter a valid choice.")


def main():

    role = choose_role()

    chat_history = []

    print(f"\nWelcome! You are logged in as {role}.")
    print("Type 'exit' to stop.")
    print("Type 'clear' to clear conversation history.")

    while True:

        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Exiting chatbot...")
            break

        if question.lower() == "clear":
            chat_history.clear()
            print("History cleared.")
            continue

        answer = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history[-10:],
            "role": role
        })

        print(f"\nLearnAI ({role}): {answer}")

        chat_history.append(
            HumanMessage(content=question)
        )

        chat_history.append(
            AIMessage(content=answer)
        )


if __name__ == "__main__":
    main()
