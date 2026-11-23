

from langchain_core.runnables import RunnableLambda

from src.config import get_retriever

from src.chains import (
    contextualize_chain,
    document_answer_chain,
    intent_chain,
    history_answer_chain
)

UNKNOWN_ANSWER = "I don't know based on the provided document."


def retrieve_documents(inputs):

    question = inputs["input"]
    history = inputs["chat_history"]
    role = inputs["role"]

    # Rewrite question internally
    standalone_question = contextualize_chain.invoke({
        "input": question,
        "chat_history": history
    })

    # Retrieve only the logged-in user's documents
    retriever = get_retriever(role)

    documents = retriever.invoke(standalone_question)

    # Verify role
    role_documents = [
        doc for doc in documents
        if doc.metadata.get("role") == role
    ]

    if not role_documents:
        return UNKNOWN_ANSWER

    context = "\n\n".join(
        doc.page_content for doc in role_documents
    )

    if not context.strip():
        return UNKNOWN_ANSWER

    # Generate answer only from document context
    answer = document_answer_chain.invoke({
        "input": standalone_question,
        "context": context,
        "role": role
    })

    return answer


def route_request(inputs):

    question = inputs["input"]
    history = inputs["chat_history"]

    # Classify internally without printing
    intent = intent_chain.invoke({
        "input": question,
        "chat_history": history
    }).strip().upper()

    if intent == "DOCUMENT":

        return retrieve_documents(inputs)

    elif intent == "HISTORY":

        return history_answer_chain.invoke({
            "input": question,
            "chat_history": history
        })

    else:

        return UNKNOWN_ANSWER


rag_chain = RunnableLambda(route_request)
