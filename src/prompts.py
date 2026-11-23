
from langchain_core.prompts import ChatPromptTemplate


# 1. Intent detection prompt
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Classify the user's question into one category:

        DOCUMENT: The question asks about information that may
        be available in the authorized policy document.

        HISTORY: The question refers to information discussed
        earlier in the conversation.

        OUT_OF_SCOPE: The question is unrelated to the policy
        document or previous document-based conversation.

        Return only one category:
        DOCUMENT, HISTORY, or OUT_OF_SCOPE.
        """
    ),
    ("human", "{input}")
])


# 2. Question contextualization prompt
contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Rewrite the user's question as a standalone question
        using the conversation history when necessary.

        Do not answer the question.
        Return only the rewritten question.
        """
    ),
    ("human", "Chat History:\n{chat_history}\n\nQuestion: {input}")
])


# 3. Document answer prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a strict role-based LearnAI document assistant.

        Answer using ONLY the provided document context.

        Rules:
        1. Do not use outside knowledge.
        2. Do not guess or make up information.
        3. Do not answer using general knowledge.
        4. Answer only if the information is available in the context.
        5. If the answer is not available, respond exactly:

        I don't know based on the provided document.

        User Role: {role}

        Document Context:
        {context}
        """
    ),
    ("human", "{input}")
])


# 4. Conversation history prompt
history_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer only using information from the previous
        document-grounded conversation.

        Do not introduce outside knowledge.

        If the answer cannot be found in the conversation,
        respond exactly:

        I don't know based on the provided document.
        """
    ),
    ("human", "Chat History:\n{chat_history}\n\nQuestion: {input}")
])


# 5. Out-of-scope prompt
general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Respond exactly with:

        I don't know based on the provided document.
        """
    ),
    ("human", "{input}")
])
