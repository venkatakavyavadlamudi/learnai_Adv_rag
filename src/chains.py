
from langchain_core.output_parsers import StrOutputParser

from src.config import llm

from src.prompts import (
    intent_prompt,
    contextualize_prompt,
    answer_prompt,
    history_prompt,
    general_prompt
)

# Intent classification chain
intent_chain = (
    intent_prompt | llm | StrOutputParser()
)

# Question rewriting chain
contextualize_chain = (
    contextualize_prompt | llm | StrOutputParser()
)

# Document answer chain
document_answer_chain = (
    answer_prompt | llm | StrOutputParser()
)

# History answer chain
history_answer_chain = (
    history_prompt | llm | StrOutputParser()
)

# General conversation chain
general_chain = (
    general_prompt | llm | StrOutputParser()
)
