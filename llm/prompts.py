from langchain_core.prompts import (
    ChatPromptTemplate,
)


def response_instructions(
    depth: str,
):

    instructions = {

        "Short":
            "Answer with at most 2-3 precise bullet points or short sentences.",

        "Medium":
            "Give a clear standard answer with important supporting details.",

        "Detailed":
            "Give a structured and comprehensive explanation using relevant details from the document.",
    }

    return instructions.get(
        depth,
        instructions["Medium"],
    )


def get_rag_prompt():

    return ChatPromptTemplate.from_template(
        """
You are a helpful and precise document
analysis assistant.

Answer the user's question using ONLY
the retrieved document context.

Use conversation history when useful
for understanding follow-up questions.

If the answer cannot be found in the
retrieved context, clearly say that the
information is not available in the
uploaded document.

Do not invent facts.

Response style:
{depth_instruction}

Conversation History:
{chat_history}

Retrieved Document Context:
{context}

Question:
{question}

Answer:
"""
    )