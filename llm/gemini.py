import time

from langchain_core.exceptions import (
    ModelRateLimitError,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnablePassthrough,
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)

from llm.prompts import (
    get_rag_prompt,
    response_instructions,
)


def create_gemini(
    api_key: str,
):

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def create_rag_chain(
    retriever,
    llm,
    chat_history,
    response_depth,
):

    prompt = get_rag_prompt()

    def format_docs(documents):

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    return (
        {
            "context": (
                retriever
                | format_docs
            ),

            "chat_history": (
                lambda _: chat_history
            ),

            "question": (
                RunnablePassthrough()
            ),

            "depth_instruction": (
                lambda _: response_instructions(
                    response_depth
                )
            ),
        }

        | prompt

        | llm

        | StrOutputParser()
    )


def retry_on_rate_limit(
    rag_chain,
    question,
    max_retries=2,
    wait_seconds=35,
):

    try:

        return rag_chain.invoke(
            question
        )

    except ModelRateLimitError:

        for attempt in range(
            max_retries
        ):

            time.sleep(
                wait_seconds
            )

            try:

                return rag_chain.invoke(
                    question
                )

            except ModelRateLimitError:

                continue

        raise