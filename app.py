import streamlit as st

from chat.session import (
    initialize_chat_sessions,
    get_active_chat,
)

from chat.history import (
    format_chat_history,
)

from config.settings import (
    FALLBACK_API_KEY,
)

from document.processor import (
    process_and_index_file,
)

from llm.gemini import (
    create_gemini,
    create_rag_chain,
    retry_on_rate_limit,
)

from ui.sidebar import (
    render_sidebar,
)

from utils.helpers import (
    calculate_file_signature,
    create_source_text,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuRAG",
    page_icon="📂",
    layout="centered",
)


# ============================================================
# CHAT INITIALIZATION
# ============================================================

initialize_chat_sessions()


# ============================================================
# SIDEBAR
# ============================================================

(
    uploaded_file,
    api_key,
    response_depth,
) = render_sidebar()


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "DocuRAG — Local Document Intelligence Assistant"
)

st.write(
    "Upload a document and ask questions about its contents."
)


# ============================================================
# REQUIRE FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👋 Upload a document from the sidebar to begin."
    )

    st.stop()


# ============================================================
# ACTIVE CHAT
# ============================================================

active_chat = get_active_chat()


# ============================================================
# FILE SIGNATURE
# ============================================================

file_bytes = uploaded_file.getvalue()

file_signature = (
    calculate_file_signature(
        file_bytes
    )
)


# ============================================================
# HANDLE FILE CHANGE
# ============================================================

if (
    st.session_state.get(
        "current_file_signature"
    )
    != file_signature
):

    st.session_state.current_file_signature = (
        file_signature
    )

    if "retriever" in st.session_state:

        del st.session_state[
            "retriever"
        ]


# ============================================================
# ASSOCIATE CHAT WITH DOCUMENT
# ============================================================

if active_chat["file_signature"] is None:

    active_chat["file_signature"] = (
        file_signature
    )


chat_matches_document = (
    active_chat["file_signature"]
    == file_signature
)


# ============================================================
# BUILD RETRIEVER
# ============================================================

if "retriever" not in st.session_state:

    with st.spinner(
        "Processing document..."
    ):

        st.session_state.retriever = (
            process_and_index_file(
                file_bytes,
                uploaded_file.name,
            )
        )


# ============================================================
# GEMINI
# ============================================================

if (
    not api_key
    or api_key == FALLBACK_API_KEY
    and api_key == "YOUR_DEFAULT_FALLBACK_API_KEY"
):

    st.error(
        "Please provide a valid Gemini API key."
    )

    st.stop()


llm = create_gemini(
    api_key
)


# ============================================================
# HISTORY
# ============================================================

chat_history = format_chat_history(
    active_chat["messages"]
)


# ============================================================
# RAG CHAIN
# ============================================================

rag_chain = create_rag_chain(
    retriever=st.session_state.retriever,
    llm=llm,
    chat_history=chat_history,
    response_depth=response_depth,
)


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in active_chat["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

if not chat_matches_document:

    st.warning(
        "This chat belongs to another document. "
        "Please start a new chat for this document."
    )

    st.stop()


user_question = st.chat_input(
    "Ask a question about the document..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:

    # Save user message
    active_chat["messages"].append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    # Generate title
    if active_chat["title"] == "New chat":

        title = (
            user_question
            .replace("\n", " ")
            .strip()
        )

        active_chat["title"] = (
            title[:35]
            + ("..." if len(title) > 35 else "")
        )

    # User message
    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # Assistant
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching document..."
        ):

            try:

                source_documents = (
                    st.session_state
                    .retriever
                    .invoke(
                        user_question
                    )
                )

                answer = (
                    retry_on_rate_limit(
                        rag_chain,
                        user_question,
                    )
                )

                source_text = (
                    create_source_text(
                        uploaded_file.name,
                        source_documents,
                    )
                )

                full_response = (
                    f"{answer}\n\n"
                    f"*{source_text}*"
                )

                st.markdown(
                    full_response
                )

                active_chat[
                    "messages"
                ].append(
                    {
                        "role": "assistant",
                        "content": full_response,
                    }
                )

            except Exception as error:

                st.error(
                    f"Request failed: {error}"
                )

                active_chat[
                    "messages"
                ].pop()