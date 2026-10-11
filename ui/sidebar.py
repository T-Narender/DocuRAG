import streamlit as st

from config.settings import (
    SUPPORTED_TYPES,
    FALLBACK_API_KEY,
)

from chat.session import (
    create_new_chat,
)


def render_sidebar():

    with st.sidebar:

        st.header(
            "Configuration"
        )

        uploaded_file = (
            st.file_uploader(
                "Upload a policy document",
                type=SUPPORTED_TYPES,
            )
        )

        custom_api_key = (
            st.text_input(
                "Gemini API key",
                type="password",
            )
        )

        response_depth = (
            st.selectbox(
                "Select response depth",
                [
                    "Short",
                    "Medium",
                    "Detailed",
                ],
            )
        )

        api_key = (
            custom_api_key.strip()
            or FALLBACK_API_KEY
        )

        st.divider()

        st.subheader(
            "Chat history"
        )

        if st.button(
            "＋ New chat",
            use_container_width=True,
        ):

            create_new_chat()

            st.rerun()

        for chat_id, chat in reversed(
            list(
                st.session_state
                .chat_sessions
                .items()
            )
        ):

            title = chat["title"]

            if chat_id == (
                st.session_state
                .active_chat_id
            ):

                title = f"💬 {title}"

            if st.button(
                title,
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):

                st.session_state.active_chat_id = (
                    chat_id
                )

                st.rerun()

    return (
        uploaded_file,
        api_key,
        response_depth,
    )