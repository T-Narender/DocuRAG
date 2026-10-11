import uuid

import streamlit as st


def initialize_chat_sessions():

    if "chat_sessions" not in st.session_state:

        chat_id = uuid.uuid4().hex

        st.session_state.chat_sessions = {
            chat_id: {
                "title": "New chat",
                "messages": [],
                "file_signature": None,
            }
        }

        st.session_state.active_chat_id = (
            chat_id
        )


def create_new_chat():

    chat_id = uuid.uuid4().hex

    st.session_state.chat_sessions[
        chat_id
    ] = {
        "title": "New chat",
        "messages": [],
        "file_signature": None,
    }

    st.session_state.active_chat_id = (
        chat_id
    )


def get_active_chat():

    return st.session_state.chat_sessions[
        st.session_state.active_chat_id
    ]