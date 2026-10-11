import streamlit as st

from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)

from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)

from config.settings import (
    RERANKER_MODEL,
    RERANK_TOP_N,
)


@st.cache_resource
def get_reranker():

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL,
        model_kwargs={
            "device": "cpu"
        },
    )

    return CrossEncoderReranker(
        model=cross_encoder,
        top_n=RERANK_TOP_N,
    )