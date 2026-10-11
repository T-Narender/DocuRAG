from langchain_classic.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    ContextualCompressionRetriever,
)

from retrieval.reranker import get_reranker

from config.settings import RETRIEVAL_K


def create_hybrid_retriever(
    vectorstore,
    chunks,
):

    # Dense retrieval
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )

    # Sparse retrieval
    bm25_retriever = (
        BM25Retriever.from_documents(
            chunks
        )
    )

    bm25_retriever.k = RETRIEVAL_K

    # Hybrid retrieval
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],
        weights=[
            0.5,
            0.5,
        ],
    )

    # Reranking
    reranker = get_reranker()

    final_retriever = (
        ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever,
        )
    )

    return final_retriever