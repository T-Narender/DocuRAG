import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config.settings import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from document.loaders import load_documents

from retrieval.hybrid import (
    create_hybrid_retriever,
)


def process_and_index_file(
    file_bytes: bytes,
    file_name: str,
):

    suffix = Path(
        file_name
    ).suffix.lower()

    temporary_path = None

    try:

        # Save temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:

            temporary_file.write(
                file_bytes
            )

            temporary_path = (
                temporary_file.name
            )

        # Load
        documents = load_documents(
            temporary_path,
            suffix,
        )

        # Chunk
        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:
            raise ValueError(
                "No extractable text found."
            )

        # Clear previous vector store
        if VECTORSTORE_DIR.exists():

            try:
                shutil.rmtree(
                    VECTORSTORE_DIR
                )
            except PermissionError:
                pass

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Embeddings
        embeddings = (
            HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
        )

        # Chroma
        vectorstore = (
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(
                    VECTORSTORE_DIR
                ),
            )
        )

        # Hybrid + reranking
        retriever = (
            create_hybrid_retriever(
                vectorstore,
                chunks,
            )
        )

        return retriever

    finally:

        if temporary_path:

            Path(
                temporary_path
            ).unlink(
                missing_ok=True
            )