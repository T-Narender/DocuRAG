from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


def load_documents(
    file_path: str,
    suffix: str,
):

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
    }

    loader_class = loaders.get(
        suffix.lower()
    )

    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    return loader_class(file_path).load()