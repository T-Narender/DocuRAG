import hashlib


def calculate_file_signature(
    file_bytes: bytes,
):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def create_source_text(
    file_name,
    documents,
):

    pages = []

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            pages.append(
                int(page) + 1
            )

    pages = sorted(
        set(pages)
    )

    if pages:

        page_string = ", ".join(
            f"Page {page}"
            for page in pages
        )

        return (
            f"📌 Sources: "
            f"{file_name} "
            f"({page_string})"
        )

    return (
        f"📌 Sources: {file_name}"
    )