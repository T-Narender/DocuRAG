def format_chat_history(messages):

    formatted = []

    for message in messages[-6:]:

        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        content = message["content"]

        if role == "Assistant":

            content = content.split(
                "\n\n📌"
            )[0]

        formatted.append(
            f"{role}: {content}"
        )

    return "\n".join(
        formatted
    )