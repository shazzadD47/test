import base64

from langchain_core.messages import HumanMessage

from app.v3.endpoints.agent_chat.services.image_agents import image_reasoning_agent


def validate_base64_image(base64_image: str) -> tuple[bool, str, int]:
    """
    Validate and clean base64 image data.

    Args:
        base64_image: The base64 encoded image string

    Returns:
        tuple: (is_valid, cleaned_base64, size_in_bytes)
    """
    try:
        # Clean the base64 string
        base64_image = base64_image.replace("\n", "").strip()

        # Remove data URI prefix if present
        if base64_image.startswith("data:image/png;base64,"):
            base64_image = base64_image.split(",")[1]

        # Verify it's valid base64
        decoded = base64.b64decode(base64_image)

        # Verify it's a valid PNG by checking header
        if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return True, base64_image, len(decoded)
        else:
            raise ValueError("Invalid PNG header")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")


async def get_image_context(
    query: str,
    base64_image: str,
) -> str:
    """
    Get the context from the image.
    """
    # Validate the base64 image data
    _, cleaned_base64, _ = validate_base64_image(base64_image)

    reasoning_chain = await image_reasoning_agent(
        description=query,
    )

    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": cleaned_base64,
                },
            },
            {
                "type": "text",
                "text": f"Describe the image where user query is {query}.",
            },
        ]
    )

    output = reasoning_chain.invoke([message])
    return output.content
