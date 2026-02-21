from typing import Any

import chardet
from langchain_core.documents import Document

from app.logging import logger


def get_unique_langchain_contexts(docs: list[Document]) -> list[Document]:
    return list({doc.page_content: doc for doc in docs}.values())


def clean_langchain_contexts(
    docs: list[Document],
    remove_embeddings: bool = True,
    remove_project_id: bool = True,
    remove_paper_id: bool = True,
    remove_user_id: bool = True,
) -> list[Document]:
    for doc in docs:
        doc.page_content = doc.page_content.strip()

        if remove_embeddings:
            doc.metadata.pop("embedding", None)

        if remove_project_id:
            doc.metadata.pop("project_id", None)

        if remove_paper_id:
            doc.metadata.pop("paper_id", None)

        if remove_user_id:
            doc.metadata.pop("user_id", None)

    return docs


def combine_langchain_contexts(
    docs: list[Document], separator: str = "\n-----------\n\n"
) -> str:
    """Combine langchain contexts into a single string.

    Args:
        docs (list[Document]): List of documents
        separator (str, optional): Separator. Defaults to "\n-----------\n\n".

    Returns:
        str: _description_
    """
    return separator.join([doc.page_content for doc in docs])


def combine_string_contexts(
    contexts: list[str], separator: str = "\n-----------\n\n"
) -> str:
    """Combine string contexts into a single string.

    Args:
        contexts (list[str]): List of string contexts
        separator (str, optional): Separator. Defaults to "\n-----------\n\n".

    Returns:
        str: _description_
    """
    return separator.join(contexts)


def convert_byte_to_string(byte_data: bytes) -> str:
    """
    Detects the encoding of byte data and decodes it to a string.

    Args:
        byte_data: The bytes object to detect and decode.

    Returns:
        The decoded string.
    """
    try:
        if not byte_data:
            return ""

        detection = chardet.detect(byte_data)

        encoding = detection["encoding"]

        if encoding:
            try:
                decoded_string = byte_data.decode(encoding)
                return decoded_string
            except Exception:
                # If chardet's guess fails for some reason, try utf-8 as a fallback
                try:
                    decoded_string = byte_data.decode("utf-8", errors="replace")
                    return decoded_string
                except Exception:
                    try:
                        # Fallback to Latin-1 if all else fails, to avoid error
                        decoded_string = byte_data.decode("latin-1", errors="replace")
                    except Exception:
                        return ""
                    return decoded_string
        else:
            # If chardet couldn't detect anything, try common encodings
            try:
                decoded_string = byte_data.decode("utf-8", errors="replace")
                return decoded_string
            except Exception:
                try:
                    decoded_string = byte_data.decode("latin-1", errors="replace")
                except Exception:
                    return ""
                return decoded_string
    except Exception as e:
        logger.info(f"Byte conversion failed. Error: {e}")
        return ""


def convert_data_to_string(data: Any) -> str:
    if not data:
        return ""
    elif isinstance(data, bytes):
        return convert_byte_to_string(data)
    else:
        try:
            return str(data)
        except Exception as e:
            logger.info("Failed to convert into string, " f"Error: {e}")
            return ""
