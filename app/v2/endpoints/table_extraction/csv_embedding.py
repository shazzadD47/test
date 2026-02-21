from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from app.core.vector_store import VectorStore
from app.logging import logger
from app.v3.endpoints.get_title_summery.utils.embedding import insert_cost_document


def extract_text_from_csv_file(file_path: str) -> str | None:
    """Extract text from a CSV file and return the content as a string.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        str | None: The content of the CSV file as a string, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        csv_content = df.to_markdown(tablefmt="grid")

        return csv_content
    except Exception as e:
        logger.exception(f"An error occurred while reading the file: {e}")
        return None


def ingest_text(
    csv_content: str,
    project_id: str,
    flag_id: str,
    file_type: str,
    user_id: str,
) -> None:
    document = Document(page_content=csv_content)
    embedding_tokens = len(document.page_content.split())

    cost_per_1000_tokens = 0.02
    embedding_cost = embedding_tokens * cost_per_1000_tokens / 1000
    insert_cost_document(user_id, embedding_tokens, embedding_cost)

    document.metadata["project_id"] = project_id
    document.metadata["flag_id"] = flag_id
    document.metadata["file_type"] = file_type
    document.metadata["user_id"] = user_id

    VectorStore.add_documents([document])


def text_csv_fileprocessing(
    file_location: str, project_id: str, flag_id: str, user_id: str
) -> str | None:
    csv_content = extract_text_from_csv_file(file_location)

    if csv_content:
        ingest_text(csv_content, project_id, flag_id, "csv", user_id)

        return csv_content
    else:
        logger.exception("Failed to process CSV file.")
        return None


def extract_csv(
    file_location: str, project_id: str, flag_id: str, user_id: str
) -> str | None:

    try:
        result = text_csv_fileprocessing(file_location, project_id, flag_id, user_id)
        Path(file_location).unlink(missing_ok=True)

        return result
    except Exception as e:
        logger.exception(f"An error occurred while processing the CSV file: {e}")
