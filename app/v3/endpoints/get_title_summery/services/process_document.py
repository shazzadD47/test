import copy
import json
import logging
from collections.abc import Iterator
from uuid import uuid4

import httpx
from celery.utils.log import get_task_logger
from langchain_core.documents import Document
from llama_index.core.node_parser import SentenceSplitter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.logging import logger

celery_logger = get_task_logger(__name__)


UPLOAD_FOLDER = "/data/external/"


def prepare_embedding_document(
    text: str,
    chunk_type: str,
    chunk_metadata: dict,
    flag_id: str,
    project_id: str,
    user_id: str,
    supplementary_id: str,
    figure_number: str = None,
    img_caption: str = None,
    img_footnote: str = None,
    table_number: str = None,
    table_body: str = None,
    table_caption: str = None,
    table_footnote: str = None,
) -> Document:
    heading = chunk_metadata.get("heading", "")

    if (
        figure_number
        and figure_number.strip()
        and figure_number not in ["no figure number"]
    ):
        heading = f"{heading} > Figure {figure_number}"
    elif (
        table_number
        and table_number.strip()
        and table_number not in ["no table number"]
    ):
        heading = f"{heading} > Table {table_number}"

    embedding_chunk = {
        "paper_section": heading,
        "chunk_type": chunk_type,
        "text": text,
    }

    if (
        figure_number
        and figure_number.strip()
        and figure_number not in ["no figure number"]
    ):
        embedding_chunk["figure_number"] = figure_number
    if img_caption:
        embedding_chunk["img_caption"] = img_caption
    if img_footnote:
        embedding_chunk["img_footnote"] = img_footnote
    if (
        table_number
        and table_number.strip()
        and table_number not in ["no table number"]
    ):
        embedding_chunk["table_number"] = table_number
    if table_body:
        embedding_chunk["table_body"] = table_body
    if table_caption:
        embedding_chunk["table_caption"] = table_caption
    if table_footnote:
        embedding_chunk["table_footnote"] = table_footnote

    metadata = {
        "page": chunk_metadata.get("page_id", 0),
        "chunk_id": chunk_metadata.get("chunk_id"),
        "flag_id": flag_id,
        "project_id": project_id,
        "user_id": user_id,
        "file_type": "document",
        "supplementary_id": supplementary_id,
    }
    if "cell_id" in chunk_metadata:
        metadata["cell_id"] = chunk_metadata.get("cell_id")

    return Document(
        page_content=json.dumps(embedding_chunk, ensure_ascii=False), metadata=metadata
    )


def prepare_embedding_chunks(
    chunks: list[dict] | Iterator[dict],
    flag_id: str,
    project_id: str,
    user_id: str,
    supplementary_id: str,
    split_text: bool = True,
) -> list[Document]:
    chunks = copy.deepcopy(chunks)

    splitter = SentenceSplitter.from_defaults(
        chunk_size=settings.EMBEDDING_CHUNK_SIZE,
        chunk_overlap=settings.EMBEDDING_CHUNK_OVERLAP,
    )

    chunk_copies = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if (
            metadata.get("heading", "").strip().lower()
            == chunk.get("value", "").strip().lower()
        ):
            continue

        if chunk.get("subfigures"):
            for subfigure in chunk.get("subfigures"):
                chunk_copies.append(subfigure)

            del chunk["subfigures"]
            chunk_copies.append(chunk)
        else:
            chunk_copies.append(chunk)

    chunk_docs = []
    chunk_id_counter = 0
    for chunk in chunk_copies:
        chunk_metadata = chunk.get("metadata", {})
        text = chunk.get("value", "")

        if (
            len(text) > 756
            and split_text
            and chunk.get("type") not in ["image", "table"]
        ):
            temp_chunks = splitter.split_text(text)

            for text in temp_chunks:
                chunk_metadata["chunk_id"] = chunk_id_counter
                chunk_id_counter += 1
                doc = prepare_embedding_document(
                    text,
                    chunk.get("type"),
                    chunk_metadata,
                    flag_id,
                    project_id,
                    user_id,
                    supplementary_id,
                    figure_number=chunk.get("figure_number"),
                    img_caption=chunk.get("img_caption"),
                    img_footnote=chunk.get("img_footnote"),
                    table_number=chunk.get("table_number"),
                    table_body=chunk.get("table_body"),
                    table_caption=chunk.get("table_caption"),
                    table_footnote=chunk.get("table_footnote"),
                )
                chunk_docs.append(doc)
        else:
            chunk_metadata["chunk_id"] = chunk_id_counter
            chunk_id_counter += 1
            doc = prepare_embedding_document(
                text,
                chunk.get("type"),
                chunk_metadata,
                flag_id,
                project_id,
                user_id,
                supplementary_id,
                figure_number=chunk.get("figure_number"),
                img_caption=chunk.get("img_caption"),
                img_footnote=chunk.get("img_footnote"),
                table_number=chunk.get("table_number"),
                table_body=chunk.get("table_body"),
                table_caption=chunk.get("table_caption"),
                table_footnote=chunk.get("table_footnote"),
            )
            chunk_docs.append(doc)

    return chunk_docs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.75, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def post_to_mineru(
    file_data: bytes, payload: dict, file_extension: str, url: str | None = None
) -> dict:
    """
    Post file data to MinerU with retry logic.
    - If `url` is None, uses the default project_settings.MINERU_SERVICE.
    - Otherwise, posts to the provided URL (appended to service base).
    """
    target_url = (
        project_settings.MINERU_SERVICE
        if url is None
        else f"{project_settings.MINERU_SERVICE.rstrip('/')}/{url.lstrip('/')}"
    )

    logger.info(f"File Extension: {file_extension}")
    logger.info(f"Mineru Payload: {payload}")
    logger.info(f"Mineru Service: {project_settings.MINERU_SERVICE}")
    logger.info(f"Posting URL: {target_url}")

    with httpx.Client(timeout=project_settings.API_TIME_OUT) as client:
        response = client.post(
            target_url,
            files={
                "file": (
                    f"uploaded_file_{uuid4()}{file_extension}",
                    file_data,
                    f"application/{file_extension}",
                )
            },
            data=payload,
            headers={"X-API-Key": project_settings.AI_MINERU_API_SECRET},
        )
        response.raise_for_status()
        return response.json()
