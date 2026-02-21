from uuid import uuid4

from celery.utils.log import get_task_logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf.errors import FileNotDecryptedError, WrongPasswordError

from app.configs import settings

celery_logger = get_task_logger(__name__)


def process_document_with_pypdf_loader(
    pdf_file_path: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supp_id: str = None,
):
    try:
        loader = PyPDFLoader(pdf_file_path, extract_images=True)
        data = loader.load()
    except Exception:
        celery_logger.exception(
            f"[flag_id: {flag_id}] Failed to load PDF with image extraction."
            " Trying without image extraction."
        )
        try:
            loader = PyPDFLoader(pdf_file_path)
            data = loader.load()
        except (FileNotDecryptedError, WrongPasswordError):
            celery_logger.error(f"[flag_id: {flag_id}] PDF is encrypted.")
            return []
        except Exception:
            celery_logger.exception(
                f"[flag_id: {flag_id}] Failed to load PDF without image extraction."
            )
            return []

    text_splitter = SentenceSplitter.from_defaults(
        chunk_size=settings.EMBEDDING_CHUNK_SIZE,
        chunk_overlap=settings.EMBEDDING_CHUNK_OVERLAP,
    )

    sc_chunks = []
    for doc in data:
        text_content = doc.page_content
        if len(text_content) >= 756:
            temp_chunks = text_splitter.split_text(text_content)
            sc_chunks.extend(temp_chunks)
        else:
            sc_chunks.append(text_content)

    langchain_docs = []
    for chunk in sc_chunks:
        metadata = {
            "project_id": project_id,
            "flag_id": flag_id,
            "user_id": user_id,
            "file_type": "document",
            "supplementary_id": supp_id if supp_id else None,
        }
        langchain_docs.append(
            Document(id=str(uuid4()), page_content=chunk, metadata=metadata)
        )

    return langchain_docs
