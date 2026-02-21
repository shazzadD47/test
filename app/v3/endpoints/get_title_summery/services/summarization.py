from celery.utils.log import get_task_logger
from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlalchemy import and_, update

from app.configs import settings
from app.core.database.base import get_db_session
from app.core.database.crud import configure_retry
from app.core.database.models import FileDetails
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.vector_store import VectorStore
from app.utils.gemini import upload_file_to_gemini
from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.schemas import PDFSummarizationPayload
from app.v3.endpoints.get_title_summery.utils.prompt import gemini_prompt as prompt

client = genai.Client(
    api_key=settings.GOOGLE_API_KEY,
    http_options=types.HttpOptions(api_version="v1alpha"),
)


celery_logger = get_task_logger(__name__)


class Document:
    def __init__(self, page_content, metadata):
        self.id = None
        self.page_content = page_content
        self.metadata = metadata


class TitleSummary(BaseModel):
    title: str
    summary: str


def update_file_summary(
    flag_id: str, summary_text: str, supplementary_id: str = None, max_retries: int = 3
):

    @configure_retry(max_retries=max_retries)
    def _update_file_summary_with_retry():
        with get_db_session() as session:
            conditions = [FileDetails.flag_id == flag_id]
            if supplementary_id:
                conditions.append(FileDetails.supplementary_id == supplementary_id)

            query = (
                update(FileDetails)
                .where(and_(*conditions))
                .values(paper_summary=summary_text)
            )

            session.execute(query)
            session.commit()

    try:
        _update_file_summary_with_retry()
        return True
    except Exception:
        celery_logger.exception(
            f"[flag_id: {flag_id}] Failed to update paper_summary "
            f"[supplementary_id: {supplementary_id}]"
        )
        return False


def summarize_pdf_with_gemini(
    file_location: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supp_id: str = None,
    title: str = None,
    file_id: str = None,
):
    if supp_id is None:
        payload_flag_id = flag_id
    else:
        payload_flag_id = f"{flag_id}-supplementary-{supp_id}"

    uploaded_file = upload_file_to_gemini(file_location, flag_id)
    celery_logger.info(f"file Uploaded: {uploaded_file.uri}")

    contents = [prompt, uploaded_file]

    try:
        token_info = client.models.count_tokens(
            model=project_settings.GEMINI_MODEL_NAME, contents=contents
        )
    except Exception as e:
        celery_logger.exception(
            f"[flag_id: {flag_id}] Error counting tokens with Gemini" f"Error: {e}"
        )
        token_info = None

    total_retries = 3
    retry_count = 1
    while retry_count <= total_retries:
        try:
            response = client.models.generate_content(
                model=project_settings.GEMINI_MODEL_NAME,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TitleSummary,
                },
            )

            parsed: TitleSummary = response.parsed
            extracted_title = parsed.title.strip()
            summary_text = parsed.summary.strip()
            break
        except Exception:
            error_message = f"[flag_id: {flag_id}] Error summarizing PDF with Gemini"
            celery_logger.exception(error_message)
            if retry_count >= total_retries:
                celery_logger.exception(error_message)
                extracted_title = None
                summary_text = None
                break

            retry_count += 1
            continue

    if extracted_title is None or summary_text is None:
        try:
            client.files.delete(name=uploaded_file.name)
            celery_logger.info(f"File deleted from Gemini: {uploaded_file.name}")
        except Exception as e:
            celery_logger.warning(f"Failed to delete file from Gemini: {e}")
        return {
            "status": "failed",
            "flag_id": flag_id,
            "project_id": project_id,
            "user_id": user_id,
            "supplementary_id": supp_id,
            "tokens_used": token_info.total_tokens if token_info else 0,
        }

    title_verified = None

    if title and extracted_title:
        title_verified = title.strip().lower() == extracted_title.strip().lower()
    elif title:
        title_verified = False

    metadata = {
        "project_id": project_id,
        "flag_id": flag_id,
        "user_id": user_id,
        "file_type": "document",
        "supplementary_id": supp_id,
    }
    chunks = Document(page_content=summary_text, metadata=metadata)
    VectorStore.add_documents([chunks])
    update_file_summary(flag_id, summary_text, supp_id)
    summary_payload = PDFSummarizationPayload(
        flag_id=payload_flag_id,
        project_id=project_id,
        supplementary_id=supp_id,
        status="SUCCESS",
        summary_text=summary_text,
        title=extracted_title,
        title_verified=title_verified,
        file_id=file_id,
    )

    send_to_backend(
        BackendEventEnumType.PDF_SUMMARIZATION, summary_payload.model_dump()
    )

    try:
        client.files.delete(name=uploaded_file.name)
        celery_logger.info(f"File deleted from Gemini: {uploaded_file.name}")
    except Exception as e:
        celery_logger.warning(f"Failed to delete file from Gemini: {e}")

    return {
        "status": "success",
        "flag_id": flag_id,
        "project_id": project_id,
        "user_id": user_id,
        "supplementary_id": supp_id,
        "tokens_used": token_info.total_tokens,
    }
