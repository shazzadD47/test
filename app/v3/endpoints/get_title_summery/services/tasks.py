import os

import fitz
from celery.utils.log import get_task_logger
from langchain_core.documents import Document

from app.configs import settings
from app.core.celery.app import celery_app
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.vector_store import VectorStore
from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.constant import TEXT_CHUNK_TYPES
from app.v3.endpoints.get_title_summery.schemas import (
    ConvertPDFToImagePayload,
    ImageInfo,
    MinerUOutputStatusPayload,
)
from app.v3.endpoints.get_title_summery.services.crud import (
    delete_existing_figure_details,
    delete_existing_table_details,
    get_page_details,
    insert_figure_details,
    insert_page_details,
    insert_table_details,
)
from app.v3.endpoints.get_title_summery.services.legend_detection import detect_legends
from app.v3.endpoints.get_title_summery.services.pdf_data_extraction import extract_text
from app.v3.endpoints.get_title_summery.services.process_document import (
    post_to_mineru,
    prepare_embedding_chunks,
)
from app.v3.endpoints.get_title_summery.services.process_documents_fallback import (
    process_document_with_pypdf_loader,
)
from app.v3.endpoints.get_title_summery.services.process_meta_document import (
    process_metadata_and_save,
)
from app.v3.endpoints.get_title_summery.services.summarization import (
    summarize_pdf_with_gemini,
)
from app.v3.endpoints.get_title_summery.utils.chunk_details import (
    prepare_figure_details,
    prepare_table_details,
)
from app.v3.endpoints.get_title_summery.utils.file_utils import _read_notebook_file
from app.v3.endpoints.get_title_summery.utils.payload_utils import (
    build_ai_annotation_payload,
)
from app.v3.endpoints.get_title_summery.utils.s3_utils import build_public_s3_url
from app.v3.endpoints.get_title_summery.utils.utils import (
    extract_and_upload_page,
    upload_file_to_storage,
    upload_single_image_to_storage,
)

celery_logger = get_task_logger(__name__)


@celery_app.task(name="extract_text_task")
def extract_text_task(file_location, project_id, flag_id, user_id, supp_id: str = None):
    result = extract_text(file_location, project_id, flag_id, user_id, supp_id)

    payload = MinerUOutputStatusPayload(
        flag_id=flag_id,
        supplementary_id=supp_id,
        status="2ND_PASS_COMPLETED",
        response_type="final",
        message="Text extraction completed successfully.",
        annotations=None,
    )
    send_to_backend(BackendEventEnumType.MINERU_OUTPUT_STATUS, payload.model_dump())

    return result


@celery_app.task(name="embedding_meta_task")
def embedding_meta_task(
    project_id: str, flag_id: str, user_id: str, extra_metadata: dict
):
    return process_metadata_and_save(project_id, flag_id, user_id, extra_metadata)


@celery_app.task(name="summarize_pdf_with_gemini_task")
def summarize_pdf_with_gemini_task(
    file_location: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supp_id: str = None,
    title: str = None,
    file_id: str = None,
):
    return summarize_pdf_with_gemini(
        file_location, project_id, flag_id, user_id, supp_id, title, file_id
    )


@celery_app.task(name="convert_pdf_to_image_task")
def convert_pdf_to_image_task(
    file_location: str,
    flag_id: str,
    project_id: str,
    supp_id: str = None,
    file_id: str = None,
):
    file_extension = os.path.splitext(file_location)[-1].lower()
    if file_extension == ".ipynb":
        file_location = _read_notebook_file(file_location)

    pdf_document = fitz.open(file_location)

    page_details_list = []

    images_payload: list[ImageInfo] = []

    for page_number in range(len(pdf_document)):
        try:
            page_details = extract_and_upload_page(
                pdf_document,
                page_number,
                settings.PDF_PAGE_IMAGE_RESOLUTION,
                flag_id,
                project_id,
                supp_id,
            )
            page_details_list.append(page_details)
            image_data = ImageInfo(
                image_number=page_details.page_number,
                image_url=build_public_s3_url(page_details.bucket_path),
                status="SUCCESS",
            )
            images_payload.append(image_data)

        except Exception:

            celery_logger.warning(f"[flag_id: {flag_id}] Failed page {page_number + 1}")
            failed_image_data = ImageInfo(
                image_number=page_number + 1,
                image_url=None,
                status="FAILED",
            )
            images_payload.append(failed_image_data)

    insert_page_details(page_details_list)
    overall_status = (
        "FAILED" if any(p.status == "FAILED" for p in images_payload) else "SUCCESS"
    )
    if supp_id is None:
        payload_flag_id = flag_id
    else:
        payload_flag_id = f"{flag_id}-supplementary-{supp_id}"

    final_payload = ConvertPDFToImagePayload(
        flag_id=payload_flag_id,
        project_id=project_id,
        supplementary_id=supp_id,
        status=overall_status,
        images=images_payload,
        file_id=file_id,
    )
    send_to_backend(
        BackendEventEnumType.CONVERT_PDF_TO_IMAGE, final_payload.model_dump()
    )

    return {
        "status": "success",
        "flag_id": flag_id,
        "project_id": project_id,
        "supplementary_id": supp_id,
        "pages_inserted": len(page_details_list),
    }


@celery_app.task(name="file_upload_fail_store")
def file_upload_fail_store(file_path: str, upload_path: str, filename: str):
    try:
        upload_file_to_storage(file_path, upload_path, filename)
        celery_logger.info(f"File {filename} uploaded to {upload_path} successfully.")
    except Exception as e:
        celery_logger.error(f"Failed to store file {filename}: {e}")
        raise e


@celery_app.task(name="single_image_upload")
def single_image_upload(
    image_bytes: bytes, flag_id: str, project_id: str, supp_id: str = None
):
    return upload_single_image_to_storage(image_bytes, flag_id, project_id, supp_id)


@celery_app.task(name="process_layout_chunks")
def process_layout_chunks(
    file_path: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supplementary_id: str = None,
    file_id: str = None,
    file_extension: str = None,
) -> dict | None:

    try:
        chunks = parse_document_layout(
            file_path,
            flag_id,
            project_id,
            user_id,
            supplementary_id,
            file_id,
            file_extension,
            retry_enabled=False,
        )
    except Exception as e:
        celery_logger.exception(
            f"Layout processing failed for flag_id: {flag_id}." f"Error: {e}"
        )

        upload_file_to_storage(
            file_path,
            upload_path="documents/failures/mineru/layout",
            filename=os.path.basename(file_path),
        )

        failed_payload = MinerUOutputStatusPayload(
            flag_id=flag_id,
            file_id=file_id,
            supplementary_id=supplementary_id,
            status="FAILED",
            response_type="failed",
            message=(f"[flag_id: {flag_id}] Layout processing failed."),
        )
        send_to_backend(
            BackendEventEnumType.MINERU_OUTPUT_STATUS,
            failed_payload.model_dump(),
        )
        return {
            "status": "failed",
            "flag_id": flag_id,
            "project_id": project_id,
            "user_id": user_id,
            "supplementary_id": supplementary_id,
            "chunks_embedded": 0,
            "figures_inserted": 0,
            "tables_inserted": 0,
        }

    if chunks:
        embedding_chunks = prepare_embedding_chunks(
            chunks, flag_id, project_id, user_id, supplementary_id, split_text=False
        )
        VectorStore.add_documents(embedding_chunks)

        image_chunks = [chunk for chunk in chunks if chunk.get("type") == "image"]
        table_chunks = [chunk for chunk in chunks if chunk.get("type") == "table"]
        page_details = get_page_details(file_path, project_id, flag_id)

        figure_details_list = []
        for image_chunk in image_chunks:
            figure_details = prepare_figure_details(
                flag_id, project_id, page_details, image_chunk
            )
            figure_details_list.extend(figure_details)

        insert_figure_details(figure_details_list)

        table_details_list = []
        for table_chunk in table_chunks:
            table_details = prepare_table_details(
                flag_id, project_id, page_details, table_chunk
            )
            table_details_list.append(table_details)

        insert_table_details(table_details_list)
        # detect_legends(flag_id, figure_details_list)

        event_payload_data = build_ai_annotation_payload(
            figure_details_list, table_details_list
        )

        payload = MinerUOutputStatusPayload(
            flag_id=flag_id,
            file_id=file_id,
            supplementary_id=supplementary_id,
            status="2ND_PASS_COMPLETED",
            response_type="final",
            message=None,
            annotations=event_payload_data,
        )
        send_to_backend(BackendEventEnumType.MINERU_OUTPUT_STATUS, payload.model_dump())

        celery_logger.info(f"Processed layout chunks for flag_id: {flag_id}.")

        return {
            "status": "success",
            "flag_id": flag_id,
            "project_id": project_id,
            "user_id": user_id,
            "supplementary_id": supplementary_id,
            "chunks_embedded": len(embedding_chunks),
            "figures_inserted": len(figure_details_list),
            "tables_inserted": len(table_details_list),
        }


@celery_app.task(name="process_layout_chunks_retry")
def process_layout_chunks_retry(
    file_path: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supplementary_id: str = None,
    file_id: str = None,
    file_extension: str = None,
) -> dict | None:
    try:

        exists = VectorStore.check_flag_id_exists(flag_id)

        if not exists:
            celery_logger.info(
                f"Vectors for flag_id: {flag_id} do not exist. Parsing document layout."
            )
            chunks = parse_document_layout(
                file_path,
                flag_id,
                project_id,
                user_id,
                supplementary_id,
                file_id,
                file_extension,
                retry_enabled=False,  # set False here
            )

            embedding_chunks = prepare_embedding_chunks(
                chunks, flag_id, project_id, user_id, supplementary_id, split_text=False
            )
            VectorStore.add_documents(embedding_chunks)
        else:
            celery_logger.info(f"Vectors for flag_id: {flag_id} already exist.")

            chunks = parse_document_layout(
                file_path,
                flag_id,
                project_id,
                user_id,
                supplementary_id,
                file_id,
                file_extension,
                retry_enabled=True,
            )

    except Exception as e:
        celery_logger.exception(
            f"Layout processing failed for flag_id: {flag_id}. Error: {e}"
        )
        upload_file_to_storage(
            file_path,
            upload_path="documents/failures/mineru/layout",
            filename=os.path.basename(file_path),
        )

        failed_payload = MinerUOutputStatusPayload(
            flag_id=flag_id,
            file_id=file_id,
            supplementary_id=supplementary_id,
            status="FAILED",
            response_type="failed",
            message=(f"[flag_id: {flag_id}] Layout processing failed."),
        )
        send_to_backend(
            BackendEventEnumType.MINERU_OUTPUT_STATUS,
            failed_payload.model_dump(),
        )
        return {
            "status": "failed",
            "flag_id": flag_id,
            "project_id": project_id,
            "user_id": user_id,
            "supplementary_id": supplementary_id,
            "chunks_embedded": 0,
            "figures_inserted": 0,
            "tables_inserted": 0,
        }
    if chunks:
        image_chunks = [chunk for chunk in chunks if chunk.get("type") == "image"]
        table_chunks = [chunk for chunk in chunks if chunk.get("type") == "table"]
        delete_existing_figure_details(flag_id)
        delete_existing_table_details(flag_id)
        page_details = get_page_details(file_path, project_id, flag_id)

        figure_details_list = []
        for image_chunk in image_chunks:
            figure_details = prepare_figure_details(
                flag_id, project_id, page_details, image_chunk
            )
            figure_details_list.extend(figure_details)
        insert_figure_details(figure_details_list)

        table_details_list = []
        for table_chunk in table_chunks:
            table_details = prepare_table_details(
                flag_id, project_id, page_details, table_chunk
            )
            table_details_list.append(table_details)
        insert_table_details(table_details_list)
        detect_legends(flag_id, figure_details_list)

        event_payload_data = build_ai_annotation_payload(
            figure_details_list, table_details_list
        )

        payload = MinerUOutputStatusPayload(
            flag_id=flag_id,
            file_id=file_id,
            supplementary_id=supplementary_id,
            status="2ND_PASS_COMPLETED",
            response_type="final",
            message=None,
            annotations=event_payload_data,
        )
        send_to_backend(BackendEventEnumType.MINERU_OUTPUT_STATUS, payload.model_dump())

        celery_logger.info(f"Processed layout chunks for flag_id: {flag_id}.")

        return {
            "status": "success",
            "flag_id": flag_id,
            "project_id": project_id,
            "user_id": user_id,
            "supplementary_id": supplementary_id,
            "chunks_embedded": len(embedding_chunks) if not exists else 0,
            "figures_inserted": len(figure_details_list),
            "tables_inserted": len(table_details_list),
        }


def parse_document_layout(
    file_path: str,
    flag_id: str,
    project_id: str,
    user_id: str = None,
    supplementary_id: str = None,
    file_id: str = None,
    file_extension: str = None,
    url: str | None = None,
    retry_enabled: bool = False,
) -> list[Document]:
    """
    Parse document layout via MinerU using the unified post_to_mineru function.

    - If retry_enabled=True, posts to `url="api/v1/parse/layout"` by default.
    - Otherwise, uses the provided URL or default MinerU service.
    """

    if supplementary_id and supplementary_id.strip():
        mineru_flag_id = f"{flag_id}-supplementary-{supplementary_id}"
    else:
        mineru_flag_id = flag_id

    payload = {
        "flag_id": mineru_flag_id,
        "project_id": project_id,
        "user_id": user_id,
        "file_id": file_id,
        "layout_mode": "true",
        "layout_ocr_mode": "true",
        "webhook_url": project_settings.MINERU_WEBHOOK_URL,
        "file_location": file_path,
    }

    # Read file and post to MinerU
    with open(file_path, "rb") as file:
        file_data = file.read()
        post_url = url or ("layout" if retry_enabled else None)
        data = post_to_mineru(file_data, payload, file_extension, post_url)

        celery_app.send_task(
            "log_mineru_output",
            (flag_id, "initial", data, file_id),
        )

    if data is not None and isinstance(data, dict):
        chunks = data.get("chunks", [])
        non_text_chunks = list(
            filter(
                lambda x: x.get("type") not in TEXT_CHUNK_TYPES and x.get("value"),
                chunks,
            )
        )

        return non_text_chunks

    else:
        return []


@celery_app.task(name="process_text_chunks")
def process_text_chunks(
    chunks: list[dict],
    file_path: str,
    flag_id: str,
    supp_id: str,
    project_id: str,
    user_id: str,
    file_id: str | None = None,
):
    if not chunks:
        celery_logger.warning(
            f"[flag_id: {flag_id}] Failed to process document or chunks found."
        )
        upload_file_to_storage(
            file_path,
            upload_path="documents/failures/mineru/text",
            filename=os.path.basename(file_path),
        )
        chunks = process_document_with_pypdf_loader(
            pdf_file_path=file_path,
            project_id=project_id,
            flag_id=flag_id,
            user_id=user_id,
            supp_id=supp_id,
        )

        chunks = list(
            filter(lambda x: x.type in TEXT_CHUNK_TYPES and x.page_content, chunks)
        )
    else:
        text_and_notebook_chunks = list(
            filter(
                lambda x: x.get("type") in TEXT_CHUNK_TYPES and x.get("value"), chunks
            )
        )

        chunks = prepare_embedding_chunks(
            text_and_notebook_chunks, flag_id, project_id, user_id, supp_id
        )

    if chunks:
        VectorStore.add_documents(chunks)
    else:
        celery_logger.warning(f"[flag_id: {flag_id}] No chunks found after fallback.")
        upload_file_to_storage(
            file_path,
            upload_path="documents/failures/no_chunks",
            filename=os.path.basename(file_path),
        )

    return {
        "status": "success",
        "flag_id": flag_id,
        "project_id": project_id,
        "user_id": user_id,
        "supplementary_id": supp_id,
        "chunks_embedded": len(chunks),
    }
