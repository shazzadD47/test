import os
from typing import Annotated
from uuid import uuid1

from boto3 import session
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.auth.S2S_Client import S2SClient, S2SSecurityModel
from app.configs import settings
from app.core.dependencies import validate_flag_id
from app.utils.utils import sanitize_flag_id
from app.v3.endpoints.get_title_summery.configs import settings as project_settings
from app.v3.endpoints.get_title_summery.constant import TEXT_CHUNK_TYPES
from app.v3.endpoints.get_title_summery.exceptions import (
    FlagIDNotFound,
)
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.schemas import MineruWebhookRequest
from app.v3.endpoints.get_title_summery.services.crud import (
    get_file_details,
    insert_file_details,
)
from app.v3.endpoints.get_title_summery.services.pdf_data_extraction import (
    extract_data_from_file_preprocess_sup,
    extract_data_from_pdf_preprocess,
)
from app.v3.endpoints.get_title_summery.services.tasks import celery_app
from app.v3.endpoints.get_title_summery.utils.embedding import flag_id_exists
from app.v3.endpoints.get_title_summery.utils.metadata import prepare_file_details
from app.v3.endpoints.get_title_summery.utils.utils import (
    generate_supp_id,
    secure_file_path,
    secure_filename,
)

router = APIRouter(tags=["Get Title Summary"])


@router.post("/get-image", status_code=status.HTTP_200_OK, response_model=list[str])
def list_image_urls(
    folder_name: Annotated[str, Depends(validate_flag_id)] = Query(
        ..., description="The name of the folder containing the images.", min_length=4
    )
) -> list[str]:
    flag_id, supplimentary_id = sanitize_flag_id(
        folder_name, return_supplimentary_info=True
    )

    client = session.Session().client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )

    if supplimentary_id:
        prefix = f"documents/pages/{flag_id}/{supplimentary_id}/"
    else:
        prefix = f"documents/pages/{flag_id}/"

    response = client.list_objects_v2(
        Bucket=settings.S3_SPACES_BUCKET, Prefix=prefix, Delimiter="/"
    )

    image_urls = []
    if "Contents" in response:
        for obj in response["Contents"]:
            image_url = f"{settings.S3_SPACES_PUBLIC_BASE_URL}/{obj['Key']}"
            image_urls.append(image_url)

    return image_urls


@router.post("/get-title-summary/", status_code=status.HTTP_200_OK)
def get_title_and_summary(
    file: UploadFile,
    user_id: Annotated[str, Form()],
    project_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    file_id: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
):
    logger.info(
        f"[client: {client}] "
        f"Received file: {file.filename} "
        f"for project_id: {project_id}, user_id: {user_id}"
        f", file_id: {file_id}"
    )

    flag_id = str(uuid1())
    image_uploaded = False
    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    file_path = f"{flag_id}{file_extension}"
    cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id)
    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)

    if image:
        image_uploaded = True
        celery_app.send_task(
            "single_image_upload",
            (
                image.file.read(),
                flag_id,
                project_id,
                None,
            ),  # Make sure image.file is read as bytes
        )

    try:
        if file_extension in [".pdf", ".ipynb"]:
            response = extract_data_from_pdf_preprocess(
                file_location=file_location,
                flag_id=flag_id,
                project_id=project_id,
                user_id=user_id,
                file_id=file_id,
                file_extension=file_extension,
                image_uploaded=image_uploaded,
            )
        else:
            celery_app.send_task(
                "extract_text_task",
                (file_location, project_id, flag_id, user_id, None),
            )

            response = {
                "message": "Text extraction task is being processed in the background",
                "flag_id": flag_id,
            }
    except HTTPException as e:
        raise e
    except Exception:
        logger.exception(
            f"[flag_id: {flag_id}] Unexpected error processing {file.filename}"
        )
        celery_app.send_task(
            "file_upload_fail_store",
            (
                file_location,
                "documents/failures/main_service",
                os.path.basename(file_location),
            ),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the file.",
        )

    if "extra_metadata" in response:
        metadata = response.pop("extra_metadata")
    else:
        metadata = {}

    file_details = prepare_file_details(
        flag_id=flag_id,
        project_id=project_id,
        user_id=user_id,
        file_extension=file_extension,
        metadata=metadata,
        title=response.get("title"),
        doi=response.get("DOI"),
        doi_url=response.get("DOI_URL"),
        summary=response.get("abstract"),
        supplementary_id=response.get("supplementary_id"),
    )

    try:
        insert_file_details(file_details)
    except Exception:
        logger.exception(
            f"[flag_id: {flag_id}] Unexpected error inserting file details"
        )
    celery_app.send_task(
        "summarize_pdf_with_gemini_task",
        (
            file_location,
            project_id,
            flag_id,
            user_id,
            None,
            response.get("title"),
            file_id,
        ),
    )

    logger.info(f"Response: {response}")

    return response


@router.post("/get-title-summary/supplementary/", status_code=status.HTTP_200_OK)
def get_title_and_summary_supplementary(
    file: UploadFile,
    flag_id: Annotated[str, Depends(validate_flag_id)],
    project_id: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    image: Annotated[UploadFile | None, File()] = None,
):
    logger.info(
        f"[client: {client}] "
        f"Received file: {file.filename} "
        f"for project_id: {project_id}, flag_id: {flag_id}, user_id: {user_id}"
    )
    flag_id = sanitize_flag_id(flag_id, return_supplimentary_info=False)
    image_uploaded = False
    if not flag_id_exists(flag_id):
        raise FlagIDNotFound()

    supp_id = generate_supp_id()

    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    file_path = f"{supp_id}{file_extension}"
    cache_dir = os.path.join(settings.PDF_CACHE_DIR, f"{flag_id}", "supplementaries")
    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)
    if image:
        image_uploaded = True
        celery_app.send_task(
            "single_image_upload",
            (
                image.file.read(),
                flag_id,
                project_id,
                supp_id,
            ),  # Make sure image.file is read as bytes
        )

    try:
        response = extract_data_from_file_preprocess_sup(
            file_location=file_location,
            flag_id=flag_id,
            supp_id=supp_id,
            project_id=project_id,
            user_id=user_id,
            file_extension=file_extension,
            image_uploaded=image_uploaded,
        )
    except HTTPException as e:
        logger.exception(
            f"[flag_id: {flag_id}] File processing failed for {file.filename}"
        )
        raise e
    except Exception:
        logger.exception(
            f"[flag_id: {flag_id}] Unexpected error processing {file.filename}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the file.",
        )

    file_details = prepare_file_details(
        flag_id=flag_id,
        project_id=project_id,
        user_id=user_id,
        file_extension=file_extension,
        metadata={},
        supplementary_id=supp_id,
    )

    try:
        insert_file_details(file_details)
    except Exception:
        logger.exception(
            f"[flag_id: {response.get('flag_id')}]"
            " Unexpected error inserting file details"
        )
    celery_app.send_task(
        "summarize_pdf_with_gemini_task",
        (file_location, project_id, flag_id, user_id, supp_id, None, None),
    )

    logger.info(f"Response: {response}")

    return response


@router.post(
    "/get-title-summary/webhook/mineru/{flag_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def mineru_webhook(
    flag_id: Annotated[str, Depends(validate_flag_id)],
    request: MineruWebhookRequest,
):
    flag_id, supplimentary_id = sanitize_flag_id(
        flag_id, return_supplimentary_info=True
    )

    file_details = get_file_details(flag_id, supplimentary_id)
    project_id = request.project_id or (
        file_details.project_id if file_details else None
    )
    user_id = request.user_id or (file_details.user_id if file_details else None)

    if (not user_id) or (not project_id):
        raise FlagIDNotFound()

    file_location = request.file_location
    if not (file_location and os.path.exists(file_location)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File location does not exist",
        )

    chunks = request.chunks
    chunks = [x.model_dump() for x in chunks if x.type in TEXT_CHUNK_TYPES and x.value]

    celery_app.send_task(
        "process_text_chunks",
        (
            chunks,
            file_location,
            flag_id,
            supplimentary_id,
            project_id,
            user_id,
            request.file_id,
        ),
    )


@router.post("/retry-fileprocess/", status_code=status.HTTP_200_OK)
def retry_file_process(
    file: UploadFile,
    flag_id: Annotated[str, Depends(validate_flag_id)],
    user_id: Annotated[str, Form()],
    project_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    supplementary_id: Annotated[str | None, Form()] = None,
    file_id: Annotated[str | None, Form()] = None,
):
    logger.debug(
        f"[client: {client}] "
        f"Received file: {file.filename} "
        f"for project_id: {project_id}, flag_id: {flag_id}, user_id: {user_id}"
    )

    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    if supplementary_id:
        file_path = f"{flag_id}_{supplementary_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id, "supplementaries")
    else:
        file_path = f"{flag_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id)

    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)

    celery_app.send_task(
        "process_layout_chunks",
        (
            file_location,
            project_id,
            flag_id,
            user_id,
            supplementary_id,
            file_id,
            file_extension,
        ),
    )
    return {
        "message": "File processing task has been retried successfully.",
        "flag_id": flag_id,
    }


@router.post("/retry-pdf-summary/", status_code=status.HTTP_200_OK)
def retry_pdf_summarization(
    file: UploadFile,
    title: Annotated[str, Form()],
    flag_id: Annotated[str, Depends(validate_flag_id)],
    user_id: Annotated[str, Form()],
    project_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    supplementary_id: Annotated[str | None, Form()] = None,
    file_id: Annotated[str | None, Form()] = None,
):
    logger.debug(
        f"[client: {client}] "
        f"Received file for retry-summary: {file.filename} "
        f"project_id={project_id}, flag_id={flag_id}, user_id={user_id}, title={title}"
    )

    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    if supplementary_id:
        file_path = f"{flag_id}_{supplementary_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id, "supplementaries")
    else:
        file_path = f"{flag_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id)

    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)

    celery_app.send_task(
        "summarize_pdf_with_gemini_task",
        (
            file_location,
            project_id,
            flag_id,
            user_id,
            supplementary_id,
            title,
            file_id,
        ),
    )
    return {
        "message": "PDF summarization task has been retried successfully.",
        "flag_id": flag_id,
    }


@router.post("/pdf-image-retry/", status_code=status.HTTP_200_OK)
def retry_pdf_image_conversion(
    file: UploadFile,
    flag_id: Annotated[str, Depends(validate_flag_id)],
    project_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    supplementary_id: Annotated[str | None, Form()] = None,
    file_id: Annotated[str | None, Form()] = None,
):
    logger.debug(
        f"[client: {client}] "
        f"Received file for pdf-image-retry: {file.filename} "
        f"project_id={project_id}, flag_id={flag_id}"
    )

    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    if supplementary_id:
        file_path = f"{flag_id}_{supplementary_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id, "supplementaries")
    else:
        file_path = f"{flag_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id)

    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)

    celery_app.send_task(
        "convert_pdf_to_image_task",
        (file_location, flag_id, project_id, supplementary_id, file_id),
    )

    return {"message": "Retry PDF-to-image conversion task triggered."}


@router.post("/get-title-summary/retry", status_code=status.HTTP_200_OK)
def retry_get_title_summary(
    file: UploadFile,
    flag_id: Annotated[str, Depends(validate_flag_id)],
    user_id: Annotated[str, Form()],
    project_id: Annotated[str, Form()],
    client: Annotated[S2SSecurityModel, Depends(S2SClient(["get-title-summary"]))],
    supplementary_id: Annotated[str | None, Form()] = None,
    file_id: Annotated[str | None, Form()] = None,
):
    logger.debug(
        f"[client: {client}] "
        f"Received file: {file.filename} "
        f"for project_id: {project_id}, flag_id: {flag_id}, user_id: {user_id}"
    )

    safe_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(safe_filename)[-1].lower()

    if supplementary_id:
        file_path = f"{flag_id}_{supplementary_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id, "supplementaries")
    else:
        file_path = f"{flag_id}{file_extension}"
        cache_dir = os.path.join(settings.PDF_CACHE_DIR, flag_id)

    file_location = secure_file_path(cache_dir, file_path)

    with open(file_location, "wb") as file_object:
        while chunk := file.file.read(project_settings.FILE_UPLOAD_CHUNK_SIZE):
            file_object.write(chunk)

    celery_app.send_task(
        "process_layout_chunks_retry",
        (
            file_location,
            project_id,
            flag_id,
            user_id,
            supplementary_id,
            file_id,
            file_extension,
        ),
    )
    celery_app.send_task(
        "convert_pdf_to_image_task",
        (file_location, flag_id, project_id, supplementary_id, file_id),
    )
    return {
        "message": "File processing task has been retried successfully.",
        "flag_id": flag_id,
    }
