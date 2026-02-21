import os
from typing import Annotated
from uuid import uuid4

from ExtractTable import ExtractTable
from ExtractTable.exceptions import ServiceError
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pandas import DataFrame

from app.configs import settings
from app.core.dependencies import validate_flag_id
from app.logging import logger
from app.v2.endpoints.table_extraction.tasks import celery_app
from utils.file_ops import upload_files_to_storage_by_name

api_csv_embedding_router = APIRouter(tags=["Table Digitization"])


def client():
    """
    Initialize and return an instance of ExtractTable with the provided API key.

    Returns:
        ExtractTable: An instance of the ExtractTable class.
    """
    return ExtractTable(settings.CSV_API_KEY)


@api_csv_embedding_router.post("/process-image-csv-api")
async def process_csv_embedding(
    file: UploadFile,
    project_id: Annotated[str, Form()],
    flag_id: Annotated[str, Depends(validate_flag_id)],
    user_id: Annotated[str, Form()],
):
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PNG files are supported.",
        )

    unique_id = uuid4().hex
    image_path = os.path.join(settings.SAVE_DIR, f"{unique_id}.png")
    csv_path = os.path.join(settings.SAVE_DIR, f"{unique_id}.csv")

    with open(image_path, "wb") as f:
        f.write(file.file.read())

    try:
        usage = client().check_usage()
        used_credits = usage.get("used", 0) + usage.get("queued", 0)
        total_credits = usage.get("credits", 0)
        remaining_credits = total_credits - used_credits

        if remaining_credits < 750:
            logger.warning(
                f"Remaining Extract Table credits: {remaining_credits}."
                " Please add more credits.",
                extra={
                    "remaining_credits": remaining_credits,
                    "total_credits": total_credits,
                    "used_credits": used_credits,
                    "alert_type": "extract_table_credits_low",
                },
            )

        results: list[DataFrame] = client().process_file(
            filepath=image_path, output_format="df"
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {e}",
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No table found in the uploaded image.",
        )

    result = results[0]
    result.to_csv(csv_path, index=False)

    upload_path = f"documents/tables/{flag_id}"
    save_filename = f"{unique_id}.csv"

    upload_files_to_storage_by_name(csv_path, upload_path, save_filename)

    table_url = (
        f"{settings.S3_SPACES_PUBLIC_BASE_URL}/" f"{upload_path}/{save_filename}"
    )

    celery_app.send_task(
        "extract_csv_task",
        (csv_path, project_id, flag_id, user_id),
    )

    return {"results_csv_url": table_url, "use": usage}
