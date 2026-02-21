from celery.utils.log import get_task_logger

from app.core.celery.app import celery_app
from app.v2.endpoints.table_extraction.csv_embedding import extract_csv

celery_logger = get_task_logger(__name__)


@celery_app.task(name="extract_csv_task")
def extract_csv_task(csv_file_path, project_id, flag_id, user_id):
    return extract_csv(
        file_location=csv_file_path,
        project_id=project_id,
        flag_id=flag_id,
        user_id=user_id,
    )
