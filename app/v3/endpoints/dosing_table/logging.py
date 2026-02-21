from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("dosing_table")
celery_logger = get_task_logger("dosing_table_celery")
