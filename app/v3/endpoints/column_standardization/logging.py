from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("column_standardization")
celery_logger = get_task_logger("column_standardization_celery")
