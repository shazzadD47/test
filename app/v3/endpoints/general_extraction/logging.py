from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("general_extraction")
celery_logger = get_task_logger("general_extraction_celery")
