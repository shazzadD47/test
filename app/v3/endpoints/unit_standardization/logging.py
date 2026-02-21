from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("unit_standardization")
celery_logger = get_task_logger("unit_standardization_celery")
