from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("dynamic_dosing")
celery_logger = get_task_logger("dynamic_dosing_celery")
