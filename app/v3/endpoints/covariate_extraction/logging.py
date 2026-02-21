from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("covariate_extraction")
celery_logger = get_task_logger("covariate_extraction_celery")
