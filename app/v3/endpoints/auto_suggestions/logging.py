from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("auto_figure_suggestion")
celery_logger = get_task_logger("auto_figure_suggestion")
