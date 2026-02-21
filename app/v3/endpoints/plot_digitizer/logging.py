from celery.utils.log import get_task_logger

from app.logging import logger

logger = logger.getChild("plot_digitizer")
celery_logger = get_task_logger("plot_digitizer")
