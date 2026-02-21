from kombu import Exchange, Queue

from app.configs import settings

CELERY_QUEUES = [
    Queue(
        settings.CELERY_TASK_QUEUE,
        Exchange(settings.CELERY_TASK_QUEUE),
        routing_key=settings.CELERY_TASK_QUEUE,
    ),
    Queue("cpu-tasks", Exchange("cpu-tasks"), routing_key="cpu-tasks"),
    Queue("dosing-table", Exchange("dosing-table"), routing_key="dosing-table"),
    Queue(
        "general-extraction",
        Exchange("general-extraction"),
        routing_key="general-extraction",
    ),
]
