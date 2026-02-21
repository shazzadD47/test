from pathlib import Path

import sentry_sdk
from celery import Celery, signals

from app.configs import settings
from app.core.celery.queues import CELERY_QUEUES
from app.core.celery.task_routes import TASK_ROUTES
from app.core.vector_store import initialize_qdrant
from app.utils.sentry import init_sentry, set_tag

celery_app = Celery(
    "worker",
    broker=str(settings.CELERY_BROKER_URL),
    backend=str(settings.CELERY_RESULT_BACKEND),
)


@signals.celeryd_init.connect
def run_at_celery_init(**kwargs):
    import logging

    logging.getLogger("pika").setLevel(logging.WARNING)

    init_sentry()
    initialize_qdrant()

    Path(settings.PDF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.SAVE_DIR).mkdir(parents=True, exist_ok=True)


@signals.task_prerun.connect
def setup_task_monitoring(task_id, task, *args, **kwargs):
    if sentry_sdk.Hub.current.client:
        set_tag("celery.task_name", task.name)
        set_tag("celery.task_id", task_id)

        sentry_sdk.set_context(
            "celery",
            {
                "task_id": task_id,
                "task_name": task.name,
                "args": repr(args),
                "kwargs": repr(kwargs),
            },
        )


@signals.task_failure.connect
def capture_task_failure(task_id, exception, traceback, einfo, *args, **kwargs):
    if sentry_sdk.Hub.current.client:
        sentry_sdk.capture_exception(exception)


@signals.task_retry.connect
def capture_task_retry(request, reason, einfo, **kwargs):
    if sentry_sdk.Hub.current.client:
        set_tag("celery.retry", True)
        set_tag("celery.retry_reason", str(reason))
        sentry_sdk.add_breadcrumb(
            category="celery", message=f"Task retry: {reason}", level="warning"
        )


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": 7200,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "retry_on_timeout": True,
    },
    worker_prefetch_multiplier=1,
    worker_concurrency=10,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=2 * 3600,
    task_soft_time_limit=1800,
    result_expires=1800,
    task_default_queue=settings.CELERY_TASK_QUEUE,
    task_queues=CELERY_QUEUES,
    task_routes=TASK_ROUTES,
    worker_log_format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
    worker_task_log_format=(
        "%(asctime)s [%(levelname)-8s] [%(name)s] "
        "[%(task_name)s(%(task_id)s)] %(message)s"
    ),
)

celery_app.autodiscover_tasks(
    [
        "app.core.tasks.mineru",
        "app.v2.endpoints.table_extraction.tasks",
        "app.v3.endpoints.get_title_summery.services.tasks",
        "app.v3.endpoints.covariate_extraction.tasks",
        "app.v3.endpoints.iterative_autofill.tasks",
        "app.v3.endpoints.plot_digitizer.services.autofill_tasks",
        "app.v3.endpoints.plot_digitizer.services.merge_tasks",
        "app.v3.endpoints.plot_digitizer.services.digitization_tasks",
        "app.v3.endpoints.dosing_table.services.tasks",
        "app.v3.endpoints.general_extraction.services.tasks",
        "app.v3.endpoints.covariate_extraction.adverse_event",
        "app.v3.endpoints.auto_suggestions.services.tasks",
        "app.v3.endpoints.unit_standardization.services.tasks",
        "app.v3.endpoints.column_standardization.services.tasks",
        "app.v3.endpoints.tag_extraction.tasks",
    ],
    force=True,
)


@celery_app.task(name="sentry-test-celery", queue="cpu-tasks")
def sentry_test_celery():
    return 1 / 0
