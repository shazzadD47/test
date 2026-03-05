from app.configs import settings

TASK_ROUTES = {
    "extract_text_task": {
        "queue": "cpu-tasks",
    },
    "convert_pdf_to_image_task": {
        "queue": "cpu-tasks",
    },
    "process_single_document_task": {
        "queue": "cpu-tasks",
    },
    "auto_figure_suggestion_task": {"queue": "cpu-tasks"},
    "prepare_dosing_table_no_figure": {"queue": "dosing-table"},
    "prepare_dosing_table_with_figure": {"queue": "dosing-table"},
    "iterative autofill metadata extraction task": {
        "queue": settings.CELERY_TASK_QUEUE
    },
    "general_extraction_task": {"queue": "general-extraction"},
    "extract_covariate_task": {"queue": "general-extraction"},
    "extract_dynamic_dosing_task": {"queue": "general-extraction"},
    "adverse_event_extraction_service": {"queue": "general-extraction"},
    "unit_standardization_task": {"queue": "cpu-tasks"},
    "column_standardization_task": {"queue": f"{settings.CELERY_TASK_QUEUE}"},
    "tag_extraction_task": {"queue": "cpu-tasks"},
}
