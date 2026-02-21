import logging
import logging.config
import os
import sys

from app.configs import settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": settings.LOGGING_FORMAT},
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": settings.LOGGING_FORMAT,
        },
    },
    "handlers": {
        "default": {
            "level": settings.LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
        },
        "queue": {
            "level": settings.LOG_LEVEL,
            "formatter": "standard",
            "class": "app.utils.logging.QueueListenerHandler",
            "handlers": ["cfg://handlers.default"],
            "queue": {"class": "queue.Queue"},
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["queue"],
            "propagate": False,
        },
        "delineate": {
            "level": settings.LOG_LEVEL,
            "handlers": ["queue"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
        },
        "uvicorn.access": {
            "level": "WARNING",
            "handlers": ["default"],
        },
        "httpx": {
            "level": "WARNING",
            "handlers": ["default"],
        },
        "langfuse": {
            "level": "ERROR",
            "handlers": ["default"],
        },
        "pika": {
            "level": "WARNING",
            "handlers": ["default"],
        },
    },
}

if os.getenv("ENV") == "production":
    LOGGING_CONFIG["loggers"]["delineate"]["handlers"] = ["queue"]

    LOGGING_CONFIG["loggers"]["gunicorn.error"] = {
        "level": "INFO",
        "handlers": ["queue"],
        "propagate": False,
    }

    LOGGING_CONFIG["loggers"]["gunicorn.access"] = {
        "level": "INFO",
        "handlers": ["queue"],
        "propagate": False,
    }

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("delineate")
