import logging
import random
import time
import uuid
from contextlib import contextmanager

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from app.configs import settings


def init_sentry():
    """
    Initialize Sentry with optimized settings for the application.
    Includes performance monitoring, profiling, and relevant integrations.
    """
    if not settings.SENTRY_DSN:
        return

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENV,
        traces_sample_rate=0.2 if settings.ENV == "production" else 1.0,
        profiles_sample_rate=0.1 if settings.ENV == "production" else 0.5,
        enable_tracing=True,
        attach_stacktrace=True,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            HttpxIntegration(),
            RedisIntegration(),
            CeleryIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        send_default_pii=True,
        before_send=before_send,
    )


def before_send(event, hint):
    """
    Filter and customize events before sending to Sentry.
    Can be used to remove sensitive information or ignore certain errors.
    """
    if "exc_info" in hint:
        _, exc_value, _ = hint["exc_info"]
        if hasattr(exc_value, "status_code") and exc_value.status_code == 404:
            return None

        if isinstance(exc_value, (ConnectionResetError, TimeoutError)):
            if random.random() < 0.1:  # nosec B311
                return event
            return None

    return event


def set_user(user_id, username=None, email=None):
    """
    Set user information for Sentry events.

    Args:
        user_id: Unique identifier for the user
        username: Optional username
        email: Optional email
    """
    if sentry_sdk.Hub.current.client:
        sentry_sdk.set_user({"id": user_id, "username": username, "email": email})


def set_tag(key, value):
    """
    Add a tag to all future Sentry events in this scope.
    Tags are searchable in Sentry.

    Args:
        key: Tag key
        value: Tag value
    """
    if sentry_sdk.Hub.current.client:
        sentry_sdk.set_tag(key, value)


@contextmanager
def monitor_performance(operation_name, **tags):
    """
    Context manager to monitor performance of a code block.

    Args:
        operation_name: Name of the operation being monitored
        tags: Additional tags to add to the transaction

    Example:
        with monitor_performance("database_query", query_type="select"):
            result = db.execute_query(...)
    """
    if not sentry_sdk.Hub.current.client:
        yield
        return

    transaction_id = str(uuid.uuid4())

    with sentry_sdk.start_transaction(
        op=operation_name,
        name=f"{operation_name}-{transaction_id}",
    ) as transaction:
        for key, value in tags.items():
            transaction.set_tag(key, value)

        start_time = time.time()
        try:
            yield
        except Exception as e:
            transaction.set_tag("error", str(e))
            transaction.set_tag("error_type", type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            transaction.set_tag("duration_seconds", f"{duration:.3f}")
            transaction.set_data("duration_seconds", duration)
