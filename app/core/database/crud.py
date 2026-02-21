import logging
from enum import Enum
from typing import Any, TypeVar

import psycopg2
import psycopg2.errors
from psycopg2.extensions import TransactionRollbackError
from sqlalchemy.exc import (
    DatabaseError,
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.sql import Select
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.database.base import get_db_session
from app.core.database.models import BaseModel
from app.logging import logger


class ResultType(Enum):
    """Enum for query result types"""

    SCALAR = "scalar"  # Single value
    SCALAR_ONE = "scalar_one"  # First scalar result
    SCALAR_ALL = "scalar_all"  # All scalar results
    ROW_ONE = "row_one"  # First row result
    ROW_ALL = "row_all"  # All row results


T = TypeVar("T")


def configure_retry(max_retries=3, min_wait=1, max_wait=10):
    """
    Configure a retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds

    Returns:
        A configured retry decorator
    """
    return retry(
        retry=retry_if_exception_type(
            (
                # SQLAlchemy exceptions
                OperationalError,
                SQLAlchemyError,
                DatabaseError,
                DisconnectionError,
                # Direct psycopg2 exceptions
                psycopg2.OperationalError,
                psycopg2.DatabaseError,
                psycopg2.InterfaceError,
                psycopg2.errors.AdminShutdown,
                psycopg2.errors.ConnectionException,
                psycopg2.errors.ConnectionFailure,
                TransactionRollbackError,
            )
        ),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        after=after_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )


def insert_single_with_retry(item, max_retries=3):
    """
    Insert a single database item with retry logic.

    Args:
        item: The database item to insert
        max_retries: Maximum number of retry attempts
    """

    @configure_retry(max_retries=max_retries)
    def _insert():
        with get_db_session() as session:
            session.add(item)
            session.commit()

    _insert()


def insert_batch_with_retry(items, max_retries=3):
    """
    Insert a batch of database items with retry logic.

    Args:
        items: The database items to insert
        max_retries: Maximum number of retry attempts
    """

    @configure_retry(max_retries=max_retries)
    def _insert_batch():
        with get_db_session() as session:
            session.bulk_save_objects(items)
            session.commit()

    _insert_batch()


def select_with_retry(
    query: Any,
    result_type: ResultType = ResultType.SCALAR_ALL,
    max_retries: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
) -> Any | list[Any] | None:
    """
    Execute a database select query with retry logic.

    Args:
        query: The database query to execute
        result_type: How to process the query results:
            - SCALAR: Single scalar value (scalar() method)
            - SCALAR_ONE: First scalar result (scalars().first())
            - SCALAR_ALL: All scalar results (scalars().all())
            - ROW_ONE: First row result (first() method)
            - ROW_ALL: All row results (all() method)
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds

    Returns:
        Query results according to the specified result_type, or None if the query fails
    """
    if not isinstance(query, Select):
        raise ValueError("Query must be a Select object")

    if not isinstance(result_type, ResultType):
        raise ValueError(f"Invalid result type: {result_type}")

    @configure_retry(max_retries=max_retries, min_wait=min_wait, max_wait=max_wait)
    def _execute_query():
        with get_db_session() as session:
            result = session.execute(query)

            if result_type == ResultType.SCALAR:
                return result.scalar()
            elif result_type == ResultType.SCALAR_ONE:
                return result.scalars().first()
            elif result_type == ResultType.SCALAR_ALL:
                return result.scalars().all()
            elif result_type == ResultType.ROW_ONE:
                return result.first()
            elif result_type == ResultType.ROW_ALL:
                return result.all()

    try:
        return _execute_query()
    except (
        OperationalError,
        SQLAlchemyError,
        DatabaseError,
        DisconnectionError,
        psycopg2.OperationalError,
        psycopg2.DatabaseError,
        psycopg2.InterfaceError,
        TransactionRollbackError,
    ) as e:
        logger.exception(f"Database query failed after {max_retries} retries: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during database query: {e}")
        return None


def delete_with_retry(
    table: BaseModel,
    filter_conditions: dict,
    max_retries: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
) -> bool:
    """
    Execute a database delete operation with retry logic.

    Args:
        model_class: The SQLAlchemy model class to delete from
        filter_conditions: Dictionary of column names and values to filter by
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds

    Returns:
        True if deletion was successful, False otherwise
    """
    from sqlalchemy import delete

    @configure_retry(max_retries=max_retries, min_wait=min_wait, max_wait=max_wait)
    def _execute_delete():
        with get_db_session() as session:
            # Build the delete query with filter conditions
            delete_query = delete(table)
            for column_name, value in filter_conditions.items():
                if hasattr(table, column_name):
                    column = getattr(table, column_name)
                    delete_query = delete_query.where(column == value)

            result = session.execute(delete_query)
            session.commit()
            return result.rowcount

    try:
        deleted_count = _execute_delete()
        logger.info(
            f"Successfully deleted {deleted_count} rows from {table.__tablename__}"
        )
        return True
    except (
        OperationalError,
        SQLAlchemyError,
        DatabaseError,
        DisconnectionError,
        psycopg2.OperationalError,
        psycopg2.DatabaseError,
        psycopg2.InterfaceError,
        TransactionRollbackError,
    ) as e:
        logger.exception(f"Database delete failed after {max_retries} retries: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during database delete: {e}")
        return False
