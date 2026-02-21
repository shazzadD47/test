import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import DuplicateColumn
from psycopg_pool import AsyncConnectionPool

from app.logging import logger
from app.v3.endpoints.report_generator.configs import settings

_connection_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_pool_lock = asyncio.Lock()
_checkpointer_lock = asyncio.Lock()


async def get_shared_connection_pool() -> AsyncConnectionPool:
    """Get or create a shared connection pool for report generator."""
    global _connection_pool

    if _connection_pool is None:
        async with _pool_lock:
            if _connection_pool is None:
                max_retries = 3
                retry_delay = 5  # seconds

                for attempt in range(max_retries):
                    try:
                        _connection_pool = AsyncConnectionPool(
                            conninfo=settings.DB_URL,
                            min_size=1,
                            max_size=10,
                            timeout=60.0,  # Increase timeout to 60 seconds
                            kwargs={
                                "autocommit": True,
                                "prepare_threshold": 0,
                                "keepalives": 1,
                                "keepalives_idle": 30,
                                "keepalives_interval": 10,
                                "keepalives_count": 3,
                            },
                        )
                        logger.info(
                            "Created shared connection pool for report "
                            "generator checkpointer"
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"Failed to create report generator connection pool "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(
                                "Failed to create report generator connection "
                                "pool after all retries"
                            )
                            raise

    return _connection_pool


async def get_shared_checkpointer() -> AsyncPostgresSaver:
    """Get or create a shared checkpointer for report generator memory."""
    global _checkpointer

    if _checkpointer is None:
        async with _checkpointer_lock:
            if _checkpointer is None:
                pool = await get_shared_connection_pool()
                _checkpointer = AsyncPostgresSaver(pool)
                logger.info("Created shared report generator checkpointer instance")

    return _checkpointer


async def setup_checkpointer():
    """Setup the checkpointer (called during app startup)."""
    checkpointer = await get_shared_checkpointer()

    try:
        await checkpointer.setup()
        logger.info("Report generator checkpointer setup completed successfully")
    except DuplicateColumn:
        logger.info("Report generator checkpointer tables already exist")
    except Exception as e:
        logger.exception(f"Error setting up report generator checkpointer: {e}")
        raise


async def cleanup_connection_pool():
    """Cleanup the connection pool (called during app shutdown)."""
    global _connection_pool, _checkpointer

    if _connection_pool is not None:
        try:
            _checkpointer = None
            await _connection_pool.close(timeout=10.0)
            logger.info("Report generator shared connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing report generator connection pool: {e}")
        finally:
            _connection_pool = None
