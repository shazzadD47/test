import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import DuplicateColumn
from psycopg_pool import AsyncConnectionPool

from app.logging import logger
from app.v3.endpoints.agent_chat.configs import settings as chat_settings

_connection_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_pool_lock = asyncio.Lock()
_checkpointer_lock = asyncio.Lock()


async def get_shared_connection_pool() -> AsyncConnectionPool:
    """Get or create a shared connection pool for the application."""
    global _connection_pool

    if _connection_pool is None:
        async with _pool_lock:
            if _connection_pool is None:
                max_retries = 3
                retry_delay = 5  # seconds

                for attempt in range(max_retries):
                    try:
                        _connection_pool = AsyncConnectionPool(
                            conninfo=chat_settings.DB_URL,
                            min_size=1,
                            max_size=15,
                            kwargs={
                                "autocommit": True,
                                "prepare_threshold": 0,
                                "keepalives": 1,
                                "keepalives_idle": 30,
                                "keepalives_interval": 10,
                                "keepalives_count": 3,
                            },
                        )
                        logger.info("Created shared connection pool for checkpointer")
                        break
                    except Exception as e:
                        logger.warning(
                            "Failed to create connection pool "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(
                                "Failed to create connection pool after all retries"
                            )
                            raise

    return _connection_pool


async def get_shared_checkpointer() -> AsyncPostgresSaver:
    """Get or create a shared AsyncPostgresSaver instance for the application."""
    global _checkpointer

    if _checkpointer is None:
        async with _checkpointer_lock:
            # Double-check pattern to avoid race conditions
            if _checkpointer is None:
                pool = await get_shared_connection_pool()
                _checkpointer = AsyncPostgresSaver(pool)
                logger.info("Created shared checkpointer instance")

    return _checkpointer


async def setup_checkpointer():
    checkpointer = await get_shared_checkpointer()

    try:
        await checkpointer.setup()
        logger.info("Checkpointer setup completed successfully")
    except DuplicateColumn:
        logger.info("Checkpointer tables already exist")
        pass
    except Exception as e:
        logger.exception(f"Error setting up checkpointer: {e}")
        raise


async def cleanup_connection_pool():
    """Clean up the shared connection pool. Call this during application shutdown."""
    global _connection_pool, _checkpointer

    if _connection_pool is not None:
        try:
            _checkpointer = None
            await _connection_pool.close(timeout=10.0)
            logger.info("Shared connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing connection pool: {e}")
        finally:
            _connection_pool = None
