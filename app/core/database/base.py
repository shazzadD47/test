import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import MetaData, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.configs import settings
from app.logging import logger

logger = logger.getChild("pg_database")


@retry(
    retry=retry_if_exception_type(SQLAlchemyError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARNING),
)
def create_db_engine():
    """Create engine with retry"""

    return create_engine(
        settings.DB_URL,
        pool_pre_ping=True,
        pool_recycle=600,
        pool_size=10,
        max_overflow=10,
        pool_timeout=15,
        pool_use_lifo=True,
        connect_args={
            "connect_timeout": 15,
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "application_name": f"ai_main_service_{settings.ENV}",
        },
    )


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

metadata = MetaData()
Base = declarative_base(metadata=metadata)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()
