from alembic import command
from alembic.config import Config

from app.configs import settings
from app.logging import logger


async def run_migrations():
    try:
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", str(settings.DB_URL))

        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.exception(f"Error running migrations: {str(e)}")
