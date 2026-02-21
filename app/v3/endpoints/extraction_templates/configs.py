from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Configs(BaseSettings):
    """Configuration for the extraction templates endpoint."""

    MAIN_AGENT: str = "gemini-2.5-flash"

    class Config:
        """Configuration for the extraction templates endpoint."""

        env_prefix = "EXTRACTION_TEMPLATES_"


settings = Configs()
