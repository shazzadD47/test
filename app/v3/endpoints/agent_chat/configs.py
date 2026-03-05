from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from app.configs import settings as global_settings

load_dotenv()


class Config(BaseSettings):
    DB_URL: str

    CODE_LLM: str = "claude-sonnet-4-6"
    CHAT_LLM: str = "gpt-4.1"
    CHAT_GEMINI_MODEL: str = "gemini-2.5-flash"
    DEEP_CHAT_FALLBACK_LLM: str = "gpt-4.1"
    REASONING_LLM: str = "claude-sonnet-4-6"

    class Config:
        env_prefix = "AGENT_CHAT_"

    @property
    def FILE_CONTENTS_ENDPOINT(self) -> str:
        """Endpoint template for fetching file contents."""
        base_url = global_settings.NOTEBOOK_AGENT_BASE_URL
        return f"{base_url}/v1/files/{{project_id}}/contents"


settings = Config()
