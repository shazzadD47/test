from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    DB_URL: str

    ASSISTANT_LLM: str = "gemini-2.5-flash"  # Same as agent_chat

    EDIT_LLM: str = "gpt-4.1"  # CHAT_LLM for faster responses

    IMAGE_EDIT_LLM: str = "gemini-2.5-flash"

    REASONING_LLM: str = "claude-sonnet-4-6"

    GEMINI_25_FLASH_INPUT_LIMIT: int = 1000000
    GEMINI_25_FLASH_OUTPUT_LIMIT: int = 65536

    class Config:
        env_prefix = "REPORT_GENERATOR_"


settings = Config()
