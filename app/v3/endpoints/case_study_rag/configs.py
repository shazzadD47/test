from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    CASE_STUDY_COLLECTION: str = "casestudyitems"
    LLM_NAME: str = "gpt-4o"

    class Config:
        env_prefix = "RAG_CASE_STUDY_"


settings = Config()
