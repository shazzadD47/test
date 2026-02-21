from pydantic_settings import BaseSettings


class Config(BaseSettings):
    API_TIME_OUT: int = 600  # 10 minutes

    MINERU_SERVICE: str
    AI_MINERU_API_SECRET: str

    LEGEND_LABEL: str = "legend_label"
    LEGEND_AREA: str = "legend_area"
    LEGEND_PATCH: str = "legend_patch"
    AREA_1: str = "AREA: 1"
    PADDING: int = 60
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    INITIAL_THRESHOLD: float = 0.3
    LEGEND_DETECTION_THRESHOLD: float = 0.7
    MINIMUM_THRESHOLD: float = 0.1
    SECOND_CATEGORY_THRESHOLD: float = 0.2

    CLAUDE_MODEL: str = "claude-sonnet-4-0"

    CHART_DETEC_ENDPOINT: str = (
        "https://chartdete.ai.delineate.pro/api/v1/chart-dete/completion"
    )
    MINERU_WEBHOOK_URL: str

    FILE_UPLOAD_CHUNK_SIZE: int = 1024 * 1024  # 1MB

    class Config:
        env_prefix = "PROJECTS_"


settings = Config()
