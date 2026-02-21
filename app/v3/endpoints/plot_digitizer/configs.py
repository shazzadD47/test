from dotenv import load_dotenv
from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    LLM_NAME: str = "gpt-5.2"
    LEGEND_EXTRACTION_MODEL_NAME: str = "gemini-2.5-flash"
    CLAUDE_MODEL_ID: str = "claude-sonnet-4-0"
    CLAUDE_MAX_TOKENS: int = 64000
    LINEFORMER_API: str
    PD_BASE_ENDPOINT: str
    CHART_DETE_API: str

    @field_validator("PD_BASE_ENDPOINT", "CHART_DETE_API")
    @classmethod
    def remove_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash if present."""
        return v.rstrip("/")

    @computed_field
    @property
    def FLORENCE_2_API(self) -> str:
        endpoint = f"{self.PD_BASE_ENDPOINT}/api/v1/florence_2/florence-2-inference"
        return endpoint.rstrip("/")

    @computed_field
    @property
    def SENTENCE_TRANSFORMER_API(self) -> str:
        endpoint = (
            f"{self.PD_BASE_ENDPOINT}/api/v1/sentence_transformer/sentence_transformer"
        )
        return endpoint.rstrip("/")

    FIX_PARSER_MODEL_NAME: str = "gpt-4.1"
    LEGEND_MAPPIND_MODEL_NAME: str = "gemini-2.5-flash"
    AXIS_MODEL_NAME: str = "gemini-2.5-flash"

    @computed_field
    @property
    def OCR_API(self) -> str:
        endpoint = f"{self.PD_BASE_ENDPOINT}/api/v1/ocr/ocr"
        return endpoint.rstrip("/")

    @computed_field
    @property
    def KAPLAIN_MEIER_API(self) -> str:
        endpoint = f"{self.PD_BASE_ENDPOINT}/api/v1/kaplain_meier_digitizer/kaplain-meier-inference"  # noqa: E501
        return endpoint.rstrip("/")

    @computed_field
    @property
    def SPIDER_PLOT_API(self) -> str:
        endpoint = f"{self.PD_BASE_ENDPOINT}/api/v1/spider_plot_digitizer/spider-plot-inference"  # noqa: E501
        return endpoint.rstrip("/")

    @computed_field
    @property
    def ERROR_BAR_ENDPOINT(self) -> str:
        endpoint = f"{self.PD_BASE_ENDPOINT}/api/v1/error_bar_inference/error-bar-inference"  # noqa: E501
        return endpoint.rstrip("/")

    POINT_GROUP_PIXEL_THRESHOLD: int = 9
    BATCH_SIZE: int = 32
    CONTEXT_QA_BATCH_SIZE: int = 10
    QUERY_REPHRASE_BATCH_SIZE: int = 64
    TOP_K: int = 20
    MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 10

    class Config:
        env_prefix = "PLOT_DIGITIZER_"


settings = Config()


class ErrorBarConfig(BaseSettings):
    GRAY_THRESHOLD: int = 240
    GAUSSIAN_BLUR_KERNEL: tuple[int, int] = (5, 5)
    MORPHOLOGICAL_LENGTH: int = 10
    FILTER_LENGTH: int = 25
    FILTER_HEIGHT: int = 10
    RANGE_X_VALUE: int = 5
    RANGE_Y_HIGH_VALUE: int = 10
    RANGE_Y_LOW_VALUE: int = 5
    THRESHOLD_HIGH_VALUE: int = 10
    THRESHOLD_LOW_VALUE: int = 3
    TOP_BAR_RANGE: int = 35
    BOTTOM_BAR_RANGE: int = 35

    class Config:
        env_prefix = "ERROR_BAR_"


error_bar_config = ErrorBarConfig()
