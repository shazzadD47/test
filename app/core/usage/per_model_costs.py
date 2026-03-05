from app.core.usage.models import ModelCostDetails

MILLION = 1000000

OPENAI_MODEL_MAPPING = {
    "gpt-5.2": ModelCostDetails(
        input_cost_details={
            "text": 1.75 / MILLION,
        },
        output_cost_details={
            "text": 14 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.175 / MILLION,
        },
    ),
    "gpt-5": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
        },
        output_cost_details={
            "text": 10 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.125 / MILLION,
        },
    ),
    "gpt-5-mini": ModelCostDetails(
        input_cost_details={
            "text": 0.25 / MILLION,
        },
        output_cost_details={
            "text": 2 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.025 / MILLION,
        },
    ),
    "text-embedding-3-large": ModelCostDetails(
        input_cost_details={
            "text": 0.13 / MILLION,
        },
    ),
    "gpt-4.1": ModelCostDetails(
        input_cost_details={
            "text": 2 / MILLION,
        },
        output_cost_details={
            "text": 8 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.50 / MILLION,
        },
    ),
    "gpt-4.1-mini": ModelCostDetails(
        input_cost_details={
            "text": 0.4 / MILLION,
        },
        output_cost_details={
            "text": 1.6 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.1 / MILLION,
        },
    ),
    "gpt-4o": ModelCostDetails(
        input_cost_details={
            "text": 2.5 / MILLION,
        },
        output_cost_details={
            "text": 10 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.25 / MILLION,
        },
    ),
    "gpt-4o-2024-08-06": ModelCostDetails(
        input_cost_details={
            "text": 2.5 / MILLION,
        },
        output_cost_details={
            "text": 10 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.25 / MILLION,
        },
    ),
    "chatgpt-4o-latest": ModelCostDetails(
        input_cost_details={
            "text": 2.5 / MILLION,
        },
        output_cost_details={
            "text": 10 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.25 / MILLION,
        },
    ),
    "gpt-4o-mini": ModelCostDetails(
        input_cost_details={
            "text": 0.15 / MILLION,
        },
        output_cost_details={
            "text": 0.6 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.075 / MILLION,
        },
    ),
    "gpt-4-turbo": ModelCostDetails(
        input_cost_details={
            "text": 10 / MILLION,
        },
        output_cost_details={
            "text": 30 / MILLION,
        },
    ),
    "gpt-4": ModelCostDetails(
        input_cost_details={
            "text": 30 / MILLION,
        },
        output_cost_details={
            "text": 60 / MILLION,
        },
    ),
    "gpt-3.5-turbo": ModelCostDetails(
        input_cost_details={
            "text": 0.5 / MILLION,
        },
        output_cost_details={
            "text": 1.5 / MILLION,
        },
    ),
    "gpt-3.5-turbo-1106": ModelCostDetails(
        input_cost_details={
            "text": 0.5 / MILLION,
        },
        output_cost_details={
            "text": 1.5 / MILLION,
        },
    ),
    "o3-mini": ModelCostDetails(
        input_cost_details={
            "text": 1.1 / MILLION,
        },
        output_cost_details={
            "text": 4.4 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.55 / MILLION,
        },
    ),
    "o4-mini": ModelCostDetails(
        input_cost_details={
            "text": 1.1 / MILLION,
        },
        output_cost_details={
            "text": 4.4 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.275 / MILLION,
        },
    ),
    "o3": ModelCostDetails(
        input_cost_details={
            "text": 2 / MILLION,
        },
        output_cost_details={
            "text": 8 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.5 / MILLION,
        },
    ),
}


ANTHROPIC_MODEL_MAPPING = {
    "claude-opus-4-6": ModelCostDetails(
        input_cost_details={
            "text": 5 / MILLION,
            "text_high": 10 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 25 / MILLION,
            "text_high": 37.50 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.5 / MILLION,
            "text_high": 1 / MILLION,
            "high_limit": 200000,
        },
        cache_write_cost_details={
            "text": 6.25 / MILLION,
            "text_high": 12.50 / MILLION,
            "high_limit": 200000,
        },
    ),
    "claude-opus-4-5": ModelCostDetails(
        input_cost_details={
            "text": 5 / MILLION,
        },
        output_cost_details={
            "text": 25 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.5 / MILLION,
        },
        cache_write_cost_details={
            "text": 6.25 / MILLION,
            "cache_5m": 6.25 / MILLION,
            "cache_1h": 10 / MILLION,
        },
    ),
    "claude-opus-4-1": ModelCostDetails(
        input_cost_details={
            "text": 15 / MILLION,
        },
        output_cost_details={
            "text": 75 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.5 / MILLION,
        },
        cache_write_cost_details={
            "text": 18.75 / MILLION,
            "cache_5m": 18.75 / MILLION,
            "cache_1h": 30 / MILLION,
        },
    ),
    "claude-sonnet-4-6": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
            "text_high": 6 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 15 / MILLION,
            "text_high": 22.50 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
            "text_high": 0.6 / MILLION,
            "high_limit": 200000,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "text_high": 7.50 / MILLION,
            "high_limit": 200000,
        },
    ),
    "claude-sonnet-4-5": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
            "text_high": 6 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 15 / MILLION,
            "text_high": 22.50 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
            "text_high": 0.6 / MILLION,
            "high_limit": 200000,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "text_high": 7.50 / MILLION,
            "high_limit": 200000,
        },
    ),
    "claude-haiku-4-5": ModelCostDetails(
        input_cost_details={
            "text": 1 / MILLION,
        },
        output_cost_details={
            "text": 5 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.1 / MILLION,
        },
        cache_write_cost_details={
            "text": 1.25 / MILLION,
            "cache_5m": 1.25 / MILLION,
            "cache_1h": 2 / MILLION,
        },
    ),
    "claude-haiku-3-5": ModelCostDetails(
        input_cost_details={
            "text": 0.80 / MILLION,
        },
        output_cost_details={
            "text": 4 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.08 / MILLION,
        },
        cache_write_cost_details={
            "text": 1 / MILLION,
            "cache_5m": 1 / MILLION,
            "cache_1h": 1.6 / MILLION,
        },
    ),
    "claude-sonnet-4-0": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-opus-4-0": ModelCostDetails(
        input_cost_details={
            "text": 15 / MILLION,
        },
        output_cost_details={
            "text": 75 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.5 / MILLION,
        },
        cache_write_cost_details={
            "text": 18.75 / MILLION,
            "cache_5m": 18.75 / MILLION,
            "cache_1h": 30 / MILLION,
        },
    ),
    "claude-sonnet-4-20250514": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-3-7-sonnet-latest": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-3-7-sonnet-20250219": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-3-5-sonnet-20240620": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-3-opus-20240229": ModelCostDetails(
        input_cost_details={
            "text": 15 / MILLION,
        },
        output_cost_details={
            "text": 75 / MILLION,
        },
        cache_read_cost_details={
            "text": 1.5 / MILLION,
        },
        cache_write_cost_details={
            "text": 18.75 / MILLION,
            "cache_5m": 18.75 / MILLION,
            "cache_1h": 30 / MILLION,
        },
    ),
    "claude-3-sonnet-20240229": ModelCostDetails(
        input_cost_details={
            "text": 3 / MILLION,
        },
        output_cost_details={
            "text": 15 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.3 / MILLION,
        },
        cache_write_cost_details={
            "text": 3.75 / MILLION,
            "cache_5m": 3.75 / MILLION,
            "cache_1h": 6 / MILLION,
        },
    ),
    "claude-3-haiku-20240307": ModelCostDetails(
        input_cost_details={
            "text": 0.25 / MILLION,
        },
        output_cost_details={
            "text": 1.25 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.03 / MILLION,
        },
        cache_write_cost_details={
            "text": 0.3 / MILLION,
            "cache_5m": 0.3 / MILLION,
            "cache_1h": 0.5 / MILLION,
        },
    ),
}

GOOGLE_GENAI_MODEL_MAPPING = {
    "gemini-3.1-pro-preview": ModelCostDetails(
        input_cost_details={
            "text": 2 / MILLION,
            "image": 2 / MILLION,
            "video": 2 / MILLION,
            "audio": 2 / MILLION,
            "text_high": 4 / MILLION,
            "image_high": 4 / MILLION,
            "video_high": 4 / MILLION,
            "audio_high": 4 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 12 / MILLION,
            "image": 12 / MILLION,
            "video": 12 / MILLION,
            "audio": 12 / MILLION,
            "text_high": 18 / MILLION,
            "image_high": 18 / MILLION,
            "video_high": 18 / MILLION,
            "audio_high": 18 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.2 / MILLION,
            "image": 0.2 / MILLION,
            "video": 0.2 / MILLION,
            "audio": 0.2 / MILLION,
            "text_high": 0.4 / MILLION,
            "image_high": 0.4 / MILLION,
            "video_high": 0.4 / MILLION,
            "audio_high": 0.4 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-3.1-flash-lite-preview": ModelCostDetails(
        input_cost_details={
            "text": 0.25 / MILLION,
            "image": 0.25 / MILLION,
            "video": 0.25 / MILLION,
            "audio": 0.50 / MILLION,
        },
        output_cost_details={
            "text": 1.5 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.025 / MILLION,
            "image": 0.025 / MILLION,
            "video": 0.025 / MILLION,
            "audio": 0.05 / MILLION,
        },
    ),
    "gemini-3-pro-preview": ModelCostDetails(
        input_cost_details={
            "text": 2 / MILLION,
            "image": 2 / MILLION,
            "video": 2 / MILLION,
            "audio": 2 / MILLION,
            "text_high": 4 / MILLION,
            "image_high": 4 / MILLION,
            "video_high": 4 / MILLION,
            "audio_high": 4 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 12 / MILLION,
            "image": 12 / MILLION,
            "video": 12 / MILLION,
            "audio": 12 / MILLION,
            "text_high": 18 / MILLION,
            "image_high": 18 / MILLION,
            "video_high": 18 / MILLION,
            "audio_high": 18 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.2 / MILLION,
            "image": 0.2 / MILLION,
            "video": 0.2 / MILLION,
            "audio": 0.2 / MILLION,
            "text_high": 0.4 / MILLION,
            "image_high": 0.4 / MILLION,
            "video_high": 0.4 / MILLION,
            "audio_high": 0.4 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-3-flash-preview": ModelCostDetails(
        input_cost_details={
            "text": 0.5 / MILLION,
            "image": 0.5 / MILLION,
            "video": 0.5 / MILLION,
            "audio": 1 / MILLION,
        },
        output_cost_details={
            "text": 3 / MILLION,
        },
    ),
    "gemini-2.5-flash": ModelCostDetails(
        input_cost_details={
            "text": 0.30 / MILLION,
            "image": 0.30 / MILLION,
            "video": 0.30 / MILLION,
            "audio": 1 / MILLION,
        },
        output_cost_details={
            "text": 2.5 / MILLION,
        },
        reasoning_cost_details={
            "text": 2.5 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.03 / MILLION,
            "image": 0.03 / MILLION,
            "video": 0.03 / MILLION,
            "audio": 0.1 / MILLION,
        },
    ),
    "gemini-2.5-flash-lite": ModelCostDetails(
        input_cost_details={
            "text": 0.1 / MILLION,
            "image": 0.1 / MILLION,
            "video": 0.1 / MILLION,
            "audio": 0.3 / MILLION,
        },
        output_cost_details={
            "text": 0.4 / MILLION,
        },
        reasoning_cost_details={
            "text": 0.4 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.01 / MILLION,
            "image": 0.01 / MILLION,
            "video": 0.01 / MILLION,
            "audio": 0.03 / MILLION,
        },
    ),
    "gemini-2.5-flash-preview-05-20": ModelCostDetails(
        input_cost_details={
            "text": 0.15 / MILLION,
            "image": 0.15 / MILLION,
            "video": 0.15 / MILLION,
            "audio": 1 / MILLION,
        },
        output_cost_details={
            "text": 0.6 / MILLION,
        },
        reasoning_cost_details={
            "text": 3.5 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.0375 / MILLION,
            "image": 0.0375 / MILLION,
            "video": 0.0375 / MILLION,
            "audio": 0.25 / MILLION,
        },
    ),
    "gemini-2.5-flash-preview-04-17": ModelCostDetails(
        input_cost_details={
            "text": 0.15 / MILLION,
            "image": 0.15 / MILLION,
            "video": 0.15 / MILLION,
            "audio": 1 / MILLION,
        },
        output_cost_details={
            "text": 0.6 / MILLION,
        },
        reasoning_cost_details={
            "text": 3.5 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.0375 / MILLION,
            "image": 0.0375 / MILLION,
            "video": 0.0375 / MILLION,
            "audio": 0.25 / MILLION,
        },
    ),
    "gemini-2.5-pro": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.25 / MILLION,
            "video": 1.25 / MILLION,
            "audio": 1.25 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        reasoning_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.125 / MILLION,
            "image": 0.125 / MILLION,
            "video": 0.125 / MILLION,
            "audio": 0.125 / MILLION,
            "text_high": 0.25 / MILLION,
            "image_high": 0.25 / MILLION,
            "video_high": 0.25 / MILLION,
            "audio_high": 0.25 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-2.5-pro-preview-06-05": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.25 / MILLION,
            "video": 1.25 / MILLION,
            "audio": 1.25 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        reasoning_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.31 / MILLION,
            "image": 0.31 / MILLION,
            "video": 0.31 / MILLION,
            "audio": 0.31 / MILLION,
            "text_high": 0.625 / MILLION,
            "image_high": 0.625 / MILLION,
            "video_high": 0.625 / MILLION,
            "audio_high": 0.625 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-2.5-pro-preview-05-06": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.25 / MILLION,
            "video": 1.25 / MILLION,
            "audio": 1.25 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        reasoning_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.31 / MILLION,
            "image": 0.31 / MILLION,
            "video": 0.31 / MILLION,
            "audio": 0.31 / MILLION,
            "text_high": 0.625 / MILLION,
            "image_high": 0.625 / MILLION,
            "video_high": 0.625 / MILLION,
            "audio_high": 0.625 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-2.5-pro-preview-03-25": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.25 / MILLION,
            "video": 1.25 / MILLION,
            "audio": 1.25 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        reasoning_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        cache_read_cost_details={
            "text": 0.31 / MILLION,
            "image": 0.31 / MILLION,
            "video": 0.31 / MILLION,
            "audio": 0.31 / MILLION,
            "text_high": 0.625 / MILLION,
            "image_high": 0.625 / MILLION,
            "video_high": 0.625 / MILLION,
            "audio_high": 0.625 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-2.5-pro-exp-03-25": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.25 / MILLION,
            "video": 1.25 / MILLION,
            "audio": 1.25 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 200000,
        },
        output_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
        reasoning_cost_details={
            "text": 10 / MILLION,
            "image": 10 / MILLION,
            "video": 10 / MILLION,
            "audio": 10 / MILLION,
            "text_high": 15 / MILLION,
            "image_high": 15 / MILLION,
            "video_high": 15 / MILLION,
            "audio_high": 15 / MILLION,
            "high_limit": 200000,
        },
    ),
    "gemini-2.0-flash": ModelCostDetails(
        input_cost_details={
            "text": 0.1 / MILLION,
            "image": 0.1 / MILLION,
            "video": 0.1 / MILLION,
            "audio": 0.7 / MILLION,
        },
        output_cost_details={
            "text": 0.4 / MILLION,
            "image": 30 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.025 / MILLION,
            "image": 0.025 / MILLION,
            "video": 0.025 / MILLION,
            "audio": 0.175 / MILLION,
        },
    ),
    "gemini-2.0-flash-001": ModelCostDetails(
        input_cost_details={
            "text": 0.1 / MILLION,
            "image": 0.1 / MILLION,
            "video": 0.1 / MILLION,
            "audio": 0.7 / MILLION,
        },
        output_cost_details={
            "text": 0.4 / MILLION,
            "image": 30 / MILLION,
        },
        cache_read_cost_details={
            "text": 0.025 / MILLION,
            "image": 0.025 / MILLION,
            "video": 0.025 / MILLION,
            "audio": 0.175 / MILLION,
        },
    ),
    "gemini-2.0-flash-lite": ModelCostDetails(
        input_cost_details={
            "text": 0.075 / MILLION,
            "image": 0.075 / MILLION,
            "video": 0.075 / MILLION,
            "audio": 0.075 / MILLION,
        },
        output_cost_details={
            "text": 0.3 / MILLION,
        },
    ),
    "gemini-2.0-flash-lite-001": ModelCostDetails(
        input_cost_details={
            "text": 0.075 / MILLION,
            "image": 0.075 / MILLION,
            "video": 0.075 / MILLION,
            "audio": 0.075 / MILLION,
        },
        output_cost_details={
            "text": 0.3 / MILLION,
        },
    ),
    "gemini-1.5-pro": ModelCostDetails(
        input_cost_details={
            "text": 1.25 / MILLION,
            "image": 1.2 / MILLION,
            "video": 1.2 / MILLION,
            "audio": 1.2 / MILLION,
            "text_high": 2.5 / MILLION,
            "image_high": 2.5 / MILLION,
            "video_high": 2.5 / MILLION,
            "audio_high": 2.5 / MILLION,
            "high_limit": 128000,
        },
        output_cost_details={
            "text": 5 / MILLION,
            "image": 5 / MILLION,
            "video": 5 / MILLION,
            "audio": 5 / MILLION,
            "text_high": 10 / MILLION,
            "image_high": 10 / MILLION,
            "video_high": 10 / MILLION,
            "audio_high": 10 / MILLION,
            "high_limit": 128000,
        },
        cache_read_cost_details={
            "text": 0.3125 / MILLION,
            "image": 0.3125 / MILLION,
            "video": 0.3125 / MILLION,
            "audio": 0.3125 / MILLION,
            "text_high": 0.625 / MILLION,
            "image_high": 0.625 / MILLION,
            "video_high": 0.625 / MILLION,
            "audio_high": 0.625 / MILLION,
            "high_limit": 128000,
        },
    ),
}
