from collections import defaultdict

from pydantic import BaseModel


class CustomUsageMetadata(BaseModel):
    input_tokens: int = 0
    """Count of input (or prompt) tokens. Sum of all input token types."""
    output_tokens: int = 0
    """Count of output (or completion) tokens. Sum of all output token types."""
    reasoning_tokens: int = 0
    """Count of reasoning tokens."""
    cache_write_tokens: int = 0
    """Count of cache creation tokens."""
    cache_read_tokens: int = 0
    """Count of cache read tokens."""
    total_tokens: int = 0
    """Total token count. Sum of input_tokens + output_tokens."""
    input_token_details: dict[str, int] = defaultdict(int)
    output_token_details: dict[str, int] = defaultdict(int)
    reasoning_token_details: dict[str, int] = defaultdict(int)
    cache_read_token_details: dict[str, int] = defaultdict(int)
    cache_write_token_details: dict[str, int] = defaultdict(int)


class ModelCostDetails(BaseModel):
    input_cost: float = 0
    output_cost: float = 0
    cache_read_cost: float = 0
    cache_write_cost: float = 0
    reasoning_cost: float = 0
    total_cost: float = 0
    input_cost_details: dict[str, float] = defaultdict(float)
    output_cost_details: dict[str, float] = defaultdict(float)
    cache_read_cost_details: dict[str, float] = defaultdict(float)
    cache_write_cost_details: dict[str, float] = defaultdict(float)
    reasoning_cost_details: dict[str, float] = defaultdict(float)
    usage_metadata: CustomUsageMetadata = CustomUsageMetadata()
