import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "claude-3-5-sonnet-20240620": 0.003,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "claude-3-5-sonnet-20240620": 0.015,
}


def _get_anthropic_claude_token_cost(
    prompt_tokens: int, completion_tokens: int, model_id: str | None
) -> float:
    """Get the cost of tokens for the Claude model."""
    if model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model_id}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    return (prompt_tokens / 1000) * MODEL_COST_PER_1K_INPUT_TOKENS[model_id] + (
        completion_tokens / 1000
    ) * MODEL_COST_PER_1K_OUTPUT_TOKENS[model_id]


class AnthropicTokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks anthropic info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        if response.llm_output is None:
            return None

        if "usage" not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None

        model_id = response.llm_output.get("model")

        if model_id and model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
            return None

        # compute tokens and cost for this request
        token_usage = response.llm_output["usage"]
        completion_tokens = token_usage.get("output_tokens", 0)
        prompt_tokens = token_usage.get("input_tokens", 0)
        total_tokens = completion_tokens + prompt_tokens

        total_cost = _get_anthropic_claude_token_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=model_id,
        )

        # update shared state behind lock
        with self._lock:
            self.total_cost += total_cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "AnthropicTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "AnthropicTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
