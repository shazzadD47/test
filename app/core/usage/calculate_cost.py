from collections import defaultdict
from typing import Any

from app.core.usage.models import (
    CustomUsageMetadata,
    ModelCostDetails,
)
from app.core.usage.per_model_costs import (
    ANTHROPIC_MODEL_MAPPING,
    GOOGLE_GENAI_MODEL_MAPPING,
    OPENAI_MODEL_MAPPING,
)


def _calculate_modality_cost(
    modality: str,
    token_count: int,
    model_cost_details: dict[str, Any],
    model_name: str,
) -> float:
    """
    Helper function to calculate the cost for a single modality,
    handling tiered pricing for Gemini models.

    Args:
        modality (str): The modality to calculate cost for (e.g., 'text', 'image').
        token_count (int): The number of tokens for this modality.
        model_cost_details (Dict[str, Any]): The cost breakdown for the model.
        model_name (str): The name of the model.

    Returns:
        float: The calculated cost for the modality.
    """
    cost = 0.0
    high_limit_key = "high_limit"
    high_cost_key = f"{modality}_high"

    # Check for tiered pricing, specifically for Gemini models
    if (
        model_name.startswith("gemini")
        and high_cost_key in model_cost_details
        and high_limit_key in model_cost_details
        and token_count > model_cost_details[high_limit_key]
    ):
        high_limit = model_cost_details[high_limit_key]
        upper_limit_tokens = token_count - high_limit

        # Cost for tokens above the high limit
        high_tier_rate = model_cost_details.get(
            high_cost_key,
            model_cost_details.get(modality, model_cost_details.get("text", 0)),
        )
        cost += high_tier_rate * upper_limit_tokens

        # Cost for tokens up to the high limit
        base_rate = model_cost_details.get(modality, model_cost_details.get("text", 0))
        cost += base_rate * high_limit
    else:
        # Standard cost calculation
        rate = model_cost_details.get(modality, model_cost_details.get("text", 0))
        cost += rate * token_count

    return cost


def calculate_cost_for_single_token_type(
    model_name: str,
    model_cost_breakdown: Any,  # Replace with ModelCostDetails
    model_usage: Any,  # Replace with CustomUsageMetadata
    token_key: str,
    token_details_key: str,
    cost_key: str,
    cost_breakdown_key: str,
) -> tuple[float, dict[str, float]]:
    """
    Calculate the cost for a single token type by summing up the costs
    for each modality involved (e.g., text, image).

    Args:
        model_name (str): The name of the model.
        model_cost_breakdown (ModelCostDetails): The cost breakdown for the model.
        model_usage (CustomUsageMetadata): The usage metadata for the model.
        token_key (str): The key for the total token count of a specific type.
        token_details_key (str): The key for the detailed token breakdown by modality.
        cost_key (str): The key for the overall cost per token.
        cost_breakdown_key (str): The key for the detailed cost breakdown by modality.

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing the total cost and a
                                       dictionary with the cost breakdown per modality.
    """
    model_token_usage = getattr(model_usage, token_key, 0)
    model_token_usage_details = getattr(model_usage, token_details_key, {})
    model_cost = getattr(model_cost_breakdown, cost_key, 0)
    model_cost_details = getattr(model_cost_breakdown, cost_breakdown_key, {})

    cost_for_each_modality = defaultdict(float)

    # If detailed usage is provided, it's the source of truth
    if model_token_usage_details:
        total_detailed_tokens = sum(model_token_usage_details.values())

        # If total usage exceeds detailed usage, attribute the rest to 'text'
        if model_token_usage > total_detailed_tokens:
            remaining_tokens = model_token_usage - total_detailed_tokens
            model_token_usage_details["text"] = (
                model_token_usage_details.get("text", 0) + remaining_tokens
            )

        for modality, token_count in model_token_usage_details.items():
            cost_for_each_modality[modality] += _calculate_modality_cost(
                modality, token_count, model_cost_details, model_name
            )

    # Fallback for when only total usage is available
    elif model_token_usage > 0:
        rate = model_cost_details.get("text", model_cost)
        cost_for_each_modality["text"] = rate * model_token_usage

    total_cost = sum(cost_for_each_modality.values())

    return total_cost, dict(cost_for_each_modality)


def calculate_llm_cost(
    model_name: str,
    model_usage: CustomUsageMetadata,
) -> ModelCostDetails:
    """
    Calculate the cost for an LLM. It takes the usage metadata for the model
    and the cost breakdown for the model and calculates the cost for each
    token type. It standardizes the cost metadata into ModelCostDetails object.

    Args:
        model_name (str): The name of the model.
        model_usage (CustomUsageMetadata): The usage metadata for the model.

    Returns:
        ModelCostDetails: The cost breakdown for the model.
    """
    if model_name in GOOGLE_GENAI_MODEL_MAPPING:
        model_cost_breakdown = GOOGLE_GENAI_MODEL_MAPPING[model_name]
    elif model_name in ANTHROPIC_MODEL_MAPPING:
        model_cost_breakdown = ANTHROPIC_MODEL_MAPPING[model_name]
    elif model_name in OPENAI_MODEL_MAPPING:
        model_cost_breakdown = OPENAI_MODEL_MAPPING[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in cost mapping")

    model_cost_info = ModelCostDetails()

    # Process each token type and calculate costs
    token_mappings = [
        ("input_tokens", "input_token_details", "input_cost", "input_cost_details"),
        ("output_tokens", "output_token_details", "output_cost", "output_cost_details"),
        (
            "reasoning_tokens",
            "reasoning_token_details",
            "reasoning_cost",
            "reasoning_cost_details",
        ),
        (
            "cache_read_tokens",
            "cache_read_token_details",
            "cache_read_cost",
            "cache_read_cost_details",
        ),
        (
            "cache_write_tokens",
            "cache_write_token_details",
            "cache_write_cost",
            "cache_write_cost_details",
        ),
    ]
    for token_key, token_details_key, cost_key, cost_details_key in token_mappings:
        (cost, cost_for_each_modality) = calculate_cost_for_single_token_type(
            model_name,
            model_cost_breakdown,
            model_usage,
            token_key,
            token_details_key,
            cost_key,
            cost_details_key,
        )
        prev_cost = getattr(model_cost_info, cost_key)
        updated_cost = prev_cost + cost
        setattr(model_cost_info, cost_key, updated_cost)

        updated_cost_details = getattr(model_cost_info, cost_details_key)
        for k, v in cost_for_each_modality.items():
            if k in updated_cost_details:
                updated_cost_details[k] += v
            else:
                updated_cost_details[k] = v
        setattr(model_cost_info, cost_details_key, updated_cost_details)

    model_cost_info.total_cost = sum(
        [
            model_cost_info.input_cost,
            model_cost_info.output_cost,
            model_cost_info.reasoning_cost,
            model_cost_info.cache_read_cost,
            model_cost_info.cache_write_cost,
        ]
    )
    model_cost_info.usage_metadata = model_usage
    return model_cost_info


def calculate_cost(
    usage_metadata: dict,
) -> dict:
    """
    Calculate the cost for an LLM. It takes the usage metadata for the model
    and the cost breakdown for the model and calculates the cost for each
    token type. It standardizes the cost metadata of each model
    into ModelCostDetails object. Finally, each model and it's cost
    metadata is returned in a dictionary.

    Args:
        usage_metadata (dict): The usage metadata for the model.
    """
    cost_metadata = {
        model_name: calculate_llm_cost(
            model_name=model_name,
            model_usage=model_usage,
        )
        for model_name, model_usage in usage_metadata.items()
    }
    return cost_metadata


def calculate_total_cost(
    cost_metadata: dict,
) -> float:
    """
    Calculate the total cost for all LLMs and RAG models.
    """
    total_cost = 0
    for _, model_cost_metadata in cost_metadata.items():
        total_cost += model_cost_metadata.total_cost
    return total_cost
