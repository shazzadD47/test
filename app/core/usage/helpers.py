from copy import deepcopy
from typing import Any

from app.core.usage.models import CustomUsageMetadata, ModelCostDetails
from app.core.usage.per_model_costs import (
    ANTHROPIC_MODEL_MAPPING,
    GOOGLE_GENAI_MODEL_MAPPING,
    OPENAI_MODEL_MAPPING,
)
from app.logging import logger


def convert_single_model_usage_metadata(
    usage_metadata: dict,
) -> CustomUsageMetadata:
    """
    Convert a single model's usage metadata to a CustomUsageMetadata object.
    The usage metadata is returned by langchain's usage tracker callback.
    Other formats are not supported.

    Args:
        usage_metadata (dict): The usage metadata for the model.
    """
    converted_usage_metadata = CustomUsageMetadata(
        input_tokens=usage_metadata["input_tokens"],
        output_tokens=usage_metadata["output_tokens"],
        total_tokens=usage_metadata["total_tokens"],
    )
    if "input_token_details" in usage_metadata:
        input_token_details = usage_metadata["input_token_details"]
        if "audio" in input_token_details:
            converted_usage_metadata.input_token_details["audio"] = input_token_details[
                "audio"
            ]
        elif "cache_creation" in input_token_details:
            converted_usage_metadata.cache_write_tokens = input_token_details[
                "cache_creation"
            ]
        elif "cache_read" in input_token_details:
            converted_usage_metadata.cache_read_tokens = input_token_details[
                "cache_read"
            ]
    if "output_token_details" in usage_metadata:
        output_token_details = usage_metadata["output_token_details"]
        if "audio" in output_token_details:
            converted_usage_metadata.output_token_details["audio"] = (
                output_token_details["audio"]
            )
        elif "reasoning" in output_token_details:
            converted_usage_metadata.reasoning_tokens = output_token_details[
                "reasoning"
            ]
    return converted_usage_metadata


def convert_to_custom_usage_metadata(
    usage_metadata: dict,
) -> CustomUsageMetadata:
    """
    Convert usage metadata for multiple models to a CustomUsageMetadata object.
    The usage metadata is returned by langchain's usage tracker callback.
    Other formats are not supported.

    Args:
        usage_metadata (dict): The usage metadata for the models.
    """
    converted_usage_metadata = {
        model_name: convert_single_model_usage_metadata(model_usage_metadata)
        for model_name, model_usage_metadata in usage_metadata.items()
    }
    return converted_usage_metadata


def find_best_matching_string(
    model_name_to_match: str,
    model_names_to_check: list[str],
) -> str:
    """
    Find the best matching model name from a list of candidate model names.
    It returns the string which matches the largest beginning of the model name
    to match.

    Args:
        model_name_to_match (str): The model name to match.
        model_names_to_check (list[str]): The list of candidate model names.

    Returns:
        str: The best matching model name.
    """
    best_match = ""
    for candidate_model in model_names_to_check:
        if model_name_to_match.startswith(candidate_model) and len(
            candidate_model
        ) > len(best_match):
            best_match = candidate_model
    return best_match


def combine_usage_metadatas(
    usages: list[CustomUsageMetadata],
) -> CustomUsageMetadata:
    """
    Combine usage metadata into a single CustomUsageMetadata object.
    It sums up the token counts in each object and returns a single
    CustomUsageMetadata object.

    Args:
        usages (list[CustomUsageMetadata]): The usage metadata for the models.
    """
    if len(usages) == 0:
        return CustomUsageMetadata()
    if len(usages) == 1:
        return usages[0]

    combined_usage = deepcopy(usages[0])
    for each in usages[1:]:
        if hasattr(each, "input_tokens") and each.input_tokens:
            combined_usage.input_tokens += each.input_tokens
        if hasattr(each, "output_tokens") and each.output_tokens:
            combined_usage.output_tokens += each.output_tokens
        if hasattr(each, "reasoning_tokens") and each.reasoning_tokens:
            combined_usage.reasoning_tokens += each.reasoning_tokens
        if hasattr(each, "cache_write_tokens") and each.cache_write_tokens:
            combined_usage.cache_write_tokens += each.cache_write_tokens
        if hasattr(each, "cache_read_tokens") and each.cache_read_tokens:
            combined_usage.cache_read_tokens += each.cache_read_tokens
        if hasattr(each, "total_tokens") and each.total_tokens:
            combined_usage.total_tokens += each.total_tokens

        if hasattr(each, "input_token_details") and len(each.input_token_details) > 0:
            for key, value in each.input_token_details.items():
                if key not in combined_usage.input_token_details:
                    combined_usage.input_token_details[key] = 0
                combined_usage.input_token_details[key] += value
        if hasattr(each, "output_token_details") and len(each.output_token_details) > 0:
            for key, value in each.output_token_details.items():
                if key not in combined_usage.output_token_details:
                    combined_usage.output_token_details[key] = 0
                combined_usage.output_token_details[key] += value
        if (
            hasattr(each, "cache_write_token_details")
            and len(each.cache_write_token_details) > 0
        ):
            for key, value in each.cache_write_token_details.items():
                if key not in combined_usage.cache_write_token_details:
                    combined_usage.cache_write_token_details[key] = 0
                combined_usage.cache_write_token_details[key] += value
        if (
            hasattr(each, "cache_read_token_details")
            and len(each.cache_read_token_details) > 0
        ):
            for key, value in each.cache_read_token_details.items():
                if key not in combined_usage.cache_read_token_details:
                    combined_usage.cache_read_token_details[key] = 0
                combined_usage.cache_read_token_details[key] += value
    return combined_usage


def format_model_names(
    usage_metadata: dict,
) -> dict:
    """
    Format model names in usage metadata. gemini models are tracked
    as 'models/gemini_model_name' in the usage metadata. Other models
    have their version name attached to them. This function normalizes
    the model names by removing the 'models/' prefix and checking which
    model name matches with the model name with version name attached to it.
    If no match found, then the model name is returned as is.

    Args:
        usage_metadata (dict): The usage metadata for the models.
    """
    updated_usage_metadata = {}
    for model_name in usage_metadata:
        if (
            model_name not in OPENAI_MODEL_MAPPING
            and model_name not in GOOGLE_GENAI_MODEL_MAPPING
            and model_name not in ANTHROPIC_MODEL_MAPPING
        ):
            if model_name.startswith("models/"):
                updated_model_name = model_name.replace("models/", "")
                if updated_model_name in updated_usage_metadata:
                    updated_usage_metadata[updated_model_name] = (
                        combine_usage_metadatas(
                            [
                                updated_usage_metadata[updated_model_name],
                                usage_metadata[model_name],
                            ]
                        )
                    )
                else:
                    updated_usage_metadata[updated_model_name] = usage_metadata[
                        model_name
                    ]
            else:
                best_openai_matched_model = find_best_matching_string(
                    model_name,
                    list(OPENAI_MODEL_MAPPING.keys()),
                )
                best_google_matched_model = find_best_matching_string(
                    model_name,
                    list(GOOGLE_GENAI_MODEL_MAPPING.keys()),
                )
                best_anthropic_matched_model = find_best_matching_string(
                    model_name,
                    list(ANTHROPIC_MODEL_MAPPING.keys()),
                )
                if best_openai_matched_model != "":
                    if best_openai_matched_model not in updated_usage_metadata:
                        updated_usage_metadata[best_openai_matched_model] = (
                            usage_metadata[model_name]
                        )
                    else:
                        updated_usage_metadata[best_openai_matched_model] = (
                            combine_usage_metadatas(
                                [
                                    updated_usage_metadata[best_openai_matched_model],
                                    usage_metadata[model_name],
                                ]
                            )
                        )
                elif best_google_matched_model != "":
                    if best_google_matched_model not in updated_usage_metadata:
                        updated_usage_metadata[best_google_matched_model] = (
                            usage_metadata[model_name]
                        )
                    else:
                        updated_usage_metadata[best_google_matched_model] = (
                            combine_usage_metadatas(
                                [
                                    updated_usage_metadata[best_google_matched_model],
                                    usage_metadata[model_name],
                                ]
                            )
                        )
                elif best_anthropic_matched_model != "":
                    if best_anthropic_matched_model not in updated_usage_metadata:
                        updated_usage_metadata[best_anthropic_matched_model] = (
                            usage_metadata[model_name]
                        )
                    else:
                        updated_usage_metadata[best_anthropic_matched_model] = (
                            combine_usage_metadatas(
                                [
                                    updated_usage_metadata[
                                        best_anthropic_matched_model
                                    ],
                                    usage_metadata[model_name],
                                ]
                            )
                        )
                else:
                    updated_usage_metadata[model_name] = usage_metadata[model_name]

        else:
            updated_usage_metadata[model_name] = usage_metadata[model_name]
    return updated_usage_metadata


def is_correct_cost_metadata(cost_metadata) -> bool:
    """
    Check if the cost metadata is in ModelCostDetails format
    or in a dict format that can be converted to ModelCostDetails.

    Args:
        cost_metadata (Any): The cost metadata to check.
    """
    if not isinstance(cost_metadata, dict):
        return False

    for _, model_cost_details in cost_metadata.items():
        if isinstance(model_cost_details, dict):
            try:
                ModelCostDetails(**model_cost_details)
                continue
            except Exception:
                return False
        elif isinstance(model_cost_details, ModelCostDetails):
            continue
        else:
            logger.error(f"Invalid model cost details: {model_cost_details}")
            return False
    return True


def correct_cost_metadata(
    model_cost_details: Any,
) -> ModelCostDetails:
    """
    Correct the cost metadata. It tries to convert the cost metadata
    to ModelCostDetails format. If failed, it returns None.

    Args:
        model_cost_details (Any): The cost metadata to correct.
    """
    if isinstance(model_cost_details, dict):
        try:
            return ModelCostDetails(**model_cost_details)
        except Exception:
            logger.error(f"Invalid model cost details: {model_cost_details}")
            return None
    elif isinstance(model_cost_details, ModelCostDetails):
        return deepcopy(model_cost_details)
    else:
        logger.error(f"Invalid model cost details: {model_cost_details}")
        return None


def combine_cost_metadatas(costs: list) -> ModelCostDetails:
    """
    Combine cost metadata for multiple models into a single ModelCostDetails object.
    It sums up the cost for each token type in each object and returns a single
    ModelCostDetails object.

    Args:
        costs (list): The cost metadata for the models.
    """
    corrected_costs = [deepcopy(correct_cost_metadata(cost)) for cost in costs]
    corrected_costs = [cost for cost in corrected_costs if cost is not None]
    if len(corrected_costs) == 0:
        return ModelCostDetails()
    elif len(corrected_costs) == 1:
        return corrected_costs[0]

    combined_cost = deepcopy(corrected_costs[0])
    combined_cost.usage_metadata = combine_usage_metadatas(
        [cost.usage_metadata for cost in corrected_costs]
    )

    for cost in corrected_costs[1:]:
        if hasattr(cost, "input_cost") and cost.input_cost:
            combined_cost.input_cost += cost.input_cost
        if hasattr(cost, "output_cost") and cost.output_cost:
            combined_cost.output_cost += cost.output_cost
        if hasattr(cost, "reasoning_cost") and cost.reasoning_cost:
            combined_cost.reasoning_cost += cost.reasoning_cost
        if hasattr(cost, "cache_read_cost") and cost.cache_read_cost:
            combined_cost.cache_read_cost += cost.cache_read_cost
        if hasattr(cost, "cache_write_cost") and cost.cache_write_cost:
            combined_cost.cache_write_cost += cost.cache_write_cost
        if hasattr(cost, "total_cost") and cost.total_cost:
            combined_cost.total_cost += cost.total_cost

        if hasattr(cost, "input_cost_details") and len(cost.input_cost_details) > 0:
            for key, value in cost.input_cost_details.items():
                if key not in combined_cost.input_cost_details:
                    combined_cost.input_cost_details[key] = 0
                combined_cost.input_cost_details[key] += value
        if hasattr(cost, "output_cost_details") and len(cost.output_cost_details) > 0:
            for key, value in cost.output_cost_details.items():
                if key not in combined_cost.output_cost_details:
                    combined_cost.output_cost_details[key] = 0
                combined_cost.output_cost_details[key] += value
        if (
            hasattr(cost, "cache_read_cost_details")
            and len(cost.cache_read_cost_details) > 0
        ):
            for key, value in cost.cache_read_cost_details.items():
                if key not in combined_cost.cache_read_cost_details:
                    combined_cost.cache_read_cost_details[key] = 0
                combined_cost.cache_read_cost_details[key] += value
        if (
            hasattr(cost, "cache_write_cost_details")
            and len(cost.cache_write_cost_details) > 0
        ):
            for key, value in cost.cache_write_cost_details.items():
                if key not in combined_cost.cache_write_cost_details:
                    combined_cost.cache_write_cost_details[key] = 0
                combined_cost.cache_write_cost_details[key] += value
        if (
            hasattr(cost, "reasoning_cost_details")
            and len(cost.reasoning_cost_details) > 0
        ):
            for key, value in cost.reasoning_cost_details.items():
                if key not in combined_cost.reasoning_cost_details:
                    combined_cost.reasoning_cost_details[key] = 0
                combined_cost.reasoning_cost_details[key] += value
    return combined_cost
