from copy import deepcopy
from typing import Any

from app.core.usage.helpers import (
    ModelCostDetails,
    combine_cost_metadatas,
    correct_cost_metadata,
    is_correct_cost_metadata,
)


def combine_cost_metadatas_of_models(
    cost_metadatas: list[dict],
) -> dict[str, ModelCostDetails]:
    """
    Takes a list of cost metadatas and combine the same model's cost
    metadata by summing up the cost and tokens.

    Args:
        cost_metadatas (list[dict]): The cost metadata for the models.

    Returns:
        dict[str, ModelCostDetails]: The combined cost metadata.
    """
    combined_cost_metadata = deepcopy(cost_metadatas[0])
    combined_cost_metadata = {
        k: correct_cost_metadata(v) for k, v in combined_cost_metadata.items()
    }
    combined_cost_metadata = {
        k: v for k, v in combined_cost_metadata.items() if v is not None
    }
    for cost_metadata in cost_metadatas[1:]:
        for model_name, model_cost_metadata in cost_metadata.items():
            if model_name in combined_cost_metadata:
                corrected_model_cost_metadata = correct_cost_metadata(
                    model_cost_metadata
                )
                if corrected_model_cost_metadata is not None:
                    combined_cost_metadata[model_name] = combine_cost_metadatas(
                        [
                            combined_cost_metadata[model_name],
                            corrected_model_cost_metadata,
                        ]
                    )
            else:
                corrected_model_cost_metadata = correct_cost_metadata(
                    model_cost_metadata
                )
                if corrected_model_cost_metadata is not None:
                    combined_cost_metadata[model_name] = corrected_model_cost_metadata
    return combined_cost_metadata


def combine_with_result_cost_metadata(
    result: Any,
    cost_metadata: dict,
) -> dict:
    """
    Combine the cost metadata with the result's cost metadata if the result
    contains any cost metadata. Otherwise, return the cost metadata unchanged.

    Args:
        result (Any): The result to attach cost metadata to.
        cost_metadata (dict): Dictionary containing cost metadata for models,
            where keys are model names and values are ModelCostDetails objects.

    Returns:
        dict: The result with cost metadata attached in the format:
    """
    combined_cost_metadata = deepcopy(cost_metadata)
    if (
        isinstance(result, dict)
        and "metadata" in result
        and isinstance(result["metadata"], dict)
        and "ai_metadata" in result["metadata"]
        and isinstance(result["metadata"]["ai_metadata"], dict)
        and "cost_metadata" in result["metadata"]["ai_metadata"]
        and isinstance(result["metadata"]["ai_metadata"]["cost_metadata"], dict)
        and "llm_cost_details" in result["metadata"]["ai_metadata"]["cost_metadata"]
        and isinstance(
            result["metadata"]["ai_metadata"]["cost_metadata"]["llm_cost_details"], dict
        )
        and (
            len(result["metadata"]["ai_metadata"]["cost_metadata"]["llm_cost_details"])
            > 0
        )
    ):
        result_cost_metadata = result["metadata"]["ai_metadata"]["cost_metadata"][
            "llm_cost_details"
        ]
        if is_correct_cost_metadata(result_cost_metadata):
            combined_cost_metadata = combine_cost_metadatas_of_models(
                [
                    combined_cost_metadata,
                    deepcopy(result_cost_metadata),
                ]
            )
    return combined_cost_metadata


def return_result_with_cost_metadata(
    result: Any,
    cost_metadata: dict,
) -> dict:
    """
    Return the result with cost metadata attached.

    This function takes a result and cost metadata, and returns the result
    with the cost metadata properly attached to the result's metadata structure.
    If the result already contains cost metadata, it will overwrite the existing
    cost metadata with the new cost metadata.

    Args:
        result (Any): The result to attach cost metadata to.
        cost_metadata (dict): Dictionary containing cost metadata for models,
            where keys are model names and values are ModelCostDetails objects.

    Returns:
        dict: The result with cost metadata attached in the format:
            {
                "result": <original_result>,
                "metadata": {
                    "ai_metadata": {
                        "cost_metadata": <combined_cost_metadata>
                    }
                }
            }
            If cost_metadata is empty, returns the original result unchanged.
    """
    if len(cost_metadata["llm_cost_details"]) == 0:
        return result

    # check if there is any existing cost metadata in the result
    if (
        isinstance(result, dict)
        and "metadata" in result
        and isinstance(result["metadata"], dict)
        and "ai_metadata" in result["metadata"]
        and isinstance(result["metadata"]["ai_metadata"], dict)
    ):
        # check whether the existing usage metadata is in standard format
        result["metadata"]["ai_metadata"]["cost_metadata"] = cost_metadata
    elif (
        isinstance(result, dict)
        and "metadata" in result
        and isinstance(result["metadata"], dict)
    ):
        result["metadata"].update(
            {
                "ai_metadata": {
                    "cost_metadata": cost_metadata,
                }
            }
        )
    else:
        result = {
            "result": result,
            "metadata": {
                "ai_metadata": {
                    "cost_metadata": cost_metadata,
                }
            },
        }
    return result
