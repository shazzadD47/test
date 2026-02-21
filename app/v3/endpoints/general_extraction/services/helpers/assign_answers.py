from typing import Any

from pydantic import BaseModel

from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    check_if_null,
    check_if_numerical_label,
    find_matching_unit_labels,
    is_best_numerical_match_for_unit_label,
)


def _assign_answers_to_numerical_labels(
    table_structure: list[dict[str, Any]],
    final_answers: list[dict[str, Any]] | list[str],
    inputs_to_label_map: dict[int, str],
    all_schemas: list[BaseModel] | None = None,
) -> list[dict[str, Any]]:
    """
    Assigns answers to numerical labels in the table structure.

    Args:
        table_structure: List of label dictionaries containing label metadata
        final_answers: List of extracted answers from LLM
        inputs_to_label_map: Maps input indices to label names
        all_schemas: List of Pydantic schemas for structured output

    Returns:
        Updated table structure with assigned answers
    """
    # Create a hash map for quick label lookup by name
    table_structure_name_hash = {label["name"]: label for label in table_structure}
    numerical_labels = []

    # Ensure we have schemas for all answers, defaulting to None if not provided
    if not all_schemas:
        all_schemas = [None] * len(final_answers)

    # Process each answer and assign it to the corresponding numerical label
    for idx, (answer, schema) in enumerate(zip(final_answers, all_schemas)):
        label_name = inputs_to_label_map[idx]
        label_data = table_structure_name_hash[label_name]

        # Only process numerical labels
        if check_if_numerical_label(label_data):
            numerical_labels.append(label_name)
            key_name = "answers"
            if (
                schema
                and "rows" in schema.model_fields
                and isinstance(answer, dict)
                and "rows" in answer
            ):
                key_name = "rows"

            if ("answers" not in label_data) or (
                "answers" in label_data and not isinstance(label_data["answers"], list)
            ):
                label_data["answers"] = []

            if ("citations" not in label_data) or (
                "citations" in label_data
                and not isinstance(label_data["citations"], list)
            ):
                label_data["citations"] = []

            # Handle structured answers (with schema)
            if schema and isinstance(answer, dict) and key_name in answer:
                # Extract values based on label type (root vs non-root)
                if label_data["c_type"] == "root":
                    # Root labels store single values
                    # add all of the values of the answers to the label data
                    label_data["answers"].extend([v["value"] for v in answer[key_name]])

                else:
                    # Non-root labels store arrays of values
                    # each array is the answer for a single question
                    label_data["answers"].extend(
                        [v["values"] for v in answer[key_name]]
                    )

                label_data["citations"].extend(
                    [v["citations"] for v in answer[key_name]]
                )
            elif check_if_null(answer):
                label_data["answers"].append(None)
                label_data["citations"].append([])
            else:
                label_data["answers"].append(answer)
                label_data["citations"].append([])

            # Handle unit labels if they exist for this numerical label
            # Find matching unit labels with exact-case-first preference.
            # e.g., for "weight", prefers "weight_unit" over "Weight_unit"
            matching_unit_labels = find_matching_unit_labels(
                label_name, list(table_structure_name_hash.keys())
            )
            if matching_unit_labels:

                for unit_label in matching_unit_labels:
                    # Skip if this numerical label is not the best match
                    # for the unit label (e.g. "Weight" should not write to
                    # "weight_unit" when "weight" exists as a numerical label)
                    if not is_best_numerical_match_for_unit_label(
                        label_name, unit_label, table_structure
                    ):
                        continue
                    unit_label_data = table_structure_name_hash[unit_label]
                    key_name = "answers"
                    if (
                        schema
                        and "rows" in schema.model_fields
                        and isinstance(answer, dict)
                        and "rows" in answer
                    ):
                        key_name = "rows"

                    # initialize answers array if it doesn't exist or is null
                    if ("answers" not in unit_label_data) or (
                        "answers" in unit_label_data
                        and not isinstance(unit_label_data["answers"], list)
                    ):
                        unit_label_data["answers"] = []

                    # Extract units from structured answers
                    if schema and isinstance(answer, dict) and key_name in answer:
                        unit_label_data["answers"].extend(
                            [v["unit"] for v in answer[key_name]]
                        )
                    else:
                        # No units available
                        unit_label_data["answers"].append("N/A")

    return table_structure


def assign_answers_to_labels(
    table_structure: list[dict[str, Any]],
    final_answers: list[dict[str, Any]] | list[str],
    inputs_to_label_map: dict[int, str],
    has_roots: bool = False,
    all_schemas: list[BaseModel] | None = None,
) -> list[dict[str, Any]]:
    """
    Assigns extracted answers to their corresponding labels in the table structure.

    This function processes both regular and numerical labels, handling different
    answer formats (structured dict responses vs simple strings) and manages
    the assignment logic based on whether root extraction is enabled.

    Args:
        table_structure: List of label dictionaries containing label metadata
        final_answers: List of extracted answers from LLM (can be dicts or strings)
        inputs_to_label_map: Maps input indices to label names
        has_roots: Whether the extraction includes root labels
        all_schemas: List of Pydantic schemas for structured output validation

    Returns:
        Updated table structure with assigned answers
    """
    # Create a hash map for quick label lookup by name
    table_structure_name_hash = {label["name"]: label for label in table_structure}

    # Ensure we have schemas for all answers, defaulting to None if not provided
    if not all_schemas:
        all_schemas = [None] * len(final_answers)

    # Process each answer and assign it to the corresponding non-numerical label
    for idx, (answer, schema) in enumerate(zip(final_answers, all_schemas)):
        label_name = inputs_to_label_map[idx]
        label_data = table_structure_name_hash[label_name]

        # Skip numerical labels as they are handled separately
        if check_if_numerical_label(label_data):
            continue

        key_name = "answers"
        if (
            schema
            and "rows" in schema.model_fields
            and isinstance(answer, dict)
            and "rows" in answer
        ):
            key_name = "rows"

        if ("answers" not in label_data) or (
            "answers" in label_data and not isinstance(label_data["answers"], list)
        ):
            label_data["answers"] = []

        # Handle structured answers (with schema) that contain an "answers" key
        if schema and isinstance(answer, dict) and key_name in answer:
            label_data["answers"].extend(answer[key_name])

        # Handle string answers when schema is expected (fallback case)
        elif check_if_null(answer):
            label_data["answers"].append("N/A")

        # Handle empty answers with schema
        else:
            label_data["answers"].append(answer)

    # Process numerical labels separately using dedicated function
    table_structure = _assign_answers_to_numerical_labels(
        table_structure=table_structure,
        final_answers=final_answers,
        inputs_to_label_map=inputs_to_label_map,
        all_schemas=all_schemas,
    )

    # For non-root extraction, format multiple answers as numbered lists
    # or flatten single-item lists to individual values
    if not has_roots:
        for label in table_structure:
            if "answers" in label and isinstance(label["answers"], list):
                # Format multiple answers as numbered list
                if len(label["answers"]) > 1:
                    label["answers"] = "\n".join(
                        f"{idx+1}. {answer}"
                        for idx, answer in enumerate(label["answers"])
                    )
                # Single answer becomes the answer itself
                elif len(label["answers"]) == 1:
                    label["answers"] = label["answers"][0]
                # No answers available
                else:
                    label["answers"] = "N/A"

    return table_structure
