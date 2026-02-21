from typing import Any

from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    check_if_numerical_label,
    unroll_root_answers,
)


def check_if_inputs_exist(
    inputs: dict[str, Any],
) -> bool:
    if "inputs" not in inputs or inputs["inputs"] is None or len(inputs["inputs"]) == 0:
        return False
    return True


def create_identifiers_for_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:

    if not check_if_inputs_exist(inputs):
        return inputs

    count_per_input_type = {}

    for item in inputs["inputs"]:
        name = item.get("name")
        if item.get("type") in ["image", "chart", "table", "equation"]:
            for item_data in item.get("data"):
                count_per_input_type[item.get("type")] = (
                    count_per_input_type.get(item.get("type"), 0) + 1
                )
                subimage_name = (
                    f"{item.get('type')}_"
                    f"{count_per_input_type[item.get('type')]}_"
                    f"name_[{name}]"
                )
                item_data["identifier"] = subimage_name
        elif item.get("type") == "text":
            count_per_input_type[item.get("type")] = (
                count_per_input_type.get(item.get("type"), 0) + 1
            )
            item["identifier"] = (
                f"text_" f"{count_per_input_type[item.get('type')]}_" f"name_[{name}]"
            )

    return inputs


def format_label_details(
    label_details: dict[str, Any],
    exclude_keys: list[str] = None,
    include_keys: list[str] = None,
) -> str:
    if exclude_keys is None:
        exclude_keys = ["c_type"]

    joined_label_details = []
    for key, value in label_details.items():
        if (include_keys and key not in include_keys) or (key in exclude_keys):
            continue
        joined_label_details.append(f"label_{key}: {value}")
    formatted_label_details = "\n".join(joined_label_details)
    return formatted_label_details


def return_root_label_names(
    table_structure: list[dict[str, Any]],
) -> list[str]:
    """
    Return the root label names.
    """
    return [label["name"] for label in table_structure if label["c_type"] == "root"]


def check_if_root_labels_extracted(
    table_structure: list[dict[str, Any]],
) -> bool:
    """
    Check if all the root labels have been extracted.
    """
    if table_structure:
        return not any(
            label["c_type"] == "root" and "answers" not in label
            for label in table_structure
        )
    return True


def check_if_all_labels_extracted(
    table_structure: list[dict[str, Any]],
) -> bool:
    if table_structure:
        return all("answers" in label for label in table_structure)
    return True


def check_if_generate_labels_extracted(
    table_structure: list[dict[str, Any]], generate_labels: list[str] = None
) -> bool:
    """
    Check if all the generate labels have been extracted.
    """
    if generate_labels:
        for label in table_structure:
            if label["name"] not in generate_labels:
                continue
            else:
                if "answers" not in label:
                    return False
    return True


def check_if_numerical_labels_extracted(
    table_structure: list[dict[str, Any]],
) -> bool:
    if table_structure:
        numerical_labels = [
            label for label in table_structure if check_if_numerical_label(label)
        ]
        if len(numerical_labels) == 0:
            return True
        return all("answers" in label for label in numerical_labels)
    return True


def find_unextracted_labels(
    table_structure: list[dict[str, Any]], labels_to_extract: list[str]
) -> list[str]:
    return [
        label["name"]
        for label in table_structure
        if label["name"] in labels_to_extract and "answers" not in label
    ]


def modify_table_structure_exact_labels(
    table_structure: list[dict],
    generate_labels: list[str] = None,
    extracted_data: dict = None,
):
    """
    Modify the table structure to include the labels to generate.
    If the label is not in the generate labels, is a root label,
    and not extracted, add that to generate labels and
    keep in the table structure.
    If is in extracted data, add the answers to the label and add
    to the table structure.
    All other labels should be removed from the table structure.
    """
    if generate_labels:
        root_labels = [
            label["name"] for label in table_structure if label["c_type"] == "root"
        ]

        updated_table_structure = []
        for label in table_structure:
            if label["name"] not in generate_labels and label["c_type"] != "root":
                continue

            elif label["name"] not in generate_labels and label["c_type"] == "root":
                root_extracted = False
                if (
                    extracted_data
                    and isinstance(extracted_data, dict)
                    and "data" in extracted_data
                    and isinstance(extracted_data["data"], list)
                    and len(extracted_data["data"]) > 0
                ):
                    for row in extracted_data["data"]:
                        if label["name"] in row:
                            root_extracted = True
                            break

                if root_extracted and len(root_labels) == 1:
                    label["answers"] = [
                        row[label["name"]] for row in extracted_data["data"]
                    ]
                    updated_table_structure.append(label)
                elif not root_extracted:
                    generate_labels.append(label["name"])
                    updated_table_structure.append(label)
            else:
                updated_table_structure.append(label)

        if len(root_labels) > 1:
            root_labels_to_add = [
                label["name"]
                for label in table_structure
                if label["c_type"] == "root" and label["name"] not in generate_labels
            ]
            root_labels_with_answers = []
            if (
                extracted_data
                and isinstance(extracted_data, dict)
                and "data" in extracted_data
                and isinstance(extracted_data["data"], list)
                and len(extracted_data["data"]) > 0
            ):
                for row in extracted_data["data"]:
                    row_answers = {}
                    for label in root_labels_to_add:
                        row_answers[label] = row[label]
                    root_labels_with_answers.append(row_answers)

            unrolled_root_labels_to_add = unroll_root_answers(
                root_labels_to_add, root_labels_with_answers
            )

            for label in table_structure:
                if label["name"] in unrolled_root_labels_to_add:
                    label["answers"] = unrolled_root_labels_to_add[label["name"]]
                    updated_table_structure.append(label)

        return generate_labels, updated_table_structure

    else:
        generate_labels = [label["name"] for label in table_structure]

        return generate_labels, table_structure
