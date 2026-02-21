from copy import deepcopy

from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    roll_root_answers,
    unroll_root_answers,
)


def modify_inputs_for_root_extraction(
    inputs: dict,
) -> dict:
    """
    Modify inputs for root extraction
    """
    root_labels = [
        label["name"]
        for label in inputs["table_structure"]
        if label["c_type"] == "root"
    ]
    updated_table_structure = [
        label for label in inputs["table_structure"] if label["name"] in root_labels
    ]
    updated_inputs = deepcopy(inputs)
    updated_inputs["table_structure"] = updated_table_structure
    if "metadata" not in updated_inputs:
        updated_inputs["metadata"] = {}

    generate_labels = updated_inputs["metadata"].get("generate_labels")
    # if generate labels is not present, then we generate all root labels
    if not generate_labels:
        generate_labels = root_labels

    # check if the labels are present in the extracted data
    # if generate labels was originally not present, then
    # extracted_data is not taken into account. Otherwise,
    # labels not in extracted_data are added to generate_labels
    # only those root labels will be generated which are not present
    # in the extracted data
    extracted_data = updated_inputs["metadata"].get("extracted_data", {}).get("data")
    if extracted_data:
        root_labels_to_generate = []
        for row in extracted_data:
            for label in root_labels:
                if (label not in row) or (generate_labels and label in generate_labels):
                    root_labels_to_generate.append(label)
            break
        updated_inputs["metadata"]["generate_labels"] = root_labels_to_generate
        generate_labels = root_labels_to_generate

    # assign answers to the labels that are already
    # present in the extracted data
    root_label_answers_to_add = [
        label for label in root_labels if label not in generate_labels
    ]
    if len(root_label_answers_to_add) > 0:
        root_label_data = []
        for row in extracted_data:
            row_data = {}
            for label in root_label_answers_to_add:
                if label in row:
                    row_data[label] = row[label]
            root_label_data.append(row_data)

        unrolled_root_labels_to_add = unroll_root_answers(
            root_label_answers_to_add, root_label_data
        )
        for label in updated_inputs["table_structure"]:
            if label["name"] in unrolled_root_labels_to_add:
                label["answers"] = unrolled_root_labels_to_add[label["name"]]

    return updated_inputs


def modify_response_for_non_root_extraction(
    response: dict,
    prev_generate_labels: list[str],
    prev_table_structure: list[dict],
) -> dict:
    """
    Modify response for non root extraction
    Args:
        response: dict: Response from general extraction
    Returns:
        dict: Modified response
    """
    updated_response = _modify_generate_labels(
        deepcopy(response), prev_generate_labels, prev_table_structure
    )

    updated_extracted_data = _modify_extracted_data(
        updated_response, prev_table_structure
    )

    if "metadata" not in updated_response:
        updated_response["metadata"] = {}
    updated_response["metadata"]["extracted_data"] = {"data": updated_extracted_data}

    send_root_labels_first = response.get("metadata", {}).get("send_root_labels_first")
    if send_root_labels_first:
        updated_response["metadata"]["send_root_labels_first"] = False

    updated_response["table_structure"] = prev_table_structure
    return updated_response


def _modify_generate_labels(
    response: dict,
    prev_generate_labels: list[str],
    prev_table_structure: list[dict],
) -> dict:
    """
    Modify generate labels
    Args:
        response: dict: Response from general extraction
        table_structure: list[dict]: Table structure
    Returns:
        dict: Modified response
    """
    # check if generate_labels is already present
    root_labels = [
        label["name"] for label in prev_table_structure if label["c_type"] == "root"
    ]
    updated_generate_labels = []
    if len(prev_generate_labels) > 0:
        for label in prev_generate_labels:
            if label not in root_labels:
                updated_generate_labels.append(label)
    else:
        for label in prev_table_structure:
            if label["c_type"] != "root":
                updated_generate_labels.append(label["name"])
    if "metadata" not in response:
        response["metadata"] = {}
    response["metadata"]["generate_labels"] = updated_generate_labels
    return response


def _modify_extracted_data(
    response: dict,
    table_structure: list[dict],
) -> dict:
    """
    Modify extracted data
    Args:
        response: dict: Response from general extraction
        table_structure: list[dict]: Table structure
    Returns:
        dict: Modified response
    """
    # We need all the root labels to be present in the extracted_data
    # for the next phase of non root extraction. combine both the reextracted
    # root and prev_extracted root data so that all root labels are present
    # in the extracted_data.

    current_extracted_data = response.get(
        "final_table",
        [
            {},
        ],
    )
    prev_extracted_data = (
        response.get("metadata", {})
        .get("extracted_data", {})
        .get(
            "data",
            [
                {},
            ],
        )
    )
    root_labels = [
        label["name"] for label in table_structure if label["c_type"] == "root"
    ]
    root_labels_in_current_extracted_data = [
        label for label in current_extracted_data[0] if label in root_labels
    ]
    root_labels_in_prev_extracted_data = [
        label for label in prev_extracted_data[0] if label in root_labels
    ]

    if len(root_labels_in_current_extracted_data) > 0:
        root_label_answers_in_current_extracted_data = unroll_root_answers(
            root_labels_in_current_extracted_data, current_extracted_data
        )
    else:
        root_label_answers_in_current_extracted_data = {}

    if len(root_labels_in_prev_extracted_data) > 0:
        root_label_answers_in_prev_extracted_data = unroll_root_answers(
            root_labels_in_prev_extracted_data, prev_extracted_data
        )
    else:
        root_label_answers_in_prev_extracted_data = {}

    joined_root_answers = deepcopy(root_label_answers_in_current_extracted_data)
    for label, answers in root_label_answers_in_prev_extracted_data.items():
        if label not in joined_root_answers:
            joined_root_answers[label] = answers

    joined_root_rolled_answers = roll_root_answers(joined_root_answers)
    return joined_root_rolled_answers
