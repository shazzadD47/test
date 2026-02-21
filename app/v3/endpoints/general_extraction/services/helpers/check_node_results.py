from typing import Any

from app.utils import check_if_null
from app.v3.endpoints.general_extraction.schemas import AgentState


def check_input_preprocessing_node(state: AgentState) -> bool:
    inputs = state["workflow_input"]
    if (
        "inputs" in inputs
        and inputs["inputs"] is not None
        and len(inputs["inputs"]) > 0
    ):
        for item in inputs["inputs"]:
            for item_data in item["data"]:
                if "identifier" not in item_data:
                    return False

    pdf_processed = "file_details" in state

    # check if generate_labels and table structure match
    table_structure_modified = True
    labels_to_generate = inputs.get("metadata", {}).get("generate_labels", [])
    if labels_to_generate:
        for label in inputs["table_structure"]:
            if label["name"] not in labels_to_generate and label["c_type"] == "root":
                # check if label data is present in extracted data
                # if yes, then it can be in table structure only if
                # the answers key are in the label data. otherwise, no.
                # if not extracted, it should be in table structure
                root_extracted = False
                extracted_data = (
                    inputs.get("metadata", {}).get("extracted_data", {}).get("data")
                )

                if extracted_data:
                    for row in extracted_data:
                        if label["name"] in row and not check_if_null(
                            row[label["name"]]
                        ):
                            root_extracted = True
                            break
                if root_extracted and "answers" not in label:
                    table_structure_modified = False
                    break
            elif label["name"] not in labels_to_generate:
                table_structure_modified = False
                break
            else:
                continue

    return pdf_processed and table_structure_modified


def check_relationship_analyzer_node(table_structure: list[dict[str, Any]]) -> bool:
    root_labels = [label for label in table_structure if label["c_type"] == "root"]
    if len(root_labels) == 0:
        return True

    return all(
        label["c_type"] == "root" or "dependent_on" in label
        for label in table_structure
    )


def check_label_query_generator_node(table_structure: list[dict[str, Any]]) -> bool:
    return all(
        "questions" in label_detail
        for label_detail in table_structure
        if label_detail["c_type"] != "root"
    )


def check_label_context_generator_node(table_structure: list[dict[str, Any]]) -> bool:
    return all("answers" in label_detail for label_detail in table_structure)
