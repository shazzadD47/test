import time
from copy import deepcopy

from app.v3.endpoints.general_extraction.logging import celery_logger as logger
from app.v3.endpoints.general_extraction.schemas import AgentState
from app.v3.endpoints.general_extraction.services.helpers import (
    check_if_generate_labels_extracted,
    check_if_root_labels_extracted,
    check_input_preprocessing_node,
    check_label_context_generator_node,
    check_label_query_generator_node,
    check_relationship_analyzer_node,
    create_final_table,
    create_identifiers_for_inputs,
    generate_label_queries,
    get_summarized_contexts,
    modify_table_structure_exact_labels,
    process_pdf_file,
    reassign_unit_label_roots,
)


def input_preprocessing_node(
    state: AgentState,
) -> AgentState:
    """
    The input preprocessing node will take the inputs and preprocess them.
    """
    logger.info("Preprocessing inputs in input_preprocessing_node")

    # If a unit label is assigned as root and has a numerical counterpart,
    # transfer the root to the numerical counterpart instead.
    state["workflow_input"]["table_structure"] = reassign_unit_label_roots(
        state["workflow_input"]["table_structure"]
    )

    state["workflow_input"]["has_root_labels"] = False
    for label in state["workflow_input"]["table_structure"]:
        if label["c_type"] == "root":
            state["workflow_input"]["has_root_labels"] = True
            break

    if "metadata" not in state["workflow_input"]:
        state["workflow_input"]["metadata"] = {}

    generate_labels = state["workflow_input"]["metadata"].get("generate_labels", [])
    if not generate_labels:
        generate_labels = [
            label["name"] for label in state["workflow_input"]["table_structure"]
        ]
        if "generate_labels" not in state["workflow_input"]["metadata"]:
            state["workflow_input"]["metadata"]["generate_labels"] = generate_labels

    if check_input_preprocessing_node(state):
        return {
            "workflow_input": state["workflow_input"],
            "table_structure": state["workflow_input"]["table_structure"],
            "file_details": state["file_details"],
        }

    if (
        "inputs" in state["workflow_input"]
        and state["workflow_input"]["inputs"] is not None
        and len(state["workflow_input"]["inputs"]) > 0
    ):
        inputs_with_identifiers = create_identifiers_for_inputs(state["workflow_input"])
        state["workflow_input"] = inputs_with_identifiers

    if "file_details" not in state:
        state["file_details"] = process_pdf_file(
            flag_id=state["workflow_input"]["flag_id"],
        )

    # modifiy table structure if generate_labels is present
    generate_labels = (
        state["workflow_input"].get("metadata", {}).get("generate_labels", [])
    )
    extracted_data = (
        state["workflow_input"]
        .get("metadata", {})
        .get("extracted_data", {"data": None})
    )
    table_structure = state["workflow_input"]["table_structure"]
    (generate_labels, table_structure) = modify_table_structure_exact_labels(
        table_structure,
        generate_labels,
        extracted_data,
    )
    state["workflow_input"]["table_structure"] = table_structure
    state["workflow_input"]["metadata"]["generate_labels"] = generate_labels

    return {
        "workflow_input": state["workflow_input"],
        "table_structure": state["workflow_input"]["table_structure"],
        "file_details": state["file_details"],
    }


def relationship_analyzer_node(
    state: AgentState,
) -> AgentState:
    if (
        check_relationship_analyzer_node(state["table_structure"])
        or not state["workflow_input"]["has_root_labels"]
        or check_if_generate_labels_extracted(
            state["workflow_input"]["table_structure"],
            state["workflow_input"]["metadata"]["generate_labels"],
        )
    ):
        return {"table_structure": state["table_structure"]}

    table_structure = deepcopy(state["table_structure"])
    root_labels = [label for label in table_structure if label["c_type"] == "root"]
    for label in table_structure:
        if label["c_type"] != "root":
            for root_label in root_labels:
                if "dependent_on" in label:
                    label["dependent_on"].append(root_label["name"])
                else:
                    label["dependent_on"] = [root_label["name"]]

    return {"table_structure": table_structure}


def label_query_generator_node(
    state: AgentState,
) -> AgentState:
    """
    The label query generator node will take the table_structure
    and generate queries for each label using the label_query_generator_agent.
    """
    if check_label_query_generator_node(
        state["table_structure"]
    ) or check_if_generate_labels_extracted(
        state["workflow_input"]["table_structure"],
        state["workflow_input"]["metadata"]["generate_labels"],
    ):
        return {"table_structure": state["table_structure"]}

    table_structure = deepcopy(state["table_structure"])
    generate_labels = (
        state["workflow_input"].get("metadata", {}).get("generate_labels", [])
    )
    table_structure = generate_label_queries(
        table_structure, state["workflow_input"]["has_root_labels"], generate_labels
    )

    return {
        "table_structure": table_structure,
    }


def label_context_generator_node(
    state: AgentState,
) -> AgentState:
    """
    The context generator node will take the table_structure,
    flag_id, project_id and other filters for RAG from the state
    and return the summarized context from the resource for each label
    using the context_generator_agent.
    """
    if check_label_context_generator_node(
        state["table_structure"]
    ) or check_if_generate_labels_extracted(
        state["workflow_input"]["table_structure"],
        state["workflow_input"]["metadata"]["generate_labels"],
    ):
        return {
            "workflow_input": state["workflow_input"],
            "table_structure": state["table_structure"],
            "file_details": state["file_details"],
        }

    start_time = time.time()
    logger.info("Generating contexts for labels in label_context_generator_node")
    inputs = deepcopy(state["workflow_input"])
    inputs["table_structure"] = deepcopy(state["table_structure"])
    table_structure = deepcopy(state["table_structure"])
    table_structure_name_hash = {label["name"]: label for label in table_structure}
    has_dependent_labels = False
    for label in table_structure:
        if "dependent_on" in label:
            has_dependent_labels = True
            break
    generate_labels = (
        state["workflow_input"].get("metadata", {}).get("generate_labels", [])
    )

    # check if creating contexts for root labels
    if check_if_root_labels_extracted(table_structure) and has_dependent_labels:
        logger.info("Generating contexts for dependent labels")

        selected_labels = [
            label["name"]
            for label in table_structure
            if label["c_type"] != "root" and "dependent_on" in label
        ]
        if generate_labels:
            selected_labels = [
                label for label in selected_labels if label in generate_labels
            ]

        # add root label answers to the selected labels
        for label in selected_labels:
            dep_info = []
            for root_label in table_structure_name_hash[label]["dependent_on"]:
                if isinstance(root_label, dict) and "root_label" in root_label:
                    root_label = root_label["root_label"]

                root_label_data = table_structure_name_hash[root_label]
                root_label_answer = "\n".join(
                    [
                        f"{i+1}. {answer}"
                        for i, answer in enumerate(root_label_data["answers"])
                    ]
                )
                dep_info.append(
                    {
                        "root_label": root_label,
                        "root_label_answer": root_label_answer,
                    }
                )
            table_structure_name_hash[label]["dependent_on"] = dep_info

        file_details = state["file_details"]

        table_structure = get_summarized_contexts(
            project_id=state["workflow_input"]["project_id"],
            flag_id=state["workflow_input"]["flag_id"],
            table_structure=table_structure,
            selected_labels=selected_labels,
            inputs=inputs,
            file_details=file_details,
        )
        inputs["table_structure"] = table_structure
        return {
            "workflow_input": inputs,
            "table_structure": table_structure,
            "file_details": file_details,
        }

    else:
        if state["workflow_input"]["has_root_labels"]:
            logger.info("Generating contexts for root labels")
            selected_labels = [
                label["name"] for label in table_structure if label["c_type"] == "root"
            ]
        else:
            logger.info("Generating contexts for all labels")
            selected_labels = [label["name"] for label in table_structure]
        if generate_labels:
            selected_labels = [
                label for label in selected_labels if label in generate_labels
            ]

        file_details = state["file_details"]

        table_structure = get_summarized_contexts(
            project_id=state["workflow_input"]["project_id"],
            flag_id=state["workflow_input"]["flag_id"],
            table_structure=table_structure,
            selected_labels=selected_labels,
            inputs=inputs,
            file_details=file_details,
        )

        inputs["table_structure"] = table_structure

        end_time = time.time()
        logger.info(
            "Time taken for label answer generation:"
            f" {end_time - start_time} seconds"
        )

        return {
            "workflow_input": inputs,
            "table_structure": table_structure,
            "file_details": file_details,
        }


def table_finalization_node(
    state: AgentState,
) -> AgentState:
    """
    The table finalization node will take the table_structure,
    the summarized answers for each field and generate the final table.
    """
    logger.info("Finalizing table in table_finalization_node")
    inputs = deepcopy(state["workflow_input"])
    table_structure = deepcopy(state["table_structure"])
    final_table, final_table_with_citations = create_final_table(
        table_structure=table_structure,
        has_roots=inputs["has_root_labels"],
        inputs=inputs,
    )
    inputs["final_table"] = final_table
    inputs["final_table_with_citations"] = final_table_with_citations
    return {
        "workflow_input": inputs,
    }
