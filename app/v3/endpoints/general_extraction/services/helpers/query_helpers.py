import io
import time

import pandas as pd
from langchain_core.runnables.config import ContextThreadPoolExecutor
from pydantic import BaseModel, Field

from app.utils.llms import batch_invoke_chain_with_retry, invoke_chain_with_retry
from app.v3.endpoints.general_extraction.configs import settings as ge_settings
from app.v3.endpoints.general_extraction.logging import celery_logger as logger
from app.v3.endpoints.general_extraction.services.agents import (
    label_query_generator_agent,
)
from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    roll_root_answers,
)
from app.v3.endpoints.general_extraction.services.helpers.input_helpers import (
    check_if_root_labels_extracted,
    format_label_details,
)


def _select_labels_for_query_generation(
    table_structure: list[dict],
    has_root_labels: bool,
    has_dependent_labels: bool,
    root_labels_extracted: bool,
    generate_labels: list[str] = None,
) -> list[str]:
    if root_labels_extracted and has_dependent_labels:
        selected_labels = [
            label["name"]
            for label in table_structure
            if label["c_type"] != "root" and "dependent_on" in label
        ]
        if generate_labels:
            selected_labels = [
                label for label in selected_labels if label in generate_labels
            ]
        logger.info(
            f"Dependent labels selected for query generation: {selected_labels}"
        )
        return selected_labels
    else:
        if has_root_labels:
            return []
        else:
            selected_labels = [label["name"] for label in table_structure]
            if generate_labels:
                selected_labels = [
                    label for label in selected_labels if label in generate_labels
                ]
            logger.info(f"All labels selected for query generation: {selected_labels}")

    return selected_labels


def _create_batched_root_labels_answers(
    root_labels_answers: list[dict],
) -> list[str]:
    """
    Creates a list of CSV strings, where each string is a CSV
    with headers as column names for up to batch_size rows.

    Args:
        root_labels_answers: A list of dictionaries, where each dictionary
                             contains the root labels and their answers.

    Returns:
        A list of strings, where each string is a CSV for a batch
        of combinations of root labels/answers.
    """
    batched_data = []
    batch_size = ge_settings.ROOT_ANSWERS_BATCH_SIZE

    if not root_labels_answers:
        return batched_data

    for i in range(0, len(root_labels_answers), batch_size):
        comb = root_labels_answers[i : i + batch_size]
        df = pd.DataFrame(comb)
        batched_data.append(df.to_csv(index=False))

    return batched_data


def _generate_query_agent_inputs(
    table_structure: list[dict],
    has_root_labels: bool,
    has_dependent_labels: bool,
    root_labels_extracted: bool,
    selected_labels: list[str],
) -> tuple[list[dict], dict | None]:
    table_structure_name_hash = {label["name"]: label for label in table_structure}
    agent_inputs = []
    if root_labels_extracted and has_dependent_labels:
        logger.info("Generating queries for dependent labels")
        root_label_name = [
            label["name"] for label in table_structure if label["c_type"] == "root"
        ]
        root_label_name = root_label_name[0]
        agent_inputs = []
        agent_input_to_query_map = {}
        count = 0
        for label in table_structure:
            if label["name"] in selected_labels:
                all_root_labels_answers = roll_root_answers(
                    {
                        root_label: table_structure_name_hash[root_label]["answers"]
                        for root_label in label["dependent_on"]
                    }
                )
                batched_root_labels_answers = _create_batched_root_labels_answers(
                    all_root_labels_answers,
                )
                label_details = format_label_details(label)
                for item in batched_root_labels_answers:
                    df = pd.read_csv(io.StringIO(item))
                    total_questions = df.shape[0]
                    agent_inputs.append(
                        {
                            "label_details": label_details,
                            "combination_of_answers_of_root_label": item,
                            "total_questions": total_questions,
                        }
                    )
                    agent_input_to_query_map[count] = label["name"]
                    count += 1
        return agent_inputs, agent_input_to_query_map

    else:
        # if table structure has root labels,
        # only create queries for root labels
        # otherwise create queries for all the labels
        if has_root_labels:
            logger.info("Generating queries for root labels")
        else:
            logger.info("Generating queries for all labels")

        agent_inputs = [
            {
                "label_details": format_label_details(
                    table_structure_name_hash[label],
                    include_keys=["name", "description"],
                )
            }
            for label in selected_labels
        ]
        return agent_inputs, None


def create_query_generation_schema(
    total_questions: int,
) -> BaseModel:
    if total_questions:

        class QueryList(BaseModel):
            queries: list[str] = Field(
                ...,
                description="A list of queries about the label",
                min_length=total_questions,
                max_length=total_questions,
            )

    else:

        class QueryList(BaseModel):
            queries: list[str] = Field(
                ..., description="A list of queries about the label"
            )

    return QueryList


def batch_execute_query_generation_agent(
    agent_inputs: list[dict],
    root_labels_extracted: bool,
    has_dependent_labels: bool,
) -> list[str]:
    if root_labels_extracted and has_dependent_labels:
        query_lengths = [agent_input["total_questions"] for agent_input in agent_inputs]
        agents = [
            label_query_generator_agent(
                generate_for_dependent_labels=True,
                schema=create_query_generation_schema(query_length),
            )
            for query_length in query_lengths
        ]

        max_workers = max(1, min(len(agents), ge_settings.MAX_PARALLEL_LLM_CALLS))
        with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    invoke_chain_with_retry,
                    agent,
                    agent_input,
                )
                for agent, agent_input in zip(agents, agent_inputs)
            ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                results.append(None)
        return results
    else:
        query_agent = label_query_generator_agent()
        return batch_invoke_chain_with_retry(
            query_agent,
            agent_inputs,
            max_retries=ge_settings.MAX_RETRIES,
            config={
                "max_concurrency": min(
                    ge_settings.BATCH_SIZE,
                    ge_settings.MAX_PARALLEL_LLM_CALLS,
                ),
            },
        )


def _generate_queries(
    table_structure: list[dict],
    has_dependent_labels: bool,
    root_labels_extracted: bool,
    selected_labels: list[str],
    agent_inputs: list[dict],
    agent_input_to_query_map: dict | None,
) -> list[str]:
    start_time = time.time()
    table_structure_name_hash = {label["name"]: label for label in table_structure}

    if root_labels_extracted and has_dependent_labels:
        label_queries = batch_execute_query_generation_agent(
            agent_inputs,
            root_labels_extracted,
            has_dependent_labels,
        )
        for i, query in enumerate(label_queries):
            label_name = agent_input_to_query_map[i]
            label_data = table_structure_name_hash[label_name]
            total_questions = agent_inputs[i]["total_questions"]
            try:
                if query:
                    if "questions" not in label_data:
                        label_data["questions"] = query.queries
                    else:
                        label_data["questions"].extend(query.queries)
                else:
                    if "questions" not in label_data:
                        label_data["questions"] = [
                            label_data["description"]
                        ] * total_questions
                    else:
                        label_data["questions"].extend(
                            [label_data["description"]] * total_questions
                        )
            except Exception as e:
                logger.info(f"Failed to generate queries for label: {label_name}")
                logger.info(f"Query: {query}")
                logger.info(f"agent inputs: {agent_inputs[i]}")
                logger.info(f"Error: {e}")
        for label_name in selected_labels:
            label_data = table_structure_name_hash[label_name]
            if "questions" not in label_data:
                label_data["questions"] = []

    else:
        selected_label_queries = batch_execute_query_generation_agent(
            agent_inputs,
            root_labels_extracted,
            has_dependent_labels,
        )
        for i, query in enumerate(selected_label_queries):
            label_name = selected_labels[i]
            label_data = table_structure_name_hash[label_name]
            if query:
                if "questions" not in label_data:
                    label_data["questions"] = query.queries
                else:
                    label_data["questions"].extend(query.queries)
            else:
                if "questions" not in label_data:
                    label_data["questions"] = [label_data["description"]]
                else:
                    label_data["questions"].extend([label_data["description"]])
        for label_name in selected_labels:
            label_data = table_structure_name_hash[label_name]
            if "questions" not in label_data:
                label_data["questions"] = [label_data["description"]]

    end_time = time.time()
    logger.info(
        "Time taken for label query generation:" f" {end_time - start_time} seconds"
    )
    return table_structure


def generate_label_queries(
    table_structure: list[dict],
    has_root_labels: bool,
    generate_labels: list[str] = None,
) -> list[str]:
    has_dependent_labels = False
    for label in table_structure:
        if "dependent_on" in label:
            has_dependent_labels = True
            break
    root_labels_extracted = check_if_root_labels_extracted(table_structure)
    selected_labels = _select_labels_for_query_generation(
        table_structure,
        has_root_labels,
        has_dependent_labels,
        root_labels_extracted,
        generate_labels,
    )
    if len(selected_labels) == 0:
        return table_structure
    else:
        agent_inputs, agent_input_to_query_map = _generate_query_agent_inputs(
            table_structure,
            has_root_labels,
            has_dependent_labels,
            root_labels_extracted,
            selected_labels,
        )
        return _generate_queries(
            table_structure,
            has_dependent_labels,
            root_labels_extracted,
            selected_labels,
            agent_inputs,
            agent_input_to_query_map,
        )
