import gc
import math
from typing import Any

import json_repair
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import ContextThreadPoolExecutor
from pydantic import BaseModel

from app.utils.llms import get_message_text, invoke_llm_with_retry
from app.utils.utils import check_if_null
from app.v3.endpoints.general_extraction.configs import settings as ge_settings
from app.v3.endpoints.general_extraction.logging import celery_logger as logger
from app.v3.endpoints.general_extraction.services.agents import (
    label_context_generator_agent,
)
from app.v3.endpoints.general_extraction.services.helpers.assign_answers import (
    assign_answers_to_labels,
)
from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    check_if_numerical_label,
    find_matching_unit_labels,
)
from app.v3.endpoints.general_extraction.services.helpers.context_chain_helpers import (  # noqa: E501
    create_chain_inputs,
)
from app.v3.endpoints.general_extraction.services.helpers.input_helpers import (
    check_if_numerical_labels_extracted,
    check_if_root_labels_extracted,
)

# Maximum number of LLM calls in a single batch.  When there are many labels
# (e.g. 120+), submitting all calls at once creates massive peak memory
# because each in-flight thread holds a full copy of the serialized message
# (~10-20 MB per call due to Base64 PDF/images).  Processing in smaller
# batches caps peak memory at  BATCH_LLM_CALLS * message_size  while still
# maintaining high throughput via concurrent threads within each batch.
BATCH_LLM_CALLS = 25


def convert_answers_into_dict(
    answers: dict | str | list[str] = None,
    schema: BaseModel = None,
) -> str | dict:
    # if answer is a proper dict, return as is
    if (isinstance(answers, dict) and "answers" in answers) or (
        isinstance(answers, dict) and "rows" in answers
    ):
        return answers

    # if answer is a proper dict, but answers or rows are in the dict,
    # then correct answers dict with either rows or answers as keys
    elif isinstance(answers, dict):
        if schema and "rows" in schema.model_fields and "rows" not in answers:
            answer_values = list(answers.values())
            return {"rows": answer_values}
        else:
            answer_values = list(answers.values())
            return {"answers": answer_values}

    # if schema is provided, try to convert the answers to a proper
    # dict format
    if schema and answers:
        if isinstance(answers, str):
            answer_dict = json_repair.loads(answers)
            if isinstance(answer_dict, str):
                if "rows" in schema.model_fields:
                    return {"rows": [answers]}
                else:
                    return {"answers": [answers]}
            elif isinstance(answer_dict, dict):
                if "answers" in answer_dict:
                    return answer_dict
                elif "rows" in schema.model_fields:
                    return {"rows": list(answer_dict.values())}
                else:
                    return {"answers": list(answer_dict.values())}

            else:
                if "rows" in schema.model_fields:
                    return {"rows": answer_dict}
                else:
                    return {"answers": answer_dict}

        elif isinstance(answers, list):
            if "rows" in schema.model_fields:
                return {"rows": answers}
            else:
                return {"answers": answers}

    elif schema and "rows" in schema.model_fields:
        return {"rows": []}
    elif schema and "answers" in schema.model_fields:
        return {"answers": []}
    elif answers:
        return answers
    else:
        return "N/A"


def batch_execute_context_agent_with_retry(
    messages: list[list[BaseMessage]],
    schemas: list[BaseModel] = None,
    model_name: str = None,
) -> dict[str, Any]:
    total = len(messages)
    max_workers = max(1, min(total, ge_settings.MAX_PARALLEL_LLM_CALLS))

    batch_size = min(total, BATCH_LLM_CALLS)
    num_batches = math.ceil(total / batch_size) if batch_size > 0 else 1
    workers_per_batch = max(1, min(batch_size, max_workers))

    results = [None] * total
    failed_extractions = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_messages = messages[start:end]
        batch_schemas = schemas[start:end] if schemas else None

        if num_batches > 1:
            logger.info(
                f"Processing LLM batch {batch_idx + 1}/{num_batches}"
                f" (labels {start + 1}-{end} of {total})"
            )

        with ContextThreadPoolExecutor(max_workers=workers_per_batch) as executor:
            futures = []
            if batch_schemas:
                for schema, message in zip(batch_schemas, batch_messages):
                    context_agent = label_context_generator_agent(
                        model_name=model_name,
                        schema=schema,
                    )
                    futures.append(
                        executor.submit(
                            invoke_llm_with_retry,
                            context_agent,
                            message,
                            max_retries=ge_settings.MAX_RETRIES,
                        )
                    )
            else:
                context_agent = label_context_generator_agent(
                    model_name=model_name,
                )
                for message in batch_messages:
                    futures.append(
                        executor.submit(
                            invoke_llm_with_retry,
                            context_agent,
                            message,
                            max_retries=ge_settings.MAX_RETRIES,
                        )
                    )

        # Collect results for this batch
        for i, future in enumerate(futures):
            global_idx = start + i
            try:
                results[global_idx] = future.result()
            except Exception as e:
                logger.info(f"Error: {e}")
                results[global_idx] = None
                failed_extractions.append(global_idx)

        # Eagerly free batch references to allow GC to reclaim serialized
        # request/response data before the next batch starts.
        del futures, batch_messages, batch_schemas
        if num_batches > 1:
            gc.collect()

    return results, failed_extractions


def _retrieve_answers_with_fallback(
    model_name: str,
    fallback_model_name: str,
    messages: list[list[BaseMessage]],
    fallback_messages: list,
    schemas: list[BaseModel] = None,
) -> dict[str, Any]:
    try:
        results, failed_indices = batch_execute_context_agent_with_retry(
            model_name=model_name,
            messages=messages,
            schemas=schemas,
        )
    except Exception as e:
        logger.info(f"Answer extraction failed, error: {e}")
        logger.info("Trying again with the fallback llm")
        failed_indices = list(range(len(fallback_messages)))

    logger.info(f"Failed indices: {failed_indices}")
    if len(failed_indices) > 0:
        messages_for_failed_extractions = [fallback_messages[i] for i in failed_indices]
        if schemas:
            schemas_for_failed_extractions = [schemas[i] for i in failed_indices]
        else:
            schemas_for_failed_extractions = None

        results_from_fallback, _ = batch_execute_context_agent_with_retry(
            model_name=fallback_model_name,
            messages=messages_for_failed_extractions,
            schemas=schemas_for_failed_extractions,
        )
        final_results = ["N/A"] * len(schemas)
        fallback_result_count = 0
        for i, schema in enumerate(schemas):
            if i in failed_indices:
                if not check_if_null(results_from_fallback[fallback_result_count]):
                    final_results[i] = results_from_fallback[fallback_result_count]
                else:
                    if schema and "answers" in schema.model_fields:
                        final_results[i] = {
                            "answers": [],
                        }
                    elif schema and "rows" in schema.model_fields:
                        final_results[i] = {
                            "rows": [],
                        }
                    else:
                        final_results[i] = "N/A"
                fallback_result_count += 1
            else:
                if not check_if_null(results[i]):
                    final_results[i] = results[i]
                else:
                    if schema:
                        if "answers" in schema.model_fields:
                            final_results[i] = {"answers": []}
                        elif "rows" in schema.model_fields:
                            final_results[i] = {"rows": []}
                        else:
                            final_results[i] = "N/A"
                    else:
                        final_results[i] = "N/A"
        return final_results
    else:
        return results


def retrieve_answers_from_llm(
    model_name: str,
    fallback_model_name: str,
    messages: list[dict[str, Any]],
    fallback_messages: list[dict[str, Any]],
    schemas: list[BaseModel] | None = None,
) -> list[dict[str, Any]]:
    final_answers = _retrieve_answers_with_fallback(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        messages=messages,
        fallback_messages=fallback_messages,
        schemas=schemas,
    )

    if schemas and isinstance(schemas, list) and len(schemas) == len(final_answers):
        formatted_final_answers = []
        for schema, answer in zip(schemas, final_answers):
            if (isinstance(answer, dict) and "answers" in answer) or (
                isinstance(answer, dict) and "rows" in answer
            ):
                formatted_final_answers.append(answer)
            elif schema and answer and hasattr(answer, "model_dump"):
                formatted_final_answers.append(answer.model_dump())
            elif schema and answer and hasattr(answer, "content"):
                formatted_final_answers.append(
                    convert_answers_into_dict(get_message_text(answer), schema)
                )
            elif schema and isinstance(answer, str):
                formatted_final_answers.append(
                    convert_answers_into_dict(answer, schema)
                )
            elif not schema and answer and hasattr(answer, "content"):
                formatted_final_answers.append(get_message_text(answer))
            elif not schema and answer:
                formatted_final_answers.append(answer)
            elif schema:
                if "answers" in schema.model_fields:
                    formatted_final_answers.append({"answers": []})
                elif "rows" in schema.model_fields:
                    formatted_final_answers.append({"rows": []})
                else:
                    formatted_final_answers.append("N/A")
            else:
                formatted_final_answers.append("N/A")

        return formatted_final_answers

    return final_answers


def get_summarized_contexts(
    project_id: str,
    flag_id: str,
    table_structure: list[dict[str, Any]],
    selected_labels: list[dict[str, Any]] = None,
    inputs: dict[str, Any] = None,
    file_details: dict | None = None,
) -> list[dict[str, Any]]:
    has_roots = inputs["has_root_labels"]
    root_labels_extracted = check_if_root_labels_extracted(table_structure)
    numericals_extracted = check_if_numerical_labels_extracted(table_structure)
    if has_roots and not root_labels_extracted:
        model_name = ge_settings.CONTEXT_GENERATOR_FOR_ROOT_LLM
        fallback_model_name = ge_settings.CONTEXT_GENERATOR_FOR_ROOT_FALLBACK_LLM
    else:
        model_name = ge_settings.CONTEXT_GENERATOR_LLM
        fallback_model_name = ge_settings.CONTEXT_GENERATOR_FALLBACK_LLM

    if model_name.startswith("claude"):
        raise Exception("Claude is not supported yet for extraction.")
    elif fallback_model_name.startswith("claude"):
        raise Exception(
            "Claude is not supported yet for using as fallback for extraction."
        )

    if selected_labels is None:
        selected_labels = [label["name"] for label in table_structure]

    logger.info("Creating inputs for main llm")
    (
        chain_inputs,
        inputs_to_label_map,
        all_schemas,
    ) = create_chain_inputs(
        project_id=project_id,
        flag_id=flag_id,
        table_structure=table_structure,
        selected_labels=selected_labels,
        inputs=inputs,
        file_details=file_details,
        has_roots=has_roots,
        numericals_extracted=numericals_extracted,
    )

    logger.info("Extracting answers for labels")

    final_answers = retrieve_answers_from_llm(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        messages=chain_inputs,
        fallback_messages=chain_inputs,
        schemas=all_schemas,
    )

    del chain_inputs
    gc.collect()

    table_structure = assign_answers_to_labels(
        table_structure=table_structure,
        final_answers=final_answers,
        inputs_to_label_map=inputs_to_label_map,
        has_roots=has_roots,
        all_schemas=all_schemas,
    )

    # Assign empty answers to non-root labels with zero questions
    # (LLM was not invoked for these labels)
    table_structure_hash = {label["name"]: label for label in table_structure}
    for label_name in selected_labels:
        label_data = table_structure_hash[label_name]
        if (
            label_data["c_type"] != "root"
            and len(label_data.get("questions", [])) == 0
            and ("answers" not in label_data or check_if_null(label_data["answers"]))
        ):
            label_data["answers"] = []

            # If this is a numerical label with zero questions, also assign empty
            # answers to its unit label. Unit labels are skipped during context
            # generation and normally get answers through
            # _assign_answers_to_numerical_labels when their numerical counterpart
            # is processed. Since the numerical label was skipped (zero questions),
            # the unit label never received answers.
            if check_if_numerical_label(label_data):
                all_label_names = [label["name"] for label in table_structure]

                # Find matching unit labels with exact-case-first preference.
                # e.g., for "weight", prefers "weight_unit" over "Weight_unit"
                matching_unit_labels = find_matching_unit_labels(
                    label_name, all_label_names
                )

                for unit_label in matching_unit_labels:
                    unit_label_data = table_structure_hash[unit_label]
                    if "answers" not in unit_label_data or check_if_null(
                        unit_label_data["answers"]
                    ):
                        unit_label_data["answers"] = []

    return table_structure
