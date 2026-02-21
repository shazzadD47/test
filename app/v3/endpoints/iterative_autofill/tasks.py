import time
import traceback
from uuid import uuid4

from celery.utils.log import get_task_logger
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe
from pydantic import BaseModel, Field, create_model

from app.constants import d_type_map
from app.core.celery.app import celery_app
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.utils.tracing import setup_langfuse_handler
from app.utils.utils import check_if_null
from app.v3.endpoints import Status
from app.v3.endpoints.iterative_autofill.chains import (
    table_definition_from_contexts_chain,
)
from app.v3.endpoints.iterative_autofill.configs import settings as iaf_settings
from app.v3.endpoints.iterative_autofill.constants import (
    ITERATIVE_AUTOFILL_ERROR_MESSAGE,
    MAX_RETRIES,
    SEPARATOR,
    TOP_K,
    llm,
    llm_claude,
)
from app.v3.endpoints.iterative_autofill.graph import Graph
from app.v3.endpoints.iterative_autofill.helpers import (
    check_all_labels_extracted,
    combine_contexts,
    extract_all_paper_texts,
    find_next_labels_to_extract,
    get_parent_labels,
    get_relation_info,
    merge_responses,
    prepare_final_response,
)
from app.v3.endpoints.iterative_autofill.langchain_schemas import (
    AnswerModel,
)
from app.v3.endpoints.iterative_autofill.rag_tasks import (
    extract_title,
    format_full_paper_contexts,
    get_context_docs_task,
    get_questions_from_table_structure,
    get_summarized_contexts,
    get_summarized_contexts_from_full_paper,
    select_contexts,
)

logger = get_task_logger("iterative_autofill")


@observe()
def get_label_answers(
    labels: list[str],
    table_structure: list[dict],
    table_structure_hash: dict[str, dict],
    paper_id: str,
    project_id: str,
    level: int = 1,
    prev_response: list[dict] = None,
    has_parents: bool = False,
    full_paper_contexts: str = None,
    total_full_paper_chunks: int = None,
    title: str = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    try:
        if len(labels) == 0:
            return prev_response, labels
        # extract labels that have generate set to true
        # if generate = false, and not in previous response,
        # then extract them too
        if prev_response is not None and len(prev_response) > 0:
            labels_marked_not_to_generate = [
                label for label in labels if not table_structure_hash[label]["generate"]
            ]
            labels_to_extract = []
            for label in labels_marked_not_to_generate:
                if all(label not in response for response in prev_response):
                    labels_to_extract.append(label)
                else:
                    value_found = False
                    for response in prev_response:
                        if label in response and not check_if_null(response[label]):
                            value_found = True
                            break
                    if not value_found:
                        labels_to_extract.append(label)
            labels_to_extract = list(set(labels_to_extract))
            labels_marked_to_generate = [
                label for label in labels if table_structure_hash[label]["generate"]
            ]
            labels = labels_marked_to_generate + labels_to_extract
            labels = list(set(labels))

            # no labels to extract
            if len(labels) == 0:
                return prev_response, labels

        start_time = time.time()
        try:
            questions = get_questions_from_table_structure(
                table_structure_hash,
                labels,
                prev_response,
                langfuse_session_id=langfuse_session_id,
            )
        except Exception as e:
            error_message = "Error occurred while generating questions."
            logger.exception(f"{error_message} {e}")
            error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE + error_message
            raise Exception(error_message)

        logger.info(f"Questions generated in {time.time() - start_time} seconds.")

        # get contexts from rag
        start_time = time.time()
        all_retrieval_questions = []
        for field in table_structure:
            if field["name"] in questions:
                all_retrieval_questions.extend(
                    questions[field["name"]]["retrieval_questions"]
                )

        try:
            contexts = get_context_docs_task(
                queries=all_retrieval_questions,
                flag_id=paper_id,
                project_id=project_id,
                top_k=TOP_K,
            )
        except Exception:
            error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
            error_message += "Error occured when retrieving contexts."
            raise Exception(error_message)

        start, end = 0, 0
        for field_name in questions:
            num_questions = len(questions[field_name]["retrieval_questions"])
            end = start + num_questions
            selected_contexts = select_contexts(contexts[start:end])
            selected_contexts = [
                f"{count + 1}. {context}"
                for count, context in enumerate(selected_contexts)
            ]
            questions[field_name]["context_length"] = len(selected_contexts)
            selected_contexts = SEPARATOR.join(selected_contexts)
            questions[field_name]["contexts"] = selected_contexts
            start = end

        for label in labels:
            if (
                table_structure_hash.get(label, {}).get("relationships", []) is not None
                and len(table_structure_hash.get(label, {}).get("relationships", []))
                > 0
            ):
                questions[label]["parent_label_answers"] = (
                    f"{label}: \n"
                    + get_relation_info(
                        table_structure_hash[label]["relationships"],
                        prev_response,
                        table_structure_hash,
                    )
                )
            else:
                questions[label][
                    "parent_label_answers"
                ] = f"{label}: No parent label answers found."

        logger.info(f"Contexts retrieved in {time.time() - start_time} seconds.")

        # summarize contexts
        try:
            start_time = time.time()
            questions_with_summarized_contexts = get_summarized_contexts(
                questions,
                has_parents=has_parents,
                title=title,
                langfuse_session_id=langfuse_session_id,
            )
        except Exception:
            error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
            error_message += "Error occured when summarizing contexts."
            raise Exception(error_message)

        rag_failed_labels = [
            label
            for label in questions_with_summarized_contexts
            if not questions_with_summarized_contexts[label]["answer_found"]
        ]

        LabelsSchema = None
        parser = None
        if len(rag_failed_labels) > 0:
            labels_schema = {}
            for label in rag_failed_labels:
                labels_schema[label] = create_model(
                    label,
                    answer=(
                        str,
                        Field(
                            ...,
                            description=f"The answer of {label}. Answer must be cited.",
                        ),
                    ),
                    __base__=AnswerModel,
                )
                labels_schema[label] = (
                    labels_schema[label],
                    Field(..., description=table_structure_hash[label]["description"]),
                )
            LabelsSchema = create_model("LabelsSchema", **labels_schema)
            parser = PydanticOutputParser(pydantic_object=LabelsSchema)

        if (
            full_paper_contexts is not None
            and isinstance(full_paper_contexts, str)
            and full_paper_contexts.strip() != ""
            and len(rag_failed_labels) > 0
        ):
            try:
                all_summarized_contexts = get_summarized_contexts_from_full_paper(
                    questions_with_summarized_contexts,
                    labels_to_extract=rag_failed_labels,
                    full_paper=full_paper_contexts,
                    total_full_paper_chunks=total_full_paper_chunks,
                    table_structure_hash=table_structure_hash,
                    parser=parser,
                    output_schema=LabelsSchema,
                    prev_response=prev_response,
                    has_parents=has_parents,
                    title=title,
                    langfuse_session_id=langfuse_session_id,
                )
                questions_with_contexts_from_full_paper = format_full_paper_contexts(
                    full_paper=full_paper_contexts,
                    labels_to_extract=rag_failed_labels,
                    all_summarized_contexts=all_summarized_contexts,
                    title=title,
                    langfuse_session_id=langfuse_session_id,
                )

                for label in rag_failed_labels:
                    questions_with_summarized_contexts[label]["contexts"] = (
                        questions_with_contexts_from_full_paper[label]["contexts"]
                    )
                    questions_with_summarized_contexts[label]["answer_found"] = (
                        questions_with_contexts_from_full_paper[label]["answer_found"]
                    )
            except Exception as e:
                logger.exception(
                    f"Error in get_summarized_contexts_from_full_paper: {e}"
                )

        contexts = combine_contexts(questions_with_summarized_contexts, labels)
        logger.info(f"Contexts summarized in {time.time() - start_time} seconds.")
        # retrieve all table definition values from RAG
        # first extract for integer and float labels
        start_time = time.time()
        integer_float_labels = [
            field["name"]
            for field in table_structure
            if field["name"] in labels and field["d_type"] in ["integer", "float"]
        ]
        logger.info(f"Integer and float labels: {integer_float_labels}")
        other_labels = [
            field["name"]
            for field in table_structure
            if field["name"] not in integer_float_labels
        ]

        labels_for_int_floats = []
        if len(integer_float_labels) > 0:
            # get parent labels for integer and float labels
            parent_labels = []
            for label in integer_float_labels:
                if (
                    table_structure_hash.get(label, {}).get("relationships", [])
                    is not None
                    and len(
                        table_structure_hash.get(label, {}).get("relationships", [])
                    )
                    > 0
                ):
                    for relationship in table_structure_hash[label]["relationships"]:
                        parent_labels.append(relationship["related_label"])

            combined_labels = integer_float_labels + parent_labels
            combined_labels = list(set(combined_labels))

            labels_schema = {
                field["name"]: (
                    d_type_map[field["d_type"]] | None,
                    Field(..., description=field["description"]),
                )
                for field in table_structure
                if field["name"] in combined_labels
            }
            TableLabels = create_model("TableLabels", **labels_schema)

            class TableLabelsSchema(BaseModel):
                data: list[TableLabels]

            parser = PydanticOutputParser(pydantic_object=TableLabelsSchema)
            format_instructions = parser.get_format_instructions()
            table_values_from_contexts_chain_claude = (
                table_definition_from_contexts_chain(
                    llm=llm_claude, output_schema=TableLabelsSchema
                )
            )
            table_values_from_contexts_chain = table_definition_from_contexts_chain(
                llm=llm, output_schema=TableLabelsSchema
            )
            if prev_response is not None and len(prev_response) > 0:
                response_of_parent_labels = [
                    {k: v for k, v in response.items() if k in parent_labels}
                    for response in prev_response
                ]
            else:
                response_of_parent_labels = []

            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    try:
                        labels_for_int_floats = (
                            table_values_from_contexts_chain_claude.invoke(
                                {
                                    "contexts": contexts,
                                    "output_instructions": format_instructions,
                                    "labels_with_answers": response_of_parent_labels,
                                },
                                config={"callbacks": [langfuse_handler]},
                            )
                        )
                    except Exception:
                        labels_for_int_floats = table_values_from_contexts_chain.invoke(
                            {
                                "contexts": contexts,
                                "output_instructions": format_instructions,
                                "labels_with_answers": response_of_parent_labels,
                            },
                            config={"callbacks": [langfuse_handler]},
                        )
                    break
                except Exception as e:
                    retry_count += 1
                    logger.exception(
                        f"An error occurred while retrieving labels from RAG: {e}"
                    )
                    if retry_count == MAX_RETRIES:
                        error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
                        error_message += (
                            "Error occured when retrieving numerical labels from RAG."
                        )
                        raise Exception(error_message)
                    continue

            labels_for_int_floats = labels_for_int_floats.dict()
            try:
                labels_for_int_floats = merge_responses(
                    prev_response, labels_for_int_floats["data"]
                )
            except Exception as e:
                logger.exception(f"Error occured merging numerical labels: {e}")
                error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
                error_message += "Error occured when merging numerical labels."
                raise Exception(error_message)

        labels_for_other_labels = [
            {
                label: questions_with_summarized_contexts[label]["contexts"]
                for label in questions_with_summarized_contexts
                if label not in integer_float_labels
            }
        ]
        try:
            labels_for_other_labels = merge_responses(
                prev_response, labels_for_other_labels
            )
        except Exception as e:
            logger.exception(f"Error occured merging non numerical labels: {e}")
            error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
            error_message += "Error occured when merging non numerical labels."
            raise Exception(error_message)

        if len(integer_float_labels) > 0 and len(other_labels) > 0:
            try:
                labels_from_rag = merge_responses(
                    labels_for_other_labels, labels_for_int_floats
                )
            except Exception as e:
                logger.exception(f"Error occured merging labels: {e}")
                error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
                error_message += "Error occured when merging labels."
                raise Exception(error_message)
        else:
            if len(other_labels) > 0:
                labels_from_rag = labels_for_other_labels
            elif len(integer_float_labels) > 0:
                labels_from_rag = labels_for_int_floats
            else:
                labels_from_rag = prev_response
        taken_time = time.time() - start_time
        logger.info(f"Labels from RAG retrieved in {taken_time} seconds.")
        return labels_from_rag, labels
    except Exception as e:
        logger.exception(f"Error occured when retrieving labels: {e}")
        error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
        error_message += "Failed to retrieve labels."
        raise Exception(error_message)


@celery_app.task(name="iterative autofill metadata extraction task")
@observe()
def iterative_metadata_extraction_task(
    paper_id: str,
    project_id: str | None,
    table_structure: list[dict],
    prev_response: list[dict] = None,
    metadata: dict = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    setup_langfuse_handler(langfuse_session_id)
    if metadata is None:
        metadata = {}
    metadata["generated_labels"] = []
    start_time = time.time()
    try:
        table_structure_hash = {field["name"]: field for field in table_structure}
        logger.info(f"Number of labels: {len(table_structure_hash)}")

        # check for circular dependencies
        graph = Graph()
        for field in table_structure:
            graph.add_node(field["name"])
        for field in table_structure:
            if (
                "relationships" in field
                and field["relationships"] is not None
                and len(field["relationships"]) > 0
            ):
                for relationship in field["relationships"]:
                    graph.add_edge(relationship["related_label"], field["name"])
        try:
            if graph.is_circular():
                raise Exception(
                    "Circular dependencies found among labels in table definition"
                )
        except Exception as e:
            logger.exception(f"Failed to check for circular dependencies: {e}")
            error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
            error_message += "Failed to check for circular dependencies."
            raise Exception(error_message)

        # only consider labels that generate is true
        # and their parent labels
        to_generate_labels = [
            field["name"] for field in table_structure if field["generate"]
        ]
        logger.info(f"Labels to generate: {to_generate_labels}")
        parent_labels_of_to_generate_labels, _ = get_parent_labels(
            table_structure_hash, to_generate_labels, [], {}
        )
        logger.info(f"Parent labels: {parent_labels_of_to_generate_labels}")
        to_generate_labels = to_generate_labels + parent_labels_of_to_generate_labels
        to_generate_labels = list(set(to_generate_labels))
        logger.info(f"Number of labels to generate: {len(to_generate_labels)}")

        if len(to_generate_labels) == 0:
            metadata["message"] = "No labels to generate"
            metadata["status"] = Status.SUCCESS.value
            metadata["generated_labels"] = []
            message = {
                "payload": prepare_final_response(prev_response, table_structure),
                "metadata": metadata,
            }
            send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_ITERATIVE, message)
            return message

        # first extract lables that are not dependent on other labels
        level_1_labels = [
            field["name"]
            for field in table_structure
            if (
                field["name"] in to_generate_labels
                and (
                    "relationships" not in field
                    or field["relationships"] is None
                    or len(field["relationships"]) == 0
                )
            )
        ]
        logger.info(f"Level 1 labels: {level_1_labels}")
        if len(level_1_labels) == 0:
            raise Exception(
                "All labels are dependent on other labels. No labels to extract"
            )

        # extract all contexts from the paper
        full_paper_contexts, title, total_chunks = None, None, None
        try:
            full_paper_contexts, total_chunks = extract_all_paper_texts(
                flag_id=paper_id
            )
            logger.info("Full paper contexts extracted successfully.")
        except Exception as e:
            logger.exception(f"Error occured when extracting all paper texts: {e}")

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                title = extract_title(
                    flag_id=paper_id,
                    langfuse_session_id=langfuse_session_id,
                )
                break
            except Exception:
                logger.exception("Error occured when extracting title")
                retry_count += 1
                if retry_count == MAX_RETRIES:
                    title = None
                time.sleep(10)
        logger.info(f"Title: {title}")

        # determin whether to extract level 1 or level 2 labels
        current_labels_to_extract = level_1_labels
        # start by extracting level 1 labels that are not dependent on other labels
        # they are also not present in the previous response
        reextracted_labels = []
        extracted_labels = []
        final_response = []
        level = 1
        total_loops = 0
        while (
            not check_all_labels_extracted(extracted_labels, to_generate_labels)
            and total_loops < iaf_settings.LOOP_LIMIT
        ):
            retries = 0
            total_loops += 1
            while retries < MAX_RETRIES:
                try:
                    if level > 1:
                        final_response, labels_reextracted = get_label_answers(
                            current_labels_to_extract,
                            table_structure,
                            table_structure_hash,
                            paper_id,
                            project_id,
                            level=level,
                            prev_response=prev_response,
                            has_parents=True,
                            full_paper_contexts=full_paper_contexts,
                            total_full_paper_chunks=total_chunks,
                            title=title,
                            langfuse_session_id=langfuse_session_id,
                        )
                    else:
                        final_response, labels_reextracted = get_label_answers(
                            current_labels_to_extract,
                            table_structure,
                            table_structure_hash,
                            paper_id,
                            project_id,
                            level=level,
                            prev_response=prev_response,
                            has_parents=False,
                            full_paper_contexts=full_paper_contexts,
                            total_full_paper_chunks=total_chunks,
                            title=title,
                            langfuse_session_id=langfuse_session_id,
                        )

                    extracted_labels.extend(current_labels_to_extract)
                    reextracted_labels.extend(labels_reextracted)
                    extracted_labels = list(set(extracted_labels))
                    logger.info(f"Extracted levels till now: {extracted_labels}")
                    prev_response = final_response
                    try:
                        current_labels_to_extract = find_next_labels_to_extract(
                            extracted_labels,
                            current_labels_to_extract,
                            to_generate_labels,
                            table_structure,
                        )
                    except Exception as e:
                        logger.exception(f"Error when tried to find next labels: {e}")
                        error_message = ITERATIVE_AUTOFILL_ERROR_MESSAGE
                        error_message += (
                            "Error occured when finding next labels to extract."
                        )
                        raise Exception(error_message)
                    level += 1
                    logger.info(
                        f"Level {level} labels to extract: {current_labels_to_extract}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Error extracting level {level} labels: {e}")
                    logger.error("Traceback: " + traceback.format_exc())
                    retries += 1
                    if retries == MAX_RETRIES:
                        if ITERATIVE_AUTOFILL_ERROR_MESSAGE in str(e):
                            metadata["message"] = str(e)
                        else:
                            metadata["message"] = ITERATIVE_AUTOFILL_ERROR_MESSAGE
                        metadata["status"] = Status.FAILED.value

                        if len(final_response) > 0:
                            metadata["generated_labels"] = reextracted_labels
                            final_response = prepare_final_response(
                                final_response, table_structure
                            )
                            message = {
                                "payload": final_response,
                                "metadata": metadata,
                            }
                        else:
                            metadata["generated_labels"] = []
                            message = {
                                "payload": prepare_final_response(
                                    prev_response, table_structure
                                ),
                                "metadata": metadata,
                            }
                        send_to_backend(
                            BackendEventEnumType.PRESET_AUTOFILL_ITERATIVE, message
                        )
                        return message
                    logger.info(f"Retrying level {level} labels extraction")

        # fill filnal response with dummy values for misssing labels
        final_response = prepare_final_response(final_response, table_structure)
        time_taken = time.time() - start_time
        metadata["message"] = "Iterative metadata extraction process completed"
        metadata["status"] = Status.SUCCESS.value
        metadata["generated_labels"] = reextracted_labels
        metadata["ai_metadata"] = {"time_taken": time_taken}
        message = {
            "payload": final_response,
            "metadata": metadata,
        }
        logger.info(f"Time taken: {time_taken}")
        send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_ITERATIVE, message)
        return message

    except Exception as e:
        if ITERATIVE_AUTOFILL_ERROR_MESSAGE in str(e):
            metadata["message"] = str(e)
            metadata["status"] = Status.FAILED.value
            metadata["generated_labels"] = []
            message = {
                "payload": prepare_final_response(prev_response, table_structure),
                "metadata": metadata,
            }
            send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_ITERATIVE, message)
            return message
        else:
            logger.exception(f"Error occured when iterative autofill: {e}")
            metadata["message"] = ITERATIVE_AUTOFILL_ERROR_MESSAGE
            metadata["status"] = Status.FAILED.value
            metadata["generated_labels"] = []
            message = {
                "payload": prepare_final_response(prev_response, table_structure),
                "metadata": metadata,
            }
            send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_ITERATIVE, message)
            return message
