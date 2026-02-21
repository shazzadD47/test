import os
import time
from uuid import uuid4

from celery import shared_task
from langchain_core.messages import HumanMessage
from langfuse import observe
from pydantic import BaseModel, Field, create_model

from app.chains import prepare_question_rephrasing_chain
from app.configs import settings
from app.constants import d_type_map
from app.core.auto.chat_model import AutoChatModel
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.utils.image import (
    convert_image_to_base64,
)
from app.utils.llms import invoke_chain_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints import Status
from app.v3.endpoints.covariate_extraction.chains import (
    table_definition_from_contexts_chain,
)
from app.v3.endpoints.covariate_extraction.configs import settings as cov_settings
from app.v3.endpoints.covariate_extraction.constants import (
    ADVERSE_EVENT_ERROR_MESSAGE,
    SEPARATOR,
)
from app.v3.endpoints.covariate_extraction.exceptions import OpenAiImageProcessingFailed
from app.v3.endpoints.covariate_extraction.helpers.covariate_context_helpers import (
    cache_pdf_file,
    process_pdf_file,
)
from app.v3.endpoints.covariate_extraction.helpers.helpers import (
    adverse_get_trial_arm_result,
    check_final_result,
    combine_adverse_contexts,
    get_questions_from_table_structure,
)
from app.v3.endpoints.covariate_extraction.helpers.labels_extractor import (
    extract_contexts_of_labels,
)
from app.v3.endpoints.covariate_extraction.helpers.trial_arms_extractor import (
    get_arms_doses_contexts,
)
from app.v3.endpoints.covariate_extraction.helpers.utils import (
    extract_data_from_table_image,
    extract_image_type,
    get_image_from_url,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger
from app.v3.endpoints.covariate_extraction.prompts import (
    ADVERSE_FINAL_PROMPT_TEMPLATE,
    FINAL_PAPER_LABEL_PROMPT_TEMPLATE,
)
from app.v3.endpoints.covariate_extraction.rag_tasks import (
    context_summarization_task,
    select_contexts,
)


@observe()
@track_all_llm_costs
def extract_adverse_event_metadata(
    figure_url: str,
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    metadata: dict = None,
    langfuse_session_id: str = None,
) -> dict:
    if metadata is None:
        metadata = {}
    if "ai_metadata" not in metadata:
        metadata["ai_metadata"] = {}

    try:
        total_time_start = time.time()
        start_time = time.time()
        supplementary_id = None

        if langfuse_session_id is None:
            langfuse_session_id = uuid4().hex
        langfuse_handler = setup_langfuse_handler(
            langfuse_session_id, name="adverse_event"
        )

        if "supplementary" in paper_id:
            supplementary_full_id = paper_id
            supplementary_id = paper_id.split("supplementary-")[1]
            paper_id = paper_id.split("-supplementary")[0]
            file_details = process_pdf_file(paper_id)
            supplementary_file_details = process_pdf_file(supplementary_full_id)
        else:
            file_details = process_pdf_file(paper_id)
            supplementary_file_details = None

        cache_name = f"ge-adverse-event-{uuid4()}"
        if supplementary_file_details:
            cache = cache_pdf_file(
                pdf_paths=[
                    file_details["pdf_path"],
                    supplementary_file_details["pdf_path"],
                ],
                flag_id=paper_id,
                cache_name=cache_name,
            )
        else:
            cache = cache_pdf_file(
                pdf_paths=file_details["pdf_path"],
                flag_id=paper_id,
                cache_name=cache_name,
            )
        if cache:
            file_details["adverse_cache_name"] = cache_name
            if supplementary_file_details:
                supplementary_file_details["adverse_cache_name"] = cache_name
        else:
            file_details["adverse_cache_name"] = "N/A"
            if supplementary_file_details:
                supplementary_file_details["adverse_cache_name"] = "N/A"

        llm_gpt = AutoChatModel.from_model_name(
            model_name=settings.GPT_4_TEXT_MODEL,
            temperature=0.2,
        )
        trial_arm_llm = AutoChatModel.from_model_name(
            model_name=cov_settings.TRIAL_ARM_LLM,
            temperature=0.2,
        )

        rephrase_chain = prepare_question_rephrasing_chain(llm=trial_arm_llm)
        logger.info(f"Chains initialized in {time.time() - start_time} seconds.")

        start_time = time.time()
        image, media_type, image_save_path = get_image_from_url(
            figure_url, return_media_type=True
        )
        image = convert_image_to_base64(image)
        image_type = extract_image_type(
            image=image,
            media_type=media_type,
        )
        logger.info(f"image type: {image_type}")
        table = extract_data_from_table_image(
            file_path=image_save_path,
        )
        table = table.to_string(index=False)
        logger.info(f"extracted_table: {table}")
        if os.path.exists(image_save_path):
            os.remove(image_save_path)

        logger.info(f"Image retrieved in {time.time() - start_time} seconds.")
        start_time = time.time()
        if image_type.lower() == "table":
            logger.info("extracting adverse events")
            arms = get_arms_doses_contexts(
                project_id=project_id,
                flag_id=paper_id,
                file_details=file_details,
                supplementary_file_details=supplementary_file_details,
                supplementary_id=supplementary_id,
                langfuse_handler=langfuse_handler,
            )
            endpoint_description = ""
            for field in table_structure:
                if field["name"].lower() == "endpoint":
                    endpoint_description = field["description"]
                    break

            logger.info(f"Arms retrieved in {time.time() - start_time} seconds.")
            start_time = time.time()

            trial_arm_result = adverse_get_trial_arm_result(
                trial_arm_llm, arms, table, endpoint_description
            )
            arms = trial_arm_result.arms
            endpoints = trial_arm_result.final_endpoints
            logger.info(f"arms:{arms}")
            logger.info(f"endpoints:{endpoints}")

            questions = get_questions_from_table_structure(
                rephrase_chain, table_structure, arms
            )

            contexts = []
            labels = {
                field["name"]: (
                    (
                        d_type_map[field["d_type"]] | None
                        if field["d_type"] != "string"
                        else d_type_map[field["d_type"]]
                    ),
                    Field(
                        None if field["d_type"] != "string" else ...,
                        description=field["description"],
                    ),
                )
                for field in table_structure
                if field["c_type"] != "paper_label"
            }

            labels["ARM NAME"] = (
                str,
                Field(..., description="The name of the trial arm"),
            )
            labels_names = list(labels.keys())
            paper_labels = {
                field["name"]: (
                    (
                        d_type_map[field["d_type"]] | None
                        if field["d_type"] != "string"
                        else d_type_map[field["d_type"]]
                    ),
                    Field(
                        None if field["d_type"] != "string" else ...,
                        description=field["description"],
                    ),
                )
                for field in table_structure
                if field["c_type"] == "paper_label"
            }
            paper_labels_names = list(paper_labels.keys())

            all_questions = []
            for _, question in questions.items():
                all_questions.extend(question["retrieval"])

            try:
                contexts = extract_contexts_of_labels(
                    flag_id=paper_id,
                    project_id=project_id,
                    questions=all_questions,
                    file_details=file_details,
                    supplementary_file_details=supplementary_file_details,
                    supplementary_id=supplementary_id,
                    langfuse_handler=langfuse_handler,
                )
                if file_details["pdf_path"] == "N/A":
                    contexts_from_main_doc, contexts_from_supp = contexts
                    start, end = 0, 0

                    for field_name in questions:
                        num_questions = len(questions[field_name]["retrieval"])
                        end = start + num_questions

                        selected_contexts = select_contexts(
                            contexts_from_main_doc[start:end]
                        )
                        selected_contexts = [
                            f"{count + 1}. {context}"
                            for count, context in enumerate(selected_contexts)
                        ]
                        selected_context_supp = []
                        if contexts_from_supp != []:
                            selected_context_supp = select_contexts(
                                contexts_from_supp[start:end]
                            )
                            selected_context_supp = [
                                f"{count + 1}. {context}"
                                for count, context in enumerate(selected_context_supp)
                            ]
                        selected_contexts = selected_contexts + selected_context_supp

                        contexts_per_field = SEPARATOR.join(selected_contexts)

                        questions[field_name]["contexts"] = contexts_per_field
                        start = end

                    all_contexts = [
                        questions[field_name]["contexts"] for field_name in questions
                    ]
                    all_questions = [
                        questions[field_name]["summarization"]
                        for field_name in questions
                    ]
                    contexts = context_summarization_task(
                        questions=all_questions,
                        contexts=all_contexts,
                        langfuse_session_id=langfuse_session_id,
                    )
                    for i, (_, info) in enumerate(questions.items()):
                        info["contexts"] = contexts[i]
                else:
                    start, end = 0, 0
                    for field_name in questions:
                        num_questions = len(questions[field_name]["retrieval"])
                        end = start + num_questions
                        questions[field_name]["contexts"] = SEPARATOR.join(
                            contexts[start:end]
                        )
                        start = end

            except Exception as e:
                error_message = "An error occurred while retrieving contexts."
                logger.exception(f"{error_message} {e}")
                error_message = ADVERSE_EVENT_ERROR_MESSAGE + error_message
                raise Exception(error_message)

            figure_contexts = combine_adverse_contexts(questions, labels_names)
            paper_contexts = combine_adverse_contexts(questions, paper_labels_names)
            figure_contexts = (
                "The arms are: " + ", ".join(arms) + "\n" + figure_contexts
            )
            logger.info(f"Contexts retrieved in {time.time() - start_time} seconds.")
            ArmDetails = create_model("ArmDetails", **labels)
            PaperDetails = create_model("PaperLabels", **paper_labels)

            class DetailsSchema(BaseModel):
                data: list[ArmDetails]  # type: ignore

            class PaperLabelsSchema(BaseModel):
                paper_labels: list[PaperDetails]  # type: ignore

            table_values_from_contexts_chain = table_definition_from_contexts_chain(
                llm=llm_gpt, schema=PaperLabelsSchema
            )
            if len(paper_labels) != 0:
                try:

                    result_from_rag = invoke_chain_with_retry(
                        chain=table_values_from_contexts_chain,
                        input={
                            "contexts": paper_contexts,
                        },
                        config={
                            "callbacks": (
                                [langfuse_handler] if langfuse_handler else None
                            ),
                        },
                    )

                    result_from_rag = result_from_rag.dict()
                except Exception as e:
                    logger.exception(
                        f"An error occurred while retrieving paper labels: {e}"
                    )

                    error_message = e + "Error occured when retrieving paper labels."
                    raise Exception(error_message)
            else:
                logger.info("No paper contexts")
                result_from_rag = None

            chain = llm_gpt.with_structured_output(DetailsSchema)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": ADVERSE_FINAL_PROMPT_TEMPLATE.format(
                            arms=arms, table=table, endpoints=endpoints
                        ),
                    },
                ]
            )

            start_time = time.time()
            try:

                result = invoke_chain_with_retry(
                    chain,
                    [message],
                    config={
                        "callbacks": [langfuse_handler] if langfuse_handler else None,
                    },
                )
                result = result.model_dump()

            except Exception as e:
                logger.exception(
                    f"An error occurred while retrieving paper labels: {e}"
                )

                error_message = e + "Error occured when retrieving paper labels."
                raise Exception(error_message)

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": FINAL_PAPER_LABEL_PROMPT_TEMPLATE.format(
                            contexts=figure_contexts,
                            output=result,
                        ),
                    },
                ]
            )
            try:
                final_result = invoke_chain_with_retry(
                    chain,
                    [message],
                    config={
                        "callbacks": [langfuse_handler] if langfuse_handler else None,
                    },
                )

            except Exception:
                logger.exception("An error occurred while processing with OpenAI.")

                raise OpenAiImageProcessingFailed()

            final_result = final_result.model_dump()
            if len(paper_labels) != 0:
                for i in range(len(final_result["data"])):
                    final_result["data"][i].update(result_from_rag["paper_labels"][0])

            final_result = check_final_result(result, final_result)

            metadata["message"] = (
                "Adverse Event data extracted succesfully from Table Image"
            )
            metadata["status"] = Status.SUCCESS.value
            metadata["ai_metadata"]["runtime"] = time.time() - total_time_start

            message = {"payload": final_result, "metadata": metadata}

            return message
        else:
            logger.error("image is not a table")
            raise Exception("image is not a table")
    except Exception as e:
        logger.exception(f"Error occured in adverse_event: {e}")
        metadata["message"] = "Error occured in adverse_event:"
        metadata["status"] = Status.FAILED.value

        message = {
            "payload": {},
            "metadata": metadata,
        }

        return message


@shared_task(
    name="adverse_event_metadata_extraction_task",
    bind=True,
    max_retries=0,
    default_retry_delay=10,
)
def adverse_event_extraction_service(
    self,
    figure_url: str,
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    metadata: dict = None,
) -> dict:
    retry_count = 0
    while retry_count < cov_settings.MAX_RETRIES:
        try:
            output = extract_adverse_event_metadata(
                figure_url, paper_id, project_id, table_structure, metadata
            )
            if (
                "metadata" in output
                and "status" in output["metadata"]
                and output["metadata"]["status"] == Status.SUCCESS.value
            ):
                send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, output)
                return output
            else:
                retry_count += 1
                retry_delay = cov_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
                time.sleep(retry_delay)
                if "metadata" in output and isinstance(output["metadata"], dict):
                    metadata.update(output["metadata"])
                if retry_count >= cov_settings.MAX_RETRIES:
                    send_to_backend(
                        BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, output
                    )
                    return output
                continue
        except Exception as exc:
            logger.exception(f"Error occured in adverse_event: {exc}")
            retry_count += 1
            retry_delay = cov_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
            time.sleep(retry_delay)
            if retry_count >= cov_settings.MAX_RETRIES:
                output = {
                    "payload": {},
                    "metadata": metadata,
                }
                output["metadata"]["status"] = Status.FAILED.value
                output["metadata"]["message"] = ADVERSE_EVENT_ERROR_MESSAGE
                send_to_backend(BackendEventEnumType.PRESET_AUTOFILL_COVERIATE, output)
                return output
            continue
