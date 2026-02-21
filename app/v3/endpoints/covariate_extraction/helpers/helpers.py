import re

from anthropic import InternalServerError as AnthropicInternalServerError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSerializable

from app.configs import settings
from app.core.auto.chat_model import AutoChatModel
from app.redis import redis_client
from app.utils.cache import retrieve_gemini_cache
from app.utils.files import create_file_input
from app.utils.llms import (
    batch_invoke_chain_with_retry,
    batch_invoke_llm_with_retry,
    get_message_text,
    invoke_chain_with_retry,
    invoke_llm_with_retry,
)
from app.v3.endpoints.covariate_extraction.configs import settings as cov_settings
from app.v3.endpoints.covariate_extraction.constants import (
    SEPARATOR,
    paper_dependent_fields,
)
from app.v3.endpoints.covariate_extraction.exceptions import (
    AnthropicServerFailed,
    ClaudeImageProcessingFailed,
    ContextSummarizationFailed,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger
from app.v3.endpoints.covariate_extraction.prompts import (
    QA_ON_PDF_PROMPT,
    TRIAL_ARM_ADVERSE_EVENT,
)
from app.v3.endpoints.covariate_extraction.rag_tasks import (
    context_summarization_task,
    get_context_docs_task,
    select_contexts,
)
from app.v3.endpoints.covariate_extraction.schemas import (
    AdversePlotDetails,
    MetaAnalysisTableField,
)
from app.v3.endpoints.get_paper_labels.exceptions import QuestionRephrasingFailed

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


def rephrase_question(
    chain: RunnableSerializable,
    questions: list[str],
    batch: bool,
    langfuse_handler=None,
) -> str:
    rephrase_questions = []
    batch_questions = []
    for question in questions:
        question_key = f"covariate_rephrase_{question}"

        if question_key in redis_client:

            cached_question = redis_client.getex(question_key, ex=REDIS_LIVE_TIME)
            rephrase_questions.append(cached_question)

        else:
            try:

                if not batch:
                    rephrased_question = invoke_chain_with_retry(
                        chain, {"question": question}
                    )

                    rephrase_questions.append(rephrased_question)
                    redis_client.setex(
                        question_key, REDIS_LIVE_TIME, rephrased_question
                    )
                else:
                    batch_question = {"question": question}
                    batch_questions.append(batch_question)

            except AnthropicInternalServerError:
                logger.exception("An error occurred at Anthropic's end.")
                raise AnthropicServerFailed()
            except Exception:
                logger.exception("An error occurred while rephrasing the question.")
                raise QuestionRephrasingFailed

    if batch_questions != []:
        try:
            batch_size = cov_settings.CONTEXT_QA_BATCH_SIZE
            rephrased_batch_questions = batch_invoke_chain_with_retry(
                chain,
                batch_questions,
                config={
                    "max_concurrency": batch_size,
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
            )
            rephrase_questions.extend(rephrased_batch_questions)
            for batch_ques in rephrased_batch_questions:
                question_key = f"rephrase_{batch_ques}"
                redis_client.setex(question_key, REDIS_LIVE_TIME, batch_ques)
        except AnthropicInternalServerError:
            logger.exception("An error occurred at Anthropic's end.")
            raise AnthropicServerFailed()
        except Exception:
            logger.exception("An error occurred while rephrasing the question.")
            raise QuestionRephrasingFailed

    return rephrase_questions


def adverse_get_trial_arm_result(
    llm: BaseChatModel, arms: str, table: str, endpoints: str
):

    prompt = TRIAL_ARM_ADVERSE_EVENT.format(arms=arms, table=table, endpoints=endpoints)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )

    try:

        result = invoke_llm_with_retry(
            llm.with_structured_output(AdversePlotDetails), [message]
        )

    except AnthropicInternalServerError:
        logger.exception("An error occurred at Anthropic's end.")
        raise AnthropicServerFailed()
    except Exception:
        logger.exception("An error occurred while processing the image with claude.")
        raise ClaudeImageProcessingFailed()

    return result


def check_final_result(result, final_result):
    null_values = [
        "N/A",
        "None",
        "none",
        "null",
        "Null",
        "nan",
        "NaN",
        0.0,
        0,
        "0",
        None,
        "",
    ]
    try:
        arm_and_endpoint = []
        for i in range(len(result["data"])):
            figure_arm = result["data"][i]["ARM NAME"].lower().strip()
            figure_endpoint = result["data"][i]["ENDPOINT"].lower().strip()
            arm_and_endpoint.append(figure_arm + figure_endpoint)

        if len(arm_and_endpoint) != len(set(arm_and_endpoint)):
            logger.error("Duplicate arm and endpoint found")
            return final_result
        else:
            logger.info("No duplicate arm and endpoint found")

        for i in range(len(result["data"])):
            figure_response = result["data"][i]
            figure_arm = result["data"][i]["ARM NAME"]
            figure_endpoint = result["data"][i]["ENDPOINT"]
            figure_arm = figure_arm.lower().strip()
            figure_endpoint = figure_endpoint.lower().strip()
            for j in range(len(final_result["data"])):
                final_arm = final_result["data"][j]["ARM NAME"]
                final_endpoint = final_result["data"][j]["ENDPOINT"]
                final_arm = final_arm.lower().strip()
                final_endpoint = final_endpoint.lower().strip()
                if figure_arm == final_arm and figure_endpoint == final_endpoint:

                    final_response = final_result["data"][j]
                    for key in figure_response.keys():
                        if (
                            figure_response[key] not in null_values
                            and figure_response[key] != final_response[key]
                        ):

                            final_response[key] = figure_response[key]

    except Exception as e:
        logger.exception(f"Error: {e}")

    return final_result


def check_if_covariate_table(table_definition: list[MetaAnalysisTableField]):
    identifying_cols = ["endpoint", "dv", "dvid"]
    names = [field.name.lower().strip() for field in table_definition]
    return all(col not in names for col in identifying_cols)


def get_summarized_contexts(
    retrieval_questions: list[str],
    summarization_question: str,
    paper_id: str,
    project_id: str,
    org_paper_id: str = None,
    supplementary_id: str = None,
    langfuse_handler=None,
):

    arm_contexts = get_context_docs_task(
        queries=retrieval_questions,
        flag_id=paper_id,
        project_id=project_id,
        langfuse_handler=langfuse_handler,
    )

    supplementary_arm_contexts = []

    if supplementary_id:
        supplementary_arm_contexts = get_context_docs_task(
            queries=retrieval_questions,
            flag_id=org_paper_id,
            project_id=project_id,
            supplementary_id=supplementary_id,
            langfuse_handler=langfuse_handler,
        )

    arm_contexts = select_contexts(arm_contexts)
    if supplementary_arm_contexts != []:
        supplementary_arm_contexts = select_contexts(supplementary_arm_contexts)
        arm_contexts += supplementary_arm_contexts
    arm_contexts = SEPARATOR.join(arm_contexts)

    arms_with_questions = {
        "trial_arms": {
            "summarization_question": summarization_question,
            "contexts": arm_contexts,
        }
    }

    try:
        contexts = context_summarization_task(
            contexts=[arms_with_questions["trial_arms"]["contexts"]],
            questions=[arms_with_questions["trial_arms"]["summarization_question"]],
        )

    except AnthropicInternalServerError:
        logger.exception("An error occurred at Anthropic's end.")
        raise AnthropicServerFailed()
    except Exception:
        logger.exception("An error occurred while summarizing the contexts.")
        raise ContextSummarizationFailed

    return contexts


def get_questions_from_table_structure(
    chain: RunnableSerializable,
    table_structure: dict,
    arms: list[str],
    langfuse_handler=None,
) -> dict[str, list[str]]:
    questions = {}
    label_to_inputs_map = {}
    label_to_summarization_inputs_map = {}
    all_questions = []
    all_summarization_questions = []
    input_count, summarization_count = 0, 0
    for field in table_structure:
        if field["name"].lower() in paper_dependent_fields:
            all_questions.append(field["description"])
            all_summarization_questions.append(field["description"])
            label_to_inputs_map[input_count] = field["name"]
            label_to_summarization_inputs_map[summarization_count] = field["name"]
            summarization_count += 1
            input_count += 1
        else:
            for arm in arms:
                all_questions.append(f"{field['description']} for {arm} arm")
                label_to_inputs_map[input_count] = field["name"]
                input_count += 1
            if len(arms) > 0:
                summarization_question = (
                    f"{field['description']} for following arms: {arms}"
                )
                all_summarization_questions.append(summarization_question)
                label_to_summarization_inputs_map[summarization_count] = field["name"]
                summarization_count += 1
            else:
                summarization_question = field["description"]
                all_summarization_questions.append(summarization_question)
                label_to_summarization_inputs_map[summarization_count] = field["name"]
                summarization_count += 1

    rephrased_questions = rephrase_question(chain, all_questions, True)
    rephrased_summarization_questions = rephrase_question(
        chain, all_summarization_questions, True, langfuse_handler
    )

    for i, question in enumerate(rephrased_questions):
        label_name = label_to_inputs_map[i]
        if label_name not in questions:
            questions[label_name] = {}

        if "retrieval" in questions[label_name]:
            questions[label_name]["retrieval"].append(question)
        else:
            questions[label_name]["retrieval"] = [question]

    for i, question in enumerate(rephrased_summarization_questions):
        label_name = label_to_summarization_inputs_map[i]
        if label_name not in questions:
            questions[label_name] = {}

        if "summarization" in questions[label_name]:
            questions[label_name]["summarization"] += f". {question}"
        else:
            questions[label_name]["summarization"] = question

    return questions


def combine_adverse_contexts(
    questions_with_contexts: dict,
    field_names: list[str],
    separator: str = "\n-----------\n\n",
) -> str:
    combined_contexts = []
    count = 1
    for field_name in field_names:
        if field_name in questions_with_contexts:
            question_with_context = f"""
            {count}.
            {field_name}: {questions_with_contexts[field_name]['contexts']}
            """
            question_with_context = re.sub(r"\s+", " ", question_with_context).strip()
            combined_contexts.append(question_with_context)
            count += 1

    return separator.join(combined_contexts)


def extract_contexts_for_query_from_pdf(
    flag_id: str,
    file_details: dict,
    supplementary_file_details: dict,
    questions: list[str] | str,
    langfuse_handler=None,
) -> list[str]:
    if file_details["adverse_cache_name"] == "N/A":
        logger.info(f"Creating uncached inputs for flag id: {flag_id}")
        context_agent = AutoChatModel.from_model_name(
            model_name=cov_settings.CONTEXT_EXTRACTOR_LLM,
            temperature=0.2,
        )
        if isinstance(questions, list):
            all_inputs = []
            for question in questions:
                prompt = QA_ON_PDF_PROMPT.format(questions=question)
                if supplementary_file_details:
                    inputs = create_inputs_to_extract_contexts(
                        pdf_paths=[
                            file_details["pdf_path"],
                            supplementary_file_details["pdf_path"],
                        ],
                        prompt=prompt,
                    )
                else:
                    inputs = create_inputs_to_extract_contexts(
                        pdf_paths=file_details["pdf_path"],
                        prompt=prompt,
                    )
                all_inputs.append(inputs)

            all_results = batch_invoke_llm_with_retry(
                context_agent,
                all_inputs,
                config={
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
            )
            all_results = [get_message_text(result) for result in all_results]
            return all_results
        else:
            prompt = QA_ON_PDF_PROMPT.format(questions=questions)
            if supplementary_file_details:
                inputs = create_inputs_to_extract_contexts(
                    pdf_paths=[
                        file_details["pdf_path"],
                        supplementary_file_details["pdf_path"],
                    ],
                    prompt=prompt,
                )
            else:
                inputs = create_inputs_to_extract_contexts(
                    pdf_paths=file_details["pdf_path"],
                    prompt=prompt,
                )
            result = invoke_llm_with_retry(
                llm=context_agent,
                messages=inputs,
                config={
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
            )
            return get_message_text(result)
    else:
        logger.info(f"Getting cached inputs for flag id: {flag_id}")
        cache = retrieve_gemini_cache(file_details["adverse_cache_name"])
        agent = AutoChatModel.from_model_name(
            model_name=cov_settings.CONTEXT_EXTRACTOR_LLM,
            temperature=0.2,
            cached_content=cache.name,
        )

        if isinstance(questions, list):
            all_inputs = []
            for question in questions:
                prompt = QA_ON_PDF_PROMPT.format(questions=question)
                all_inputs.append(
                    [HumanMessage(content=[{"type": "text", "text": prompt}])]
                )
            all_results = batch_invoke_llm_with_retry(
                agent,
                all_inputs,
                config={
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
            )
            all_results = [get_message_text(result) for result in all_results]
            return all_results
        else:
            prompt = QA_ON_PDF_PROMPT.format(questions=questions)
            messages = [HumanMessage(content=[{"type": "text", "text": prompt}])]
            result = invoke_llm_with_retry(
                llm=agent,
                messages=messages,
                config={
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
            )
            return get_message_text(result)


def create_inputs_to_extract_contexts(
    pdf_paths: list[str] | str,
    prompt: str,
) -> list[str]:
    if isinstance(pdf_paths, list):
        all_pdf_data = []
        for pdf_path in pdf_paths:
            pdf_data = create_file_input(
                pdf_path, model_name=cov_settings.CONTEXT_EXTRACTOR_LLM
            )
            all_pdf_data.append(pdf_data)
    else:
        pdf_data = create_file_input(
            pdf_paths, model_name=cov_settings.CONTEXT_EXTRACTOR_LLM
        )
        all_pdf_data = [pdf_data]

    all_messages = [
        HumanMessage(content=all_pdf_data),
        HumanMessage(content=prompt),
    ]

    return all_messages
