import re
from uuid import uuid4

from anthropic import RateLimitError as AnthropicRateLimitError
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from openai import RateLimitError as OpenAIRateLimitError

from app.configs import settings
from app.core.vector_store import VectorStore
from app.redis import redis_client
from app.utils.llms import batch_invoke_chain_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.covariate_extraction.chains import (
    context_qa_chain,
    context_qa_chain_claude,
    rephrase_chain,
)
from app.v3.endpoints.covariate_extraction.configs import settings as cov_settings
from app.v3.endpoints.covariate_extraction.constants import MAX_RETRIES
from app.v3.endpoints.covariate_extraction.exceptions import (
    ContextSummarizationFailed,
    QuestionRephrasingFailed,
)
from app.v3.endpoints.covariate_extraction.helpers.utils import (
    find_suggested_loc_from_query,
)
from app.v3.endpoints.covariate_extraction.logging import celery_logger as logger

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


@observe()
def rephrase_question(
    questions: list[str],
    keys: list[str] = None,
    langfuse_session_id: str = None,
) -> list[str] | list[dict[str, str]]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)

    batch_size = cov_settings.QUERY_REPHRASE_BATCH_SIZE
    if keys is None:
        rephrased_questions = dict(enumerate(questions))
    else:
        rephrased_questions = dict(zip(keys, questions))

    indices_to_rephrase = []
    for key, question in rephrased_questions.items():
        question_key = f"paraphrased_{question}"
        if question_key in redis_client:
            rephrased_question = redis_client.getex(question_key, ex=REDIS_LIVE_TIME)
            rephrased_questions[key] = rephrased_question
        else:
            indices_to_rephrase.append(key)

    selected_questions = [rephrased_questions[key] for key in indices_to_rephrase]

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            rephrased_questions_for_selected_questions = batch_invoke_chain_with_retry(
                chain=rephrase_chain,
                input=[{"question": question} for question in selected_questions],
                config={"max_concurrency": batch_size, "callbacks": [langfuse_handler]},
                max_retries=1,
            )
            break
        except AnthropicRateLimitError:
            logger.exception("Anthropic rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
        except OpenAIRateLimitError:
            logger.exception("OpenAI rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
        except Exception:
            logger.exception("An error occurred while rephrasing the question.")
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
    for i, key in enumerate(indices_to_rephrase):
        rephrased_questions[key] = rephrased_questions_for_selected_questions[i]

    if keys is None:
        only_rephrased_questions = {
            rephrased_questions[i] for i in range(len(questions))
        }
        return list(only_rephrased_questions)
    else:
        return [{key: rephrased_questions[key]} for key in keys]


def select_contexts(
    contexts: list[list[str]],
) -> list[str]:
    if len(contexts) == 1:
        return contexts
    selected_contexts = []
    count_limit = len(contexts[0]) // 2
    if len(contexts) == 2:
        selected_contexts = contexts[0] + contexts[1][: count_limit // 2]
        unique_contexts = []
        for context in selected_contexts:
            if context not in unique_contexts:
                unique_contexts.append(context)
        return unique_contexts
    for count, context in enumerate(zip(*contexts[:-1])):
        for single_context in context:
            if single_context not in selected_contexts:
                selected_contexts.append(single_context)
        if len(set(selected_contexts)) < cov_settings.TOP_K:
            continue
        if count >= count_limit:
            break

    remaining_contexts = [
        context
        for context in contexts[-1][: count_limit // 2]
        if context not in selected_contexts
    ]
    selected_contexts.extend(remaining_contexts)
    unique_contexts = []
    for context in selected_contexts:
        if context not in unique_contexts:
            unique_contexts.append(context)
    return unique_contexts


def get_context_docs_task(
    queries: list[str],
    top_k: int = 20,
    flag_id: str = None,
    project_id: str = None,
    supplementary_id: str = None,
    langfuse_handler=None,
) -> list[str]:
    batch_size = cov_settings.BATCH_SIZE
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id
    if supplementary_id:
        filter_params["supplementary_id"] = supplementary_id

    filter_params["file_type"] = "document"

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": top_k}
    )
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            similar_docs = batch_invoke_chain_with_retry(
                chain=retriever,
                input=queries,
                config={
                    "max_concurrency": batch_size,
                    "callbacks": [langfuse_handler] if langfuse_handler else None,
                },
                max_retries=1,
            )
            break
        except AnthropicRateLimitError:
            logger.exception("Anthropic rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
        except OpenAIRateLimitError:
            logger.exception("OpenAI rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
        except Exception:
            logger.exception("An error occurred while retrieving similar documents.")
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
    updated_similar_docs = []
    for docs_for_single_query in similar_docs:
        unique_docs = []
        for doc in docs_for_single_query:
            if doc.page_content not in unique_docs:
                unique_docs.append(doc.page_content)
        updated_similar_docs.append(unique_docs)
    return updated_similar_docs


@observe()
def context_summarization_task(
    contexts: list[str],
    questions: list[str],
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_client = get_client()
    langfuse_client.update_current_trace(session_id=langfuse_session_id)
    langfuse_handler = CallbackHandler()

    retry_count = 0
    result = contexts.copy()
    batch_size = cov_settings.CONTEXT_QA_BATCH_SIZE
    while retry_count < MAX_RETRIES:
        try:
            try:
                result = batch_invoke_chain_with_retry(
                    chain=context_qa_chain,
                    input=[
                        {
                            "contexts": contexts[i],
                            "questions": questions[i],
                        }
                        for i in range(len(contexts))
                    ],
                    config={
                        "max_concurrency": batch_size,
                        "callbacks": [langfuse_handler],
                    },
                    max_retries=1,
                )
            except Exception:
                result = batch_invoke_chain_with_retry(
                    chain=context_qa_chain_claude,
                    input=[
                        {
                            "contexts": contexts[i],
                            "questions": questions[i],
                        }
                        for i in range(len(contexts))
                    ],
                    config={
                        "max_concurrency": batch_size,
                        "callbacks": [langfuse_handler],
                    },
                    max_retries=1,
                )
            break
        except AnthropicRateLimitError:
            logger.exception("Anthropic rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
        except OpenAIRateLimitError:
            logger.exception("OpenAI rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
        except Exception:
            logger.exception("An error occurred while summarizing the contexts.")
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise ContextSummarizationFailed()
            continue
    return result


@observe()
def get_questions_from_table_structure(
    table_structure: dict,
    paper_dependent_fields: list[str],
    arms: list[str],
    langfuse_session_id: str = None,
) -> dict[str, list[str]]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_client = get_client()
    langfuse_client.update_current_trace(session_id=langfuse_session_id)

    all_questions = {}
    rephrase_input = {
        "questions": [],
        "keys": [],
    }

    for field in table_structure:
        all_questions[field["name"]] = {
            "description": field["description"],
            "location_info_added": False,
        }
        if field["c_type"] == "paper_label" or field["name"] in paper_dependent_fields:
            field_info = field["name"] + ": " + field["description"]
            if field["name"] not in rephrase_input["keys"]:
                rephrase_input["keys"].append(field["name"])
                rephrase_input["questions"].append(field_info)

        else:
            field_info = field["name"] + ": " + field["description"]
            if field["name"] not in rephrase_input["keys"]:
                rephrase_input["keys"].append(field["name"])
                rephrase_input["questions"].append(field_info)
            for arm in arms:
                field_info = (
                    field["name"] + ": " + field["description"] + f"|_for:{arm}"
                )
                field_key = field["name"] + f"|_for:{arm}"
                if field_key not in rephrase_input["keys"]:
                    rephrase_input["keys"].append(field_key)
                    rephrase_input["questions"].append(field_info)

            summarization_field_info = (
                field["name"] + ": " + field["description"] + f"|_for_all:{arms}"
            )
            if "summarization_" + field["name"] not in rephrase_input["keys"]:
                rephrase_input["keys"].append("summarization_" + field["name"])
                rephrase_input["questions"].append(summarization_field_info)

    questions = rephrase_question(
        questions=rephrase_input["questions"],
        keys=rephrase_input["keys"],
        langfuse_session_id=langfuse_session_id,
    )

    for question_info in questions:
        for key, value in question_info.items():
            if key.startswith("summarization"):
                try:
                    question, _ = find_suggested_loc_from_query(value)
                except Exception:
                    question = value
                field_name = key.split("summarization_")[1]
                all_questions[field_name]["summarization_question"] = question
            else:
                if key in all_questions:
                    field_name = key
                    try:
                        question, _ = find_suggested_loc_from_query(value)
                    except Exception:
                        question = value
                    if "retrieval_questions" in all_questions[field_name]:
                        all_questions[field_name]["retrieval_questions"].append(
                            question
                        )
                    else:
                        all_questions[field_name]["retrieval_questions"] = [question]
                    if key in paper_dependent_fields:
                        all_questions[field_name]["summarization_question"] = question
                else:
                    for arm in arms:
                        if arm in key:
                            field_name = key.replace(f"|_for:{arm}", "")
                            if field_name in all_questions:
                                try:
                                    question, _ = find_suggested_loc_from_query(value)
                                except Exception:
                                    question = value
                                if "retrieval_questions" in all_questions[field_name]:
                                    all_questions[field_name][
                                        "retrieval_questions"
                                    ].append(question)
                                else:
                                    all_questions[field_name]["retrieval_questions"] = [
                                        question
                                    ]
                            break

    for field_name in all_questions:
        if "retrieval_questions" not in all_questions[field_name]:
            all_questions[field_name]["retrieval_questions"] = [
                all_questions[field_name]["description"]
            ]
        else:
            all_questions[field_name]["retrieval_questions"].append(
                all_questions[field_name]["description"]
            )

    for question_info in questions:
        for key, value in question_info.items():
            if key.startswith("summarization"):
                continue
            else:
                if key in all_questions:
                    field_name = key
                    try:
                        _, suggested_location = find_suggested_loc_from_query(value)
                    except Exception:
                        all_questions[field_name]["location_info_added"] = True
                        continue
                    if not all_questions[field_name]["location_info_added"]:
                        location_info = f"""
                        Find all information from the following
                        locations in paper:{suggested_location}
                        """
                        location_info = re.sub(r"\s+", " ", location_info).strip()
                        all_questions[field_name]["retrieval_questions"].append(
                            location_info
                        )
                        all_questions[field_name]["location_info_added"] = True

    for field_name in all_questions:
        unique_retrieval_questions = []
        if "retrieval_questions" in all_questions[field_name]:
            for retrieval_question in all_questions[field_name]["retrieval_questions"]:
                if retrieval_question not in unique_retrieval_questions:
                    unique_retrieval_questions.append(retrieval_question)
            all_questions[field_name][
                "retrieval_questions"
            ] = unique_retrieval_questions

    return all_questions
