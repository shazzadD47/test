import json
import re
from uuid import uuid4

import anthropic
import json_repair
from anthropic import RateLimitError as AnthropicRateLimitError
from celery.utils.log import get_task_logger
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langfuse import get_client, observe
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel

from app.configs import settings
from app.core.vector_store import VectorStore
from app.redis import redis_client
from app.utils.llms import batch_invoke_chain_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.iterative_autofill.chains import (
    prepare_contexts_given_qa_chain,
)
from app.v3.endpoints.iterative_autofill.configs import settings as iaf_settings
from app.v3.endpoints.iterative_autofill.constants import (
    MAX_RETRIES,
    SEPARATOR,
    TOP_K,
    llm,
    llm_claude,
    rephrase_chain,
)
from app.v3.endpoints.iterative_autofill.exceptions import (
    ContextSummarizationFailed,
    QuestionRephrasingFailed,
)
from app.v3.endpoints.iterative_autofill.helpers import (
    find_suggested_loc_from_query,
    format_reference_text,
    get_relation_info,
)
from app.v3.endpoints.iterative_autofill.prompts import (
    NESTED_LABEL_CONTEXT_INSTRUCTION,
    QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS,
    SYSTEM_PROMPT,
)

logger = get_task_logger("iterative_autofill")

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


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
        if len(set(selected_contexts)) < TOP_K:
            continue
        if count >= count_limit:
            break

    remaining_contexts = [
        context
        for context in contexts[-1][: count_limit // 2]
        if context not in selected_contexts
    ]
    selected_contexts.extend(remaining_contexts)
    unqiue_contexts = []
    for context in selected_contexts:
        if context not in unqiue_contexts:
            unqiue_contexts.append(context)
    return unqiue_contexts


@observe()
def get_context_docs_task(
    queries: list[str],
    top_k: int = 20,
    flag_id: str = None,
    project_id: str = None,
    langfuse_session_id: str = None,
    trace_flag: bool = False,
) -> list[str]:
    if trace_flag:
        if langfuse_session_id is None:
            langfuse_session_id = uuid4().hex
        langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    batch_size = iaf_settings.BATCH_SIZE
    filter_params = {}
    if flag_id:
        filter_params["flag_id"] = flag_id
    if project_id:
        filter_params["project_id"] = project_id

    filter_params["file_type"] = "document"

    retriever = VectorStore.get_retriever(
        search_kwargs={"filter": filter_params, "k": top_k}
    )
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            if trace_flag:
                similar_docs = batch_invoke_chain_with_retry(
                    chain=retriever,
                    input=queries,
                    config={
                        "max_concurrency": batch_size,
                        "callbacks": [langfuse_handler],
                    },
                )
            else:
                similar_docs = batch_invoke_chain_with_retry(
                    chain=retriever,
                    input=queries,
                    config={"max_concurrency": batch_size},
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
def rephrase_question_task(
    questions: list[str],
    keys: list[str] = None,
    langfuse_session_id: str = None,
) -> str:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    batch_size = iaf_settings.QUERY_REPHRASE_BATCH_SIZE
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
                config={
                    "max_concurrency": batch_size,
                    "callbacks": [langfuse_handler],
                },
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
        return only_rephrased_questions
    else:
        return [{key: rephrased_questions[key]} for key in keys]


def format_citations(
    input_text: str,
    chunk_numbers: list[int],
) -> str:
    # the follwoing pattern can match
    chunk_pattern = (
        r"(<cite>|<\/cite>)\s*?\[(\d+(?:-\d+)?"
        r"(?:,\s*\d+(?:-\d+)?)*)\]\s*?(<cite>|<\/cite>)"
    )
    range_pattern = r"(\d+-\d+)"
    # If chunk_numbers is empty, remove all citations
    if not chunk_numbers:
        return re.sub(chunk_pattern, "", input_text)

    # Find all citations in the text
    citations = re.finditer(chunk_pattern, input_text)
    citations = list(citations)

    if len(citations) == 0:
        # If there are no citations in the text but chunk_numbers is not empty,
        # add all citations at the end of the text
        citation_text = (
            "[" + ", ".join(str(i + 1) for i in range(len(chunk_numbers))) + "]"
        )
        return input_text + " " + citation_text

    # Create a mapping from original chunk numbers to new sequential numbers
    chunk_map = {
        str(chunk): str(i + 1) for i, chunk in enumerate(sorted(chunk_numbers))
    }
    # Replace each citation with the new sequential numbers
    for citation in citations:
        original_citation = citation.group(0)
        citation_nums = citation.group(2)
        all_new_chunk_numbers = []
        for range_match in re.finditer(range_pattern, citation_nums):
            range_nums = range_match.group(0)
            range_nums = [num.strip() for num in range_nums.split("-")]
            start = int(range_nums[0])
            end = int(range_nums[-1])
            if all(str(i) in chunk_map for i in range(start, end + 1)):
                all_new_chunk_numbers.append(
                    f"{chunk_map[str(start)]}-{chunk_map[str(end)]}"
                )
        without_range_nums = re.sub(range_pattern, "", citation_nums)
        without_range_nums = re.sub(r",", "", without_range_nums)
        without_range_nums = [num.strip() for num in without_range_nums.split()]
        without_range_nums = [
            chunk_map[num] for num in without_range_nums if num in chunk_map
        ]
        all_new_chunk_numbers.extend(without_range_nums)
        all_new_chunk_numbers = [num.strip() for num in all_new_chunk_numbers]
        if len(all_new_chunk_numbers) > 0:
            formatted_citation = ",".join(all_new_chunk_numbers)
            formatted_citation = f"[{formatted_citation}]"
            input_text = input_text.replace(original_citation, formatted_citation)
        else:
            input_text = input_text.replace(original_citation, "")
    return input_text


def correct_answer_format(
    results: list[dict],
    splitted_contexts: list[list[str]],
    title: str = None,
) -> list[dict]:
    for i, (context, each_result) in enumerate(zip(splitted_contexts, results)):
        chunk_numbers = each_result["chunk_numbers"]
        references = "References:\n"
        processed_chunks = []
        processed_chunk_numbers = []
        for chunk_number in chunk_numbers:
            try:
                used_chunk = context[chunk_number - 1]
                # remove the order number 1.,2.,3.
                used_chunk = re.sub(r"^\d+\.\s*", "", used_chunk)
            except Exception as e:
                logger.exception(f"Error occured when removing order number: {e}")
                logger.info(f"Chunk number: {chunk_number}")
                logger.info(f"Context length: {len(context)}")
                logger.info(f"Result: {each_result}")
                continue
            formatted_chunk = ""
            try:
                used_chunk_dict = json_repair.loads(used_chunk)
                if isinstance(used_chunk_dict, str):
                    formatted_chunk += f"{format_reference_text(used_chunk)}\n"
                else:
                    if "paper_section" in used_chunk_dict:
                        section_name = used_chunk_dict["paper_section"]
                        if not (
                            title is not None
                            and isinstance(title, str)
                            and title.lower().strip() in section_name.lower().strip()
                        ):
                            formatted_chunk += f"{section_name}: "
                    if "text" in used_chunk_dict:
                        formatted_chunk += (
                            f"{format_reference_text(used_chunk_dict['text'])}\n"
                        )
                    else:
                        formatted_chunk += f"{format_reference_text(used_chunk)}\n"
            except Exception:
                logger.exception("Error occured when formatting chunk")
                continue
            processed_chunks.append(formatted_chunk)
            processed_chunk_numbers.append(chunk_number)
        if len(processed_chunks) > 0:
            each_result["references"] = references + "\n".join(
                [f"{i+1}. {processed_chunks[i]}" for i in range(len(processed_chunks))]
            )
        each_result["chunk_numbers"] = processed_chunk_numbers

    for each_result in results:
        formatted_answer = each_result["answer"]
        try:
            formatted_answer = format_citations(
                formatted_answer, each_result["chunk_numbers"]
            )
        except Exception as e:
            logger.exception(f"Error occured when formatting answer: {e}")
            if len(each_result["chunk_numbers"]) > 0:
                references = each_result["references"].split("\n")
                references = [
                    re.sub(r"^\d+\.\s*", "", reference) for reference in references
                ]
                references = [
                    f"{chunk_number}. {reference}"
                    for chunk_number, reference in zip(
                        each_result["chunk_numbers"], references
                    )
                ]
                references = "\n".join(references)
                each_result["references"] = references

        if len(each_result["chunk_numbers"]) > 0:
            formatted_answer = f"{formatted_answer}\n{each_result['references']}"
        each_result["answer"] = formatted_answer
    return results


@observe()
def context_summarization_task(
    contexts: list[str],
    context_lengths: list[int],
    questions: list[str],
    parent_label_answers: list[str] = None,
    has_parents: bool = False,
    output_schema: type[BaseModel] = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)
    chain = prepare_contexts_given_qa_chain(
        llm=llm,
        has_parents=has_parents,
        output_schema=output_schema,
    )
    chain_claude = prepare_contexts_given_qa_chain(
        llm=llm_claude,
        has_parents=has_parents,
        output_schema=output_schema,
    )
    retry_count = 0
    result = contexts.copy()
    batch_size = iaf_settings.CONTEXT_QA_BATCH_SIZE
    while retry_count < MAX_RETRIES:
        try:
            try:
                if has_parents:
                    result = batch_invoke_chain_with_retry(
                        chain=chain,
                        input=[
                            {
                                "contexts": contexts[i],
                                "total_number_of_chunks": context_lengths[i],
                                "questions": questions[i],
                                "parent_label_answers": parent_label_answers[i],
                            }
                            for i in range(len(contexts))
                        ],
                        config={
                            "max_concurrency": batch_size,
                            "callbacks": [langfuse_handler],
                        },
                    )
                else:
                    result = batch_invoke_chain_with_retry(
                        chain=chain,
                        input=[
                            {
                                "contexts": contexts[i],
                                "questions": questions[i],
                                "total_number_of_chunks": context_lengths[i],
                            }
                            for i in range(len(contexts))
                        ],
                        config={
                            "max_concurrency": batch_size,
                            "callbacks": [langfuse_handler],
                        },
                    )
            except Exception:
                if has_parents:
                    result = batch_invoke_chain_with_retry(
                        chain=chain_claude,
                        input=[
                            {
                                "contexts": contexts[i],
                                "questions": questions[i],
                                "parent_label_answers": parent_label_answers[i],
                                "total_number_of_chunks": context_lengths[i],
                            }
                            for i in range(len(contexts))
                        ],
                        config={
                            "max_concurrency": batch_size,
                            "callbacks": [langfuse_handler],
                        },
                    )
                else:
                    result = batch_invoke_chain_with_retry(
                        chain=chain_claude,
                        input=[
                            {
                                "contexts": contexts[i],
                                "questions": questions[i],
                                "total_number_of_chunks": context_lengths[i],
                            }
                            for i in range(len(contexts))
                        ],
                        config={
                            "max_concurrency": batch_size,
                            "callbacks": [langfuse_handler],
                        },
                    )
            result = [answer.dict() for answer in result]
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
def get_summarized_contexts(
    questions_with_contexts: dict,
    has_parents: bool = False,
    parent_label_answers: list[str] = None,
    title: str = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    setup_langfuse_handler(langfuse_session_id)

    all_contexts = []
    all_context_lengths = []
    all_questions = []
    if has_parents:
        all_parent_label_answers = []

    idx_mapping = {}
    count = 0
    for i, (field_name, info) in enumerate(questions_with_contexts.items()):
        if info["contexts"].strip() == "":
            continue
        idx_mapping[i] = count
        count += 1
        all_contexts.append(info["contexts"])
        all_context_lengths.append(info["context_length"])
        all_questions.append(field_name + ": " + info["summarization_question"])
        if has_parents:
            all_parent_label_answers.append(info["parent_label_answers"])

    if has_parents:
        all_summarized_contexts = context_summarization_task(
            contexts=all_contexts,
            context_lengths=all_context_lengths,
            questions=all_questions,
            has_parents=has_parents,
            parent_label_answers=all_parent_label_answers,
            langfuse_session_id=langfuse_session_id,
        )
    else:
        all_summarized_contexts = context_summarization_task(
            contexts=all_contexts,
            context_lengths=all_context_lengths,
            questions=all_questions,
            langfuse_session_id=langfuse_session_id,
        )
    splitted_contexts = [context.split(SEPARATOR) for context in all_contexts]
    all_summarized_contexts = correct_answer_format(
        all_summarized_contexts,
        splitted_contexts,
        title=title,
    )
    for i, (_, info) in enumerate(questions_with_contexts.items()):
        if i in idx_mapping:
            info["contexts"] = all_summarized_contexts[idx_mapping[i]]["answer"]
            info["answer_found"] = all_summarized_contexts[idx_mapping[i]][
                "answer_found"
            ]
            info["chunk_numbers"] = all_summarized_contexts[idx_mapping[i]][
                "chunk_numbers"
            ]
        else:
            info["contexts"] = ""
            info["answer_found"] = False
            info["chunk_numbers"] = []

    return questions_with_contexts


@observe(as_type="generation")
def get_summarized_contexts_from_full_paper(
    questions_with_summarized_contexts: dict,
    labels_to_extract: list[str],
    full_paper: str,
    total_full_paper_chunks: int,
    table_structure_hash: dict,
    output_schema: type[BaseModel] = None,
    parser: PydanticOutputParser = None,
    prev_response: list[dict] = None,
    has_parents: bool = False,
    title: str = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    setup_langfuse_handler(langfuse_session_id)

    logger.info("Extracting contexts from full paper")
    logger.info(f"labels_to_extract: {labels_to_extract}")

    all_prompts = []
    all_parent_label_answers = []
    for label in labels_to_extract:
        prompt = (
            label
            + ": "
            + questions_with_summarized_contexts[label]["summarization_question"]
        )
        all_prompts.append(prompt)
        if (
            table_structure_hash.get(label, {}).get("relationships", []) is not None
            and len(table_structure_hash.get(label, {}).get("relationships", [])) > 0
        ):
            all_parent_label_answers.append(
                label
                + ": \n"
                + get_relation_info(
                    table_structure_hash[label]["relationships"],
                    prev_response,
                    table_structure_hash,
                )
            )
        else:
            all_parent_label_answers.append(f"{label}: No parent label answers found.")
    all_parent_label_answers = [
        f"{i+1}. {parent_label_answer}"
        for i, parent_label_answer in enumerate(all_parent_label_answers)
    ]
    all_parent_label_answers = "\n".join(all_parent_label_answers)

    all_prompts = [f"{i+1}. {prompt}" for i, prompt in enumerate(all_prompts)]

    all_prompts = "\n".join(all_prompts)

    # first try with antropic
    initial_prompt = f"""
    Find the answers to the following prompts using the given
    contexts. \n Prompts: {all_prompts} \n
    """
    additional_instructions = QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS.format(
        total_number_of_chunks=total_full_paper_chunks
    )
    output_instructions = parser.get_format_instructions()
    output_format = f"""
    Output Format: {output_instructions}
    \n {additional_instructions}
    """
    nested_label_instruction = NESTED_LABEL_CONTEXT_INSTRUCTION.format(
        parent_label_answers=all_parent_label_answers
    )
    client = anthropic.Anthropic(
        api_key=settings.ANTHROPIC_API_KEY,
        max_retries=MAX_RETRIES,
    )
    user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": initial_prompt,
                },
                {
                    "type": "text",
                    "text": f"Contexts: {full_paper}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": output_format,
                },
            ],
        }
    ]
    if has_parents:
        user_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": nested_label_instruction,
                    }
                ],
            }
        )
    langfuse_client = get_client()
    langfuse_client.update_current_observation(
        input=user_messages,
        model=iaf_settings.CLAUDE_MODEL_ID,
    )
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = client.messages.create(
                model=iaf_settings.CLAUDE_MODEL_ID,
                max_tokens=iaf_settings.CLAUDE_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    },
                ],
                messages=user_messages,
            )
            langfuse_client.update_current_observation(
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                }
            )
            try:
                all_summarized_contexts = parser.parse(response.content[0].text)
                all_summarized_contexts = all_summarized_contexts.dict()
            except Exception as e:
                logger.error(f"Error in parsing full paper contexts: {e}")
                try:
                    all_summarized_contexts = JsonOutputParser().parse(
                        response.content[0].text
                    )
                    all_summarized_contexts = json_repair.loads(
                        json.dumps(all_summarized_contexts)
                    )
                    all_summarized_contexts = parser.parse(
                        json.dumps(all_summarized_contexts)
                    )
                    all_summarized_contexts = all_summarized_contexts.dict()
                except Exception as e:
                    logger.error(f"Error in parsing full paper contexts: {e}")
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        raise Exception(f"Error in parsing full paper contexts: {e}")
                continue
            return all_summarized_contexts
        except Exception as e:
            logger.error(f"Error in claude processing full paper contexts: {e}")
            # second try with openai
            all_summarized_contexts = context_summarization_task(
                contexts=[full_paper],
                context_lengths=[total_full_paper_chunks],
                questions=[all_prompts],
                has_parents=has_parents,
                parent_label_answers=all_parent_label_answers,
                output_schema=output_schema,
                langfuse_session_id=langfuse_session_id,
            )[0]
            return all_summarized_contexts


@observe()
def format_full_paper_contexts(
    full_paper: str,
    labels_to_extract: list[str],
    all_summarized_contexts: dict,
    title: str = None,
    langfuse_session_id: str = None,
):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_client = get_client()
    langfuse_client.update_current_trace(session_id=langfuse_session_id)
    summarized_context_list = [result for _, result in all_summarized_contexts.items()]
    splitted_contexts = [
        full_paper.split(SEPARATOR) for i in range(len(labels_to_extract))
    ]
    formatted_summarized_contexts = correct_answer_format(
        summarized_context_list,
        splitted_contexts,
        title=title,
    )
    for i, label in enumerate(labels_to_extract):
        formatted_answer = formatted_summarized_contexts[i]["answer"]
        all_summarized_contexts[label]["answer"] = formatted_answer

    questions_with_contexts = {}
    for label in labels_to_extract:
        questions_with_contexts[label] = {
            "contexts": all_summarized_contexts[label]["answer"],
            "answer_found": all_summarized_contexts[label]["answer_found"],
            "chunk_numbers": all_summarized_contexts[label]["chunk_numbers"],
        }

    return questions_with_contexts


@observe()
def get_questions_from_table_structure(
    table_structure_hash: dict[str, dict],
    labels_to_extract: list[str],
    non_dependent_label_answers: list[dict] = None,
    langfuse_session_id: str = None,
) -> dict[str, list[str]]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    setup_langfuse_handler(langfuse_session_id)

    all_questions = {}
    rephrase_input = {
        "questions": [],
        "keys": [],
    }
    for label in labels_to_extract:
        if label in table_structure_hash:
            all_questions[label] = {
                "description": table_structure_hash[label]["description"],
                "location_info_added": False,
            }
            field_info = label + ": " + table_structure_hash[label]["description"]
            if label not in rephrase_input["keys"]:
                rephrase_input["keys"].append(label)
                rephrase_input["questions"].append(field_info)
            if (
                non_dependent_label_answers is not None
                and table_structure_hash.get(label, {}).get("relationships", [])
                is not None
                and len(table_structure_hash.get(label, {}).get("relationships", []))
                > 0
            ):
                answers_of_parent_labels = []
                related_label_names = list(
                    {
                        relationship["related_label"]
                        for relationship in table_structure_hash[label]["relationships"]
                    }
                )

                for labels in non_dependent_label_answers:
                    single_answer_of_parent_label = []
                    for related_label in related_label_names:
                        if related_label in labels:
                            single_answer_of_parent_label.append(
                                f"{related_label}:{labels[related_label]}"
                            )
                    single_answer_of_parent_label = ", ".join(
                        single_answer_of_parent_label
                    )
                    answers_of_parent_labels.append(single_answer_of_parent_label)

                answers_of_parent_labels = list(set(answers_of_parent_labels))

                if len(answers_of_parent_labels) > 0:
                    for answers in answers_of_parent_labels:
                        field_info = (
                            label
                            + ": "
                            + table_structure_hash[label]["description"]
                            + " for "
                            + answers
                        )
                        field_key = label + f"|_for: {answers}"
                        if field_key not in rephrase_input["keys"]:
                            rephrase_input["keys"].append(field_key)
                            rephrase_input["questions"].append(field_info)

    questions = rephrase_question_task(
        questions=rephrase_input["questions"],
        keys=rephrase_input["keys"],
        langfuse_session_id=langfuse_session_id,
    )

    for question_info in questions:
        for key, value in question_info.items():
            if key in table_structure_hash:
                try:
                    question, _ = find_suggested_loc_from_query(value)
                except Exception:
                    question = value
                if "retrieval_questions" not in all_questions[key]:
                    all_questions[key]["retrieval_questions"] = []
                all_questions[key]["retrieval_questions"].append(question)
                all_questions[key]["summarization_question"] = question
            else:
                field_name = key
                if "|_for:" in field_name:
                    field_name = field_name[: field_name.rfind("|_for:")]
                try:
                    question, _ = find_suggested_loc_from_query(value)
                except Exception:
                    question = value
                if "retrieval_questions" in all_questions[field_name]:
                    all_questions[field_name]["retrieval_questions"].append(question)
                else:
                    all_questions[field_name]["retrieval_questions"] = [question]

    for field_name in all_questions:
        if "retrieval_questions" not in all_questions[field_name]:
            all_questions[field_name]["retrieval_questions"] = []
        all_questions[field_name]["retrieval_questions"].append(
            all_questions[field_name]["description"]
        )

    for question_info in questions:
        for key, value in question_info.items():
            field_name = key
            if "|_for:" in field_name:
                field_name = field_name[: field_name.rfind("|_for:")]
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
                if "retrieval_questions" not in all_questions[field_name]:
                    all_questions[field_name]["retrieval_questions"] = []
                all_questions[field_name]["retrieval_questions"].append(location_info)
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


@observe()
def extract_title(
    flag_id: str,
    langfuse_session_id: str = None,
) -> str | None:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_client = get_client()
    langfuse_client.update_current_trace(session_id=langfuse_session_id)
    chunks = get_context_docs_task(
        [
            "What is the title of the paper?",
        ],
        flag_id=flag_id,
        langfuse_session_id=langfuse_session_id,
        trace_flag=True,
    )
    chunks = chunks[0]
    title_prefix = "The Title of the paper is:".lower()
    for chunk in chunks:
        try:
            formatted_chunk = chunk.lower().strip()
            if title_prefix in formatted_chunk:
                title = formatted_chunk.split(title_prefix)[1].strip()
                return title
        except Exception:
            logger.exception("Error occured when extracting title")
            continue
    return None
