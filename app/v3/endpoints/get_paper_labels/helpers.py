import asyncio
import re

from anthropic import InternalServerError as AnthropicInternalServerError
from langchain_core.runnables import RunnableSerializable

from app.configs import settings
from app.exceptions.http import AnthropicServerFailed
from app.redis import redis_client
from app.utils.rag import get_context_docs
from app.utils.texts import combine_langchain_contexts, get_unique_langchain_contexts
from app.v3.endpoints.get_paper_labels.exceptions import QuestionRephrasingFailed
from app.v3.endpoints.get_paper_labels.logging import logger
from app.v3.endpoints.get_paper_labels.schemas import PaperLabelsTableField

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


async def rephrase_question(
    chain: RunnableSerializable, question: str, key=None
) -> str:

    question_key = f"rephrased_{question}"

    if question_key in redis_client:
        rephrased_question = redis_client.getex(question_key, ex=REDIS_LIVE_TIME)
    else:
        try:
            rephrased_question = await chain.ainvoke({"question": question})
            redis_client.setex(question_key, REDIS_LIVE_TIME, rephrased_question)
        except AnthropicInternalServerError:
            logger.exception("An error occurred at Anthropic's end.")
            raise AnthropicServerFailed()
        except Exception:
            logger.exception("An error occurred while rephrasing the question.")
            raise QuestionRephrasingFailed()
    if key:
        return {key: rephrased_question}
    return rephrased_question


def find_suggested_loc_from_query(
    query: str,
):
    rephrased_question, suggested_location = query.split(
        "Suggestion regarding where to find:"
    )
    return rephrased_question, suggested_location


async def get_questions_from_table_structure(
    chain: RunnableSerializable,
    table_structure: list[PaperLabelsTableField],
    paper_labels_in_table: list[str],
) -> dict[str, list[str]]:

    all_questions = {}
    invokes = []

    for field in table_structure:
        if field.name in paper_labels_in_table:
            all_questions[field.name] = {
                "description": field.description,
                "location_info_added": False,
            }

            field_info = field.name + ": " + field.description
            invokes.append(rephrase_question(chain, field_info, field.name))

    questions = await asyncio.gather(*invokes)

    for question_info in questions:
        for key, value in question_info.items():
            field_name = key
            try:
                question, suggested_location = find_suggested_loc_from_query(value)
            except Exception:
                question = value
                all_questions[field_name]["location_info_added"] = True
                suggested_location = "Not Found."

            if "retrieval_questions" in all_questions[field_name]:
                all_questions[field_name]["retrieval_questions"].append(question)
            else:
                all_questions[field_name]["retrieval_questions"] = [question]
            if not all_questions[field_name]["location_info_added"]:
                location_info = f"""
                Find all information from the following
                locations in paper:{suggested_location}
                """
                location_info = re.sub(r"\s+", " ", location_info).strip()
                all_questions[field_name]["retrieval_questions"].append(location_info)
                all_questions[field_name]["location_info_added"] = True
            all_questions[field_name]["summarization_question"] = question

    for field_name in all_questions:
        if "retrieval_questions" not in all_questions[field_name]:
            all_questions[field_name]["retrieval_questions"] = [
                all_questions[field_name]["description"]
            ]
        else:
            all_questions[field_name]["retrieval_questions"].append(
                all_questions[field_name]["description"]
            )

    return all_questions


async def get_contexts_from_rag(
    retrieval_questions: list[str],
    paper_id: str,
    project_id: str,
):
    invokes = []

    for question in retrieval_questions:
        if not isinstance(question, str):
            question = str(question)
        invoke = get_context_docs(
            query=question,
            flag_id=paper_id,
            project_id=project_id,
        )
        invokes.append(invoke)

    contexts = await asyncio.gather(*invokes)
    contexts = [c for context in contexts for c in context]
    contexts = get_unique_langchain_contexts(contexts)
    contexts = combine_langchain_contexts(contexts)
    return contexts


async def get_summarized_contexts(
    chain: RunnableSerializable,
    questions_with_contexts: dict,
):
    invokes = []
    for _, info in questions_with_contexts.items():
        contexts = info["contexts"]
        contexts_with_desc = f"""
        Field Description: {info['description']} \n\n {contexts}
        """
        contexts_with_desc = re.sub(r"\s+", " ", contexts_with_desc).strip()
        invokes.append(
            chain.ainvoke(
                {
                    "contexts": contexts_with_desc,
                    "question": info["summarization_question"],
                }
            )
        )
    all_contexts = await asyncio.gather(*invokes)

    for i, (_, info) in enumerate(questions_with_contexts.items()):
        info["contexts"] = all_contexts[i]

    return questions_with_contexts


def combine_contexts(
    questions_with_contexts: dict,
    field_names: list[str],
):
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

    return "\n".join(combined_contexts)
