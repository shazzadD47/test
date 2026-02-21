import asyncio
import json
from itertools import product

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field, create_model

from app.chains import (
    prepare_context_given_summarization_chain,
    prepare_question_generation_chain,
)
from app.configs import settings
from app.logging import logger
from app.redis import redis_client
from app.v2.endpoints.plot_digitizer.services.utils import async_get_table_contents
from app.v3.endpoints.autofill.constants import LANGSMITH_TAGS, paper_dependent_fields
from app.v3.endpoints.autofill.schemas import MetaAnalysisAutofillRequest
from app.v3.endpoints.autofill.utils.chains import prepare_information_extraction_chain
from app.v3.endpoints.autofill.utils.contexts import (
    get_contexts_from_generated_questions,
)

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60

logger = logger.getChild("autofill")
d_type_map = {"integer": int, "float": float, "string": str, "number": float}


async def generate_questions(
    chain: RunnableSerializable, question: str, output_instructions: str
) -> list[str]:
    question_key = f"generate_questions_{question}"

    if redis_client.exists(question_key):
        questions = redis_client.lrange(question_key, 0, -1)
        redis_client.expire(question_key, REDIS_LIVE_TIME)
    else:
        try:
            generated_questions = await chain.ainvoke(
                {"question": question, "output_format": output_instructions}
            )
            questions = [question] + generated_questions.questions

            redis_client.rpush(question_key, *questions)
            redis_client.expire(question_key, REDIS_LIVE_TIME)
        except Exception:
            logger.exception("An error occurred while generating questions.")
            questions = [question]

    return questions


async def autofill_meta_analysis_data(payload: MetaAnalysisAutofillRequest):
    table_contents = await async_get_table_contents(
        payload.paper_id, payload.project_id
    )
    table_contents = "\n---------------\n\n".join(table_contents)

    if not payload.is_root:
        choices_with_type = []
        for root in payload.root_choices:
            choices = [f"{root.name}:{value}" for value in root.values]

            if choices:
                choices_with_type.append(choices)

        choice_tuples = list(product(*choices_with_type))
    else:
        choice_tuples = None

    class GeneratedQuestionSchema(BaseModel):
        original_question: str = Field(..., description="Original question")
        questions: list[str] = Field(..., description="Generated questions")

    generated_question_parser = PydanticOutputParser(
        pydantic_object=GeneratedQuestionSchema
    )
    question_generation_chain = prepare_question_generation_chain(
        generated_question_parser, tags=LANGSMITH_TAGS
    )

    if payload.is_root:
        questions = [
            {"name": field.name, "description": field.description}
            for field in payload.table_structure
        ]
    else:
        roots = {choice.name for choice in payload.root_choices}
        roots.add("arm")

        questions = []

        for field in payload.table_structure:
            if (field.name in roots) or (
                field.name.lower().strip() in paper_dependent_fields
            ):
                questions.append({"name": field.name, "description": field.description})
                continue

            for choice_tuple in choice_tuples:
                choice_tuple_str = ", ".join(choice_tuple)
                updated_description = f"{field.description} for {choice_tuple_str}"
                questions.append(
                    {"name": field.name, "description": updated_description}
                )

    invokes = []
    question_output_instructions = generated_question_parser.get_format_instructions()
    for field in questions:
        invoke = generate_questions(
            question_generation_chain,
            field["description"],
            question_output_instructions,
        )
        invokes.append(invoke)

    questions = await asyncio.gather(*invokes)

    context_chain = prepare_context_given_summarization_chain(tags=LANGSMITH_TAGS)
    contexts, sources = await get_contexts_from_generated_questions(
        context_chain,
        questions,
        payload.project_id,
        payload.paper_id,
        payload.is_root,
        table_contents,
        return_sources=True,
        tags=LANGSMITH_TAGS,
    )
    contexts = "\n---------------\n\n".join(contexts)

    labels = {
        field.name: (
            (
                list[d_type_map[field.d_type]]
                if payload.is_root
                else (
                    d_type_map[field.d_type]
                    if field.d_type == "string"
                    else d_type_map[field.d_type] | None
                )
            ),
            Field(..., description=field.description),
        )
        for field in payload.table_structure
    }

    DetailsSchema = create_model("ArmDetails", **labels)

    if not payload.is_root:

        class DetailsSchema(BaseModel):
            data: list[DetailsSchema]

    parser = PydanticOutputParser(pydantic_object=DetailsSchema)
    extraction_chain = prepare_information_extraction_chain(parser, payload.is_root)

    if payload.is_root:
        inputs = {
            "contexts": contexts,
            "output_format": parser.get_format_instructions(),
        }
    else:
        inputs = {
            "contexts": contexts,
            "output_format": parser.get_format_instructions(),
            "choice_tuples": choice_tuples,
        }

    try:
        result = await extraction_chain.ainvoke(inputs)
        result = json.loads(result.json())

        if result.get("data"):
            result = result["data"]
    except Exception:
        logger.exception("Failed to extract information")

    return result, sources
