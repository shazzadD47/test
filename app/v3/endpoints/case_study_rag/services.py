import json
from collections.abc import Generator

import httpx
import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from app.configs import settings
from app.core.auto import AutoChatModel
from app.integrations.backend import generate_api_token
from app.v3.endpoints.case_study_rag.configs import settings as case_study_settings
from app.v3.endpoints.case_study_rag.logging import logger
from app.v3.endpoints.case_study_rag.prompts import SYSTEM_PROMPT
from app.v3.endpoints.case_study_rag.utils import get_plot_context, process_items


def get_case_study_items_list(case_study_id: str) -> list[dict]:
    token = generate_api_token(settings.BACKEND_SECRET)
    headers = {"x-api-key": f"{settings.BACKEND_KEY}###{token}"}

    try:
        url = f"{settings.BACKEND_BASE_URL}/case-study/by-id/{case_study_id}"
        response = httpx.get(url, headers=headers)
        return response.json()
    except Exception as e:
        logger.exception(f"failed to fetch case study items from backend: {e}")
        return []


def generate_response_chunks(
    message: str, case_study_id: str, project_id: str
) -> Generator[str, None, None]:
    try:
        items = get_case_study_items_list(case_study_id)
    except Exception as e:
        yield f"Error: {str(e)}"
        return

    context = process_items(items)
    plot_context = get_plot_context(items, project_id)
    context = context + "\nPlot Context:\n" + str(plot_context) + "\n\n"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context: {context}\n\n{message}\n\n"),
    ]

    encoding = tiktoken.encoding_for_model(settings.GPT_4_TEXT_MODEL)
    generated_tokens = 0
    input_tokens = sum(len(encoding.encode(msg.content)) for msg in messages)

    llm = AutoChatModel.from_model_name(
        case_study_settings.LLM_NAME,
        temperature=0.1,
        max_tokens=1400,
        streaming=True,
    )

    for chunk in llm.stream(messages):
        if chunk.content is not None:
            content = chunk.content
            tokens = len(encoding.encode(content))
            generated_tokens += tokens
            yield content

    source_info = {"context": []}

    yield "😊"
    total_input_token_cost = settings.GPT_4O_INPUT_TOKEN * input_tokens
    total_output_token_cost = settings.GPT_4O_OUTPUT_TOKEN * generated_tokens
    yield json.dumps({"source_info": source_info})
    yield "🔥"
    yield json.dumps(
        {
            "input_tokens": input_tokens,
            "usd_input_cost": total_input_token_cost,
            "total_generated_tokens": generated_tokens,
            "usd_output_cost": total_output_token_cost,
        }
    )
