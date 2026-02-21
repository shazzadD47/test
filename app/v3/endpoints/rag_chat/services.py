import json
from collections.abc import Generator

import tiktoken
from langchain_community.document_transformers import LongContextReorder
from langchain_core.messages import HumanMessage, SystemMessage

from app.configs import settings
from app.core.auto import AutoChatModel
from app.core.vector_store import VectorStore
from app.v3.endpoints.rag_chat.configs import settings as rag_chat_settings
from app.v3.endpoints.rag_chat.logging import logger
from app.v3.endpoints.rag_chat.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.v3.endpoints.rag_chat.schema import ContextInfo, SourceInfo, TokenInfo

reordering = LongContextReorder()
previous_messages: list[str] = []


def generate_response_chunks(
    message: str,
    project_id: str | None,
    flag_id: str | None,
    user_id: str | None,
    title: str | None,
    description: str | None,
    system_prompt: str | None = None,
) -> Generator[str, None, None]:
    try:
        contexts = retrieve_contexts(
            project_id, flag_id, user_id, title, description, message
        )
        message = message.strip() + " search table for parameters/variables values"
        human_query = USER_PROMPT_TEMPLATE.format(contexts=contexts, message=message)

        # Combine the existing SYSTEM_PROMPT with the optional system_prompt
        combined_system_prompt = SYSTEM_PROMPT
        if system_prompt:
            combined_system_prompt += f"\n\nAdditional Instructions: {system_prompt}"

        messages = [
            SystemMessage(content=combined_system_prompt),
            HumanMessage(content=human_query),
        ]

        encoding = tiktoken.encoding_for_model(settings.GPT_4_TEXT_MODEL)
        input_tokens = sum(len(encoding.encode(msg.content)) for msg in messages)

        llm = AutoChatModel.from_model_name(
            rag_chat_settings.LLM_NAME,
            temperature=0.1,
            max_tokens=1400,
            streaming=True,
        )

        generated_tokens = 0
        for chunk in llm.stream(messages):
            content = chunk.content
            if content:
                tokens = len(encoding.encode(content))
                generated_tokens += tokens
                yield content

        yield "😊"
        yield json.dumps({"source_info": ContextInfo(context=contexts).dict()})
        yield "🔥"
        yield json.dumps(
            TokenInfo(
                input_tokens=input_tokens,
                usd_input_cost=settings.GPT_4O_INPUT_TOKEN * input_tokens,
                total_generated_tokens=generated_tokens,
                usd_output_cost=settings.GPT_4O_OUTPUT_TOKEN * generated_tokens,
            ).dict()
        )

    except Exception as e:
        logger.exception(f"Error in generate_response_chunks: {e}")


def retrieve_contexts(
    project_id: str | None,
    flag_id: str | None,
    user_id: str | None,
    title: str | None,
    description: str | None,
    message: str,
) -> list[SourceInfo]:
    try:
        prefilter = (
            {"flag_id": flag_id}
            if flag_id
            else ({"user_id": user_id} if user_id else {"project_id": project_id})
        )

        retriever = VectorStore.get_retriever(search_kwargs={"filter": prefilter})

        results = retriever.invoke(message)
        results = reordering.transform_documents(results)

        contexts = []
        if title:
            contexts.append(
                SourceInfo(
                    page_content=f"Project Title: {title}",
                    page=None,
                    flag_id=None,
                    title=None,
                )
            )
        if description:
            contexts.append(
                SourceInfo(
                    page_content=f"Project Description: {description}",
                    page=None,
                    flag_id=None,
                    title=None,
                )
            )

        for result in results:
            try:
                metadata = result.metadata
                contexts.append(
                    SourceInfo(
                        page_content=result.page_content,
                        page=metadata.get("page"),
                        flag_id=metadata.get("flag_id"),
                        title=metadata.get("title"),
                    )
                )
            except Exception as e:
                logger.exception(f"Error in processing result: {e}")

        previous_messages.append(message)
        if len(previous_messages) > 4:
            previous_messages.pop(0)

        context_with_prev_msgs = "Previous User Messages:\n"
        for prev_msg in previous_messages[:-1]:
            context_with_prev_msgs += f"User: {prev_msg}\n"
        contexts.append(
            SourceInfo(
                page_content=context_with_prev_msgs,
                page=None,
                flag_id=None,
                title=None,
            )
        )

        return contexts

    except Exception as e:
        logger.exception(f"Error in retrieve_contexts: {e}")
