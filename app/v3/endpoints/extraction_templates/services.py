from collections.abc import AsyncGenerator
from datetime import datetime

from langchain_core.messages import HumanMessage

from app.logging import logger
from app.utils.llms import get_message_text
from app.v3.endpoints.extraction_templates.agent_services.graph import get_graph
from app.v3.endpoints.extraction_templates.agent_services.schemas import (
    ExtractionOutputRow,
    UserInputRow,
)
from app.v3.endpoints.extraction_templates.schemas import (
    StreamChunk,
    StreamChunkMetadata,
)
from app.v3.endpoints.extraction_templates.utils import (
    prepare_template_schema,
    prepare_tool_response,
)


async def chat_extraction_template(
    user_input: str,
    thread_id: str,
    user_token: str,
    project_id: str,
    task_type: str = "template_creation",
    table_name: str | None = None,
    table_description: str | None = None,
    input_rows: list[UserInputRow] | None = None,
    output_rows: list[ExtractionOutputRow] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate response chunks for extraction template chat.
    Yields JSON-formatted strings for SSE streaming.
    """
    graph = await get_graph()

    state = {
        "messages": [HumanMessage(content=user_input)],
        "suggested_action": None,
        "project_id": project_id,
        "user_token": user_token,
        "task_type": task_type,
    }

    if table_name:
        state["table_name"] = table_name
    if table_description:
        state["table_description"] = table_description
    if input_rows:
        state["incoming_input_rows"] = input_rows
    if output_rows:
        state["incoming_output_rows"] = output_rows

    configs = {
        "recursion_limit": 50,
        "configurable": {"thread_id": thread_id},
        "run_name": "agent_task",
    }

    chunk_order = 0
    start_time = datetime.now().timestamp()
    final_state = None

    async for chunk, chunk_metadata in graph.astream(
        state, configs, stream_mode="messages"
    ):
        langgraph_step = chunk_metadata.get("langgraph_step")

        if chunk.content:
            chunk_order += 1

            chunk_type = chunk.type
            chunk_name = chunk.name
            content = chunk.content

            if chunk_type == "AIMessageChunk":
                content = get_message_text(chunk)
            elif chunk_type == "tool":
                content = prepare_tool_response(chunk_name, content, chunk)
            elif chunk_type == "system":
                continue
            else:
                raise ValueError(f"Unsupported chunk type: {chunk_type}")

            if not content.strip():
                continue

            if (not chunk_name) and chunk_type == "AIMessageChunk":
                chunk_name = "Delineate AI"

            latest_state = await graph.aget_state(configs)
            stream_chunk = StreamChunk(
                type=chunk_type,
                header=chunk_name,
                body_message=content,
                metadata=StreamChunkMetadata(
                    chunk_order=chunk_order,
                    step_id=langgraph_step,
                    active_time_seconds=datetime.now().timestamp() - start_time,
                    template_schema=prepare_template_schema(latest_state.values),
                ),
            )

            yield f"data: {stream_chunk.model_dump_json()}\n\n"

    final_state = await graph.aget_state(configs)
    if final_state and final_state.values.get("suggested_action"):
        chunk_order += 1
        suggested_action = final_state.values["suggested_action"]

        logger.info(f"Streaming suggested action: {suggested_action}")

        stream_chunk = StreamChunk(
            type="suggested_action",
            header="Suggested Action",
            body_message=suggested_action.reason,
            metadata=StreamChunkMetadata(
                chunk_order=chunk_order,
                step_id=0,
                active_time_seconds=datetime.now().timestamp() - start_time,
                action=suggested_action.action,
                template_schema=prepare_template_schema(final_state.values),
            ),
        )

        yield f"data: {stream_chunk.model_dump_json()}\n\n"
