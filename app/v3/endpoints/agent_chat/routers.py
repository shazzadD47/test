from typing import Annotated

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.v3.endpoints.agent_chat.logging import logger
from app.v3.endpoints.agent_chat.schema import (
    AgentChatResponse,
    GenerateTitleRequest,
    GenerateTitleResponse,
)
from app.v3.endpoints.agent_chat.services.chat import chat
from app.v3.endpoints.agent_chat.services.chat_room_naming import (
    generate_chat_title_with_llm,
)
from app.v3.endpoints.agent_chat.utils import is_valid_request

router = APIRouter(tags=["Agent Chat"])


@router.websocket("/ws/chat/agents/{thread_id}")
async def agent_chat_ws(
    websocket: WebSocket,
    thread_id: Annotated[str, "A thread ID that uniquely identifies a conversation"],
):
    await websocket.accept()

    try:
        request = await websocket.receive_json()

        logger.debug(
            "Chat request received: thread_id: %s, request: %s", thread_id, request
        )

        if not is_valid_request(request):
            response = AgentChatResponse(
                node="error",
                node_type="error",
                process_description="Invalid request",
            )
            await websocket.send_json(response.model_dump())
        else:
            await chat(websocket, thread_id, request)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await websocket.close()


@router.post("/chat/generate-title", response_model=GenerateTitleResponse)
async def generate_title(request: GenerateTitleRequest):
    """
    Generate a concise title for a chat conversation based on the user's query.

    Uses an LLM (GPT-4o-mini) to create a 3-5 word title that summarizes
    the conversation topic. Falls back to intelligent truncation if LLM fails.

    Returns a title suitable for displaying in chat history.
    """
    logger.debug("Generate title request: %s", request.query)

    title = await generate_chat_title_with_llm(request.query)

    logger.info("Generated title: %s", title)

    return GenerateTitleResponse(title=title)
