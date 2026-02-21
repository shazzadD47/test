from typing import Annotated

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.v3.endpoints.report_generator.logging import logger
from app.v3.endpoints.report_generator.schema import ReportGenerationResponse
from app.v3.endpoints.report_generator.services.chat import (
    report_chat,
)
from app.v3.endpoints.report_generator.utils import is_valid_report_request

router = APIRouter(tags=["Report Generator"])


@router.websocket("/ws/report-generator/agents/{thread_id}")
async def report_generator_ws(
    websocket: WebSocket,
    thread_id: Annotated[
        str, "A thread ID that uniquely identifies a report generation session"
    ],
):
    await websocket.accept()

    try:
        request = await websocket.receive_json()

        logger.debug(
            f"Report generation request received: thread_id: {thread_id}, "
            f"request: {request}"
        )

        if not is_valid_report_request(request):
            response = ReportGenerationResponse(
                node="error",
                node_type="error",
                generation_type=request.get("generation_type", "ai_assistant"),
                process_description="Invalid request format",
            )
            await websocket.send_json(response.model_dump())
        else:
            await report_chat(websocket, thread_id, request)
    except WebSocketDisconnect:
        logger.info("Report generation WebSocket disconnected")
    except Exception as e:
        logger.exception(f"Error in report generation WebSocket: {str(e)}")
    finally:
        await websocket.close()
