import random
from asyncio import sleep
from collections.abc import AsyncGenerator
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langfuse.langchain import CallbackHandler

from app.configs import settings
from app.exceptions.system import (
    AnthropicBadRequestError,
    OpenAIBadRequestError,
)
from app.v3.endpoints.report_generator.constants import (
    ReportAgents,
)
from app.v3.endpoints.report_generator.logging import logger
from app.v3.endpoints.report_generator.schema import (
    ReportGenerationResponse,
    ReportState,
)
from app.v3.endpoints.report_generator.services.graph import get_report_graph


def extract_message_content(message: AIMessage) -> str:
    """Extract content from AI message, handling both string and list formats."""
    content = message.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    text_parts.append(block["content"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    else:
        return str(content) if content is not None else ""


async def prepare_report_output(
    state: ReportState,
) -> AsyncGenerator[ReportGenerationResponse, None]:
    """Prepare report generation output for streaming."""
    messages = state["messages"]
    generation_type = state.get("generation_type", "ai_assistant")
    ai_message: AIMessage | None = None

    for message in messages[-1::-1]:
        if isinstance(message, AIMessage):
            ai_message = message
            break

    if (
        state.get("sender") == ReportAgents.ASSISTANT_AGENT.value
        and state.get("sources")
        and generation_type == "ai_assistant"
    ):
        yield ReportGenerationResponse(
            node="context_retrieval",
            node_type="tool",
            generation_type=generation_type,
            header="Retrieving Research Context",
            body_message=(
                "Gathering relevant information from research papers " "and files..."
            ),
            sources=state.get("sources", []),
        )
        await sleep(random.uniform(0.5, 1.0))  # nosec B311

    if state.get("tool_calls"):
        for tool_call in state.get("tool_calls"):
            tool_name = tool_call.get("name")
            if not tool_name:
                continue

            query = tool_call.get("args", {}).get("query", "")
            yield ReportGenerationResponse(
                node=tool_name,
                node_type="tool",
                generation_type=generation_type,
                header=f"Searching {tool_name.replace('_', ' ').title()}",
                body_message=f"Looking up: {query}",
                sources=state.get("sources", []),
            )

    if ai_message:
        # header = "AI Assistant" if generation_type == "ai_assistant" else "AI Edit"
        if generation_type == "ai_insights":
            header = "AI Insights"
        elif generation_type == "ai_assistant":
            header = "AI Assistant"
        else:
            header = "AI Edit"

        content = extract_message_content(ai_message)

        yield ReportGenerationResponse(
            node="agent",
            node_type="agent",
            generation_type=generation_type,
            header=header,
            body_message=content,
            sources=state.get("sources", []),
            is_complete=True,
        )


async def report_chat(websocket: WebSocket, thread_id: str, request: dict):
    """Main chat function for report generation using graph workflow."""
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "report_generation",
        "callbacks": [],
    }

    logger.debug(f"Report generation request: thread_id={thread_id}, request={request}")

    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        config["callbacks"].append(CallbackHandler())

    config["metadata"] = {
        "langfuse_session_id": str(uuid4()),
        "thread_id": thread_id,
        "project_id": request.get("project_id"),
        "flag_id": request.get("flag_ids"),
        "user_query": request["query"],
        "generation_type": request.get("generation_type", "ai_assistant"),
    }

    generation_type = request.get("generation_type", "ai_assistant")

    state = ReportState(
        messages=[HumanMessage(content=request["query"])],
        user_query=request["query"],
        flag_id=request.get("flag_ids"),
        project_id=request.get("project_id"),
        file_contents=request.get("file_contents"),
        user_selected_context=request.get("user_selected_context", False),
        generation_type=generation_type,
        report_context=request.get("report_context"),
        selected_text=request.get("selected_text"),
        selected_report_context=request.get("selected_report_context"),
    )

    try:
        async with get_report_graph() as graph:
            async for output in graph.astream(state, config):
                for node_name, node_state in output.items():
                    try:
                        if node_name in [
                            "assistant_agent",
                            "edit_agent",
                            "insight_agent",
                        ]:
                            logger.debug(f"Agent node executed: {node_name}")
                            async for response in prepare_report_output(node_state):
                                await websocket.send_json(response.model_dump())
                                logger.debug(
                                    f"Sent report response: {response.model_dump()}"
                                )
                        elif node_name == "tool":
                            # Tool responses are handled by prepare_report_output
                            # when the agent processes the tool results
                            logger.debug(
                                "Tool node executed. Results processed by agent."
                            )
                            continue
                        else:
                            logger.debug(f"Unknown node: {node_name}")
                            continue
                    except Exception as e:
                        logger.exception(f"Error processing node {node_name}: {str(e)}")

    except OpenAIBadRequestError as e:
        error_body = e.body.get("error", {})
        code = error_body.get("code")

        if code == "context_length_exceeded":
            header = "Context length exceeded"
            body_message = "Please try again with a shorter query or less context"
        else:
            header = "Bad request"
            body_message = error_body.get(
                "message", "Something went wrong. Please try again."
            )

        await websocket.send_json(
            ReportGenerationResponse(
                node="error",
                node_type="error",
                generation_type=generation_type,
                header=header,
                body_message=body_message,
            ).model_dump()
        )

    except AnthropicBadRequestError as e:
        error_body = e.body.get("error", {})
        error_message = error_body.get("message", "")

        if error_message.startswith("prompt is too long"):
            header = "Context length exceeded"
            body_message = "Please try again with a shorter query or less context"
        else:
            header = "Bad request"
            body_message = error_body.get(
                "message", "Something went wrong. Please try again."
            )

        await websocket.send_json(
            ReportGenerationResponse(
                node="error",
                node_type="error",
                generation_type=generation_type,
                header=header,
                body_message=body_message,
            ).model_dump()
        )

    except ChatGoogleGenerativeAIError as e:
        error_message = str(e)

        if "token count" in error_message and "exceeds" in error_message:
            header = "Content too large"
            body_message = (
                "The content is too large for processing. "
                "Please try with shorter content or fewer files."
            )
        elif "quota" in error_message or "exceeded" in error_message:
            header = "Rate limit exceeded"
            body_message = "API rate limit exceeded. Please try again in a few minutes."
        else:
            header = "Processing error"
            body_message = (
                "An error occurred while processing your request. Please try again."
            )

        await websocket.send_json(
            ReportGenerationResponse(
                node="error",
                node_type="error",
                generation_type=generation_type,
                header=header,
                body_message=body_message,
            ).model_dump()
        )

    except Exception as e:
        logger.exception(f"Error in report generation: {str(e)}")

        await websocket.send_json(
            ReportGenerationResponse(
                node="error",
                node_type="error",
                generation_type=generation_type,
                header="Something went wrong",
                body_message="Please try again after some time",
                process_description=f"Error processing request: {str(e)}",
            ).model_dump()
        )
