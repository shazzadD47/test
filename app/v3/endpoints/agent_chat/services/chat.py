import json
import random
from asyncio import sleep
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from uuid import uuid4

from fastapi import WebSocket
from langchain_core.messages import AIMessage, HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError

from app.configs import settings
from app.exceptions.system import (
    AnthropicBadRequestError,
    AnthropicInternalServerError,
    OpenAIBadRequestError,
    OpenAIServerError,
)
from app.v3.endpoints.agent_chat.constants import Agents
from app.v3.endpoints.agent_chat.jupyter.context_manager import (
    jupyter_context_manager,
)
from app.v3.endpoints.agent_chat.logging import logger
from app.v3.endpoints.agent_chat.notebook_chat.agent import NotebookAgent
from app.v3.endpoints.agent_chat.notebook_chat.tool_name_mapping import (
    TOOL_NAMES_MAPPING,
)
from app.v3.endpoints.agent_chat.schema import AgentChatResponse, AgentState
from app.v3.endpoints.agent_chat.services.graph import get_graph
from app.v3.endpoints.agent_chat.services.tools import (
    code_context_retrieval,
    paper_context_retrieval,
)
from app.v3.endpoints.agent_chat.utils import convert_to_title_case
from app.v3.endpoints.agent_chat.utils.files import (
    fetch_file_contents_parallel,
    fetch_stash_file_contents,
)
from app.v3.endpoints.agent_chat.utils.retriever import retrieve_context


def get_content_text(content: list[dict]) -> str:
    """
    Extract and concatenate the 'text' values from a
    list of dictionaries.
    Args:
        content (list[dict]):
        A list of dictionaries, where each dictionary
        may contain a 'text' key.
    Returns:
        str: A concatenated string of all 'text' values from the
             dictionaries in the list.
             If the input is not a list, or
             if a dictionary does not contain a 'text' key,
            those elements are ignored.
             Returns an empty string if the input is invalid
             or no 'text' keys are found.
    """

    if not isinstance(content, list):
        return ""

    text_pieces = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            text_pieces.append(item["text"])
    return "".join(text_pieces)


def is_notebook_file(file_path: str | None) -> bool:
    """
    Check if a file path has exactly the '.ipynb' extension.

    Args:
        file_path (Optional[str]): The path to the file.

    Returns:
        bool: True if the file ends in '.ipynb' (any case), False otherwise.
    """
    if not file_path:
        return False

    return Path(file_path).suffix.lower() == ".ipynb"


async def stream_notebook_agent_responses(
    state: AgentState,
) -> AsyncGenerator[AgentChatResponse, None]:
    """
    Stream responses from the external notebook agent in real-time.

    Converts SSE chunks from the notebook agent into AgentChatResponse
    objects as they arrive.
    Each tool_use becomes an AgentChatResponse with node=tool_name.
    Each tool_result becomes an AgentChatResponse with the result content.
    Each ai_message content becomes an AgentChatResponse with the agent's message.
    """

    knowledge_context = None

    if state.get("flag_id") is not None:
        context = await retrieve_context(
            query=state["user_query"],
            flag_id=state["flag_id"],
            project_id=state["project_id"],
        )

        knowledge_context = [content.page_content for content in context]

    # Initialize the NotebookAgent
    notebook_agent = NotebookAgent(
        project_id=state["project_id"],
        notebook_url=state["notebook_url"],
        notebook_token=state["notebook_token"],
        current_working_dir=state.get("current_notebook_path", ""),
        session_id=state.get("session_id"),
        current_notebook_file=state.get("current_notebook_path"),
        current_kernel=state.get("current_kernel", "python3"),
        knowledge_context=knowledge_context,
        file_contents=state.get("file_contents"),
    )

    # Track tool calls to map results to their tools
    # Maps tool_use_id -> (original_tool_name, display_node_name)
    tool_calls_map = {}

    # Stream the SSE chunks
    async for chunk in notebook_agent.generate_chat(state["user_query"]):
        # Parse SSE chunk
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                data_type = data.get("type")

                # Handle ai_message content
                if data_type == "ai_message" and data.get("content"):
                    content = data["content"]

                    # Stream the content as it arrives
                    yield AgentChatResponse(
                        node="code_generator",
                        node_type="agent",
                        header="Code Generator",
                        body_message=content,
                    )

                # Handle tool_use
                elif data_type == "tool_use":
                    tool_use_id = data.get("tool_use_id")
                    tool_name = data.get("tool_name", "unknown_tool")
                    tool_input = data.get("input", {})

                    # Check if tool name is in the mapping
                    if tool_name in TOOL_NAMES_MAPPING:
                        mapping = TOOL_NAMES_MAPPING[tool_name]
                        node_name = mapping["generic_name"]
                        body_message = mapping["description"]
                        header = node_name
                    else:
                        # Create a descriptive message for the tool call
                        node_name = tool_name
                        body_message = f"Executing {tool_name}"
                        if tool_input:
                            # Add a brief description of the args if available
                            arg_summary = ", ".join(
                                [
                                    f"{k}: {str(v)[:50]}"
                                    for k, v in list(tool_input.items())[:2]
                                ]
                            )
                            if arg_summary:
                                body_message += f" with {arg_summary}"
                        header = convert_to_title_case(tool_name)

                    # Store for later matching with results
                    # (store both original and display name)
                    if tool_use_id:
                        tool_calls_map[tool_use_id] = (tool_name, node_name)

                    yield AgentChatResponse(
                        node=node_name,
                        node_type="tool",
                        header=header,
                        body_message=body_message,
                    )

                # Handle tool_result
                elif data_type == "tool_result":
                    tool_use_id = data.get("tool_use_id")
                    result_content = data.get("content", [])

                    # Find the corresponding tool name (use display name if available)
                    tool_info = tool_calls_map.get(tool_use_id)
                    if tool_info:
                        _, display_node_name = tool_info
                        node_name = display_node_name
                    else:
                        node_name = "tool_result"
                        display_node_name = "tool_result"

                    # Format the result content
                    if isinstance(result_content, list):
                        result_text = "\n".join(
                            [str(c) for c in result_content[:3]]
                        )  # Show first 3 items
                    else:
                        result_text = str(result_content)[:200]  # Limit length

                    yield AgentChatResponse(
                        node=node_name,  # Use same node_name (mapped or original)
                        node_type="tool",
                        header=f"{display_node_name} Result",
                        body_message=(
                            result_text if result_text else "Tool executed successfully"
                        ),
                    )

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE chunk: {chunk[:100]}...")
                continue

        elif chunk.startswith("Error:"):
            # Handle error messages
            yield AgentChatResponse(
                node="error",
                node_type="error",
                header="Notebook Agent Error",
                body_message=chunk,
            )


async def prepare_agent_output(
    state: AgentState,
) -> AsyncGenerator[AgentChatResponse, None]:
    # Check if we should stream from the notebook agent
    if state.get("use_notebook_agent_streaming"):
        # Stream responses from the notebook agent in real-time
        async for response in stream_notebook_agent_responses(state):
            yield response
        return

    messages = state["messages"]
    header = "Delineate AI"
    ai_message: AIMessage | None = None

    for message in messages[-1::-1]:
        if isinstance(message, AIMessage):
            ai_message = message
            break

    if (
        state.get("sender") == Agents.MAIN_AGENT.value
        and (not state.get("tool_calls"))
        and state.get("sources")
    ):
        yield AgentChatResponse(
            node="agent",
            node_type="agent",
            header=header,
            body_message="Agent is looking up information with your query.",
        )
        await sleep(random.uniform(0.5, 1.5))  # nosec B311 - UI delay randomization

        yield AgentChatResponse(
            node="paper_context_retrieval",
            node_type="tool",
            header="Paper Context Retrieval",
            sources=state.get("sources"),
        )
        await sleep(random.uniform(0.75, 1.99))  # nosec B311 - UI delay randomization

    if state.get("tool_calls"):

        for tool_call in state.get("tool_calls"):

            tool_name = tool_call.get("name")

            if not tool_name:
                continue

            if tool_name in [
                paper_context_retrieval.name,
                code_context_retrieval.name,
            ]:
                query = tool_call.get("args").get("query")
                yield AgentChatResponse(
                    node=tool_name,
                    node_type="tool",
                    header=convert_to_title_case(tool_name),
                    body_message=f"Agent is looking up information with query: {query}",
                    sources=state.get("sources"),
                )

    if ai_message:
        if "FINISHED" in ai_message.content:
            header = "Final Answer"
            ai_message.content = ai_message.content.replace("FINISHED", "").strip()

        yield AgentChatResponse(
            node="agent",
            node_type="agent",
            header=header,
            body_message=ai_message.content,
        )


def prepare_tool_result_output(
    state: AgentState,
) -> Generator[AgentChatResponse, None, None]:
    messages = state["tool_results"]
    tool_request_message = state.get("tool_request_message")
    text = get_content_text(
        content=tool_request_message.content if tool_request_message else ""
    )

    for tool_message in messages:
        tool_name = tool_message.name

        yield AgentChatResponse(
            node=tool_name,
            node_type="tool",
            header=convert_to_title_case(tool_name),
            body_message=text,
            sources=state.get("sources"),
        )


async def chat(websocket: WebSocket, thread_id: str, request: dict):
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "agent_chat",
        "callbacks": [],
        "recursion_limit": 30,
    }

    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        config["callbacks"].append(CallbackHandler())

    config["metadata"] = {
        "langfuse_session_id": str(uuid4()),
        "thread_id": thread_id,
        "project_id": request.get("project_id"),
        "flag_id": request.get("flag_ids"),
        "user_query": request["query"],
        "notebook_chat": request.get("current_notebook_path") is not None,
    }

    agent_chat_type = request.get("agent_chat_type") or "chat"

    # Initialize Jupyter clients if we have a notebook path
    using_jupyter = is_notebook_file(request.get("current_notebook_path"))
    kernel_language = None
    kernel_language_version = None
    if using_jupyter:
        # Get or create kernel and notebook clients using the context manager
        # But don't store them in the state since they can't be serialized
        await jupyter_context_manager.get_or_create_clients(
            project_id=request.get("project_id"),
            notebook_url=request.get("notebook_url"),
            notebook_token=request.get("notebook_token"),
            notebook_path=request.get("current_notebook_path"),
        )
        kernel_language = jupyter_context_manager.get_kernel_language(
            project_id=request.get("project_id")
        )
        kernel_language_version = jupyter_context_manager.get_kernel_version(
            project_id=request.get("project_id")
        )

    file_contents = []
    file_paths = request.get("file_paths", [])
    file_type = request.get("file_type")

    try:
        if file_type and file_type == "stashFile":
            if request.get("project_id"):
                request["project_id"] = None
            file_contents = await fetch_stash_file_contents(file_paths)
        else:
            file_contents = await fetch_file_contents_parallel(
                file_paths=file_paths,
                project_id=request.get("project_id"),
                access_token=request.get("token"),
            )
    except Exception as e:
        logger.error("Failed to fetch file contents: %s", str(e))
        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header="Failed to Load File Contents",
                body_message=(
                    "Unable to load the requested file contents. "
                    "Please try again or check if the files are accessible."
                ),
                process_description=f"Error fetching file contents: {str(e)}",
            ).model_dump()
        )
        return

    state = AgentState(
        messages=[HumanMessage(content=request["query"])],
        user_query=request["query"],
        flag_id=request.get("flag_ids"),
        project_id=request.get("project_id"),
        current_notebook_path=request.get("current_notebook_path"),
        file_contents=file_contents,
        using_jupyter=using_jupyter,
        language=kernel_language,
        language_version=kernel_language_version,
        user_selected_context=request.get("user_selected_context", False),
        agent_chat_type=agent_chat_type,
        notebook_url=request.get("notebook_url"),
        notebook_token=request.get("notebook_token"),
        session_id=thread_id,
        current_kernel=kernel_language,
    )

    try:
        async with get_graph() as graph:
            async for output in graph.astream(state, config):
                for agent, state in output.items():
                    try:
                        if agent in [
                            Agents.MAIN_AGENT,
                            Agents.CODE_GENERATOR,
                            Agents.DEEP_AGENT,
                        ]:
                            async for response in prepare_agent_output(state):
                                await websocket.send_json(response.model_dump())
                                logger.debug(
                                    f"Sent Agent response: {response.model_dump()}"
                                )
                        elif agent == "tool":
                            for response in prepare_tool_result_output(state):
                                await websocket.send_json(response.model_dump())
                                logger.debug(
                                    f"Sent Tool response: {response.model_dump()}"
                                )
                        elif agent == "setup":
                            logger.debug("Setup node. No response sent.")
                            continue
                        else:
                            response = AgentChatResponse(
                                node="error",
                                node_type="error",
                                header="Something went wrong",
                                body_message="Please try again after some time",
                                process_description=f"Unknown node: {agent}",
                            )
                            await websocket.send_json(response.model_dump())
                            logger.debug(
                                f"Sent Error response: {response.model_dump()}"
                            )
                    except Exception as e:
                        logger.exception(f"Error sending message: {str(e)}")
                        continue
    except OpenAIBadRequestError as e:
        error_body = e.body.get("error", {})
        code = error_body.get("code")

        if code == "context_length_exceeded":
            header = "Context length exceeded"
            body_message = "Please try again with a shorter query or start a new chat"
        else:
            header = "Bad request"
            body_message = error_body.get(
                "message",
                "Opps! Something went wrong. Please try again or start a new chat.",
            )

        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header=header,
                body_message=body_message,
            ).model_dump()
        )
    except AnthropicBadRequestError as e:
        error_body = e.body.get("error", {})
        error_message = error_body.get("message")

        if error_message.startswith("prompt is too long"):
            header = "Context length exceeded"
            body_message = "Please try again with a shorter query or start a new chat"
        else:
            header = "Bad request"
            body_message = error_body.get(
                "message",
                "Opps! Something went wrong. Please try again or start a new chat.",
            )

        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header=header,
                body_message=body_message,
            ).model_dump()
        )
    except AnthropicInternalServerError as e:
        status_code = e.status_code

        if status_code == 429:
            header = "Rate limit exceeded"
            body_message = (
                "Anthropic server is exhuasted. Please try again after some time."
            )
        else:
            header = "Internal server error"
            body_message = (
                "Anthropic server is experiencing issues."
                " Please try again after some time."
            )

        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header=header,
                body_message=body_message,
            ).model_dump()
        )
    except OpenAIServerError as e:
        status_code = e.status_code

        if status_code == 429:
            header = "Rate limit exceeded"
            body_message = (
                "OpenAI server is exhuasted. Please try again after some time."
            )
        else:
            header = "Internal server error"
            body_message = (
                "OpenAI server is experiencing issues."
                " Please try again after some time."
            )

        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header=header,
                body_message=body_message,
            ).model_dump()
        )
    except GraphRecursionError as e:
        logger.exception(f"Rate Limit: {str(e)}")
        await websocket.send_json(
            AgentChatResponse(
                node="limit",
                node_type="error",
                header="Continue to iterate?",
                body_message="<>[UI:Continue-button]</>",
                process_description="",
            ).model_dump()
        )
    except Exception as e:
        logger.exception(f"Error in chat stream: {str(e)}")

        await websocket.send_json(
            AgentChatResponse(
                node="error",
                node_type="error",
                header="Something went wrong",
                body_message="Please try again after some time",
                process_description=f"Error processing chat: {str(e)}",
            ).model_dump()
        )
    finally:
        if using_jupyter:
            await jupyter_context_manager.close_clients(
                project_id=request.get("project_id"),
            )
