import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.configs import settings

logger = logging.getLogger(__name__)


class NotebookAgentResponse:
    """Structured response from NotebookAgent."""

    def __init__(self):
        self.content: str = ""
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []
        self.raw_data: list[dict] = []

    def add_content(self, text: str):
        """Add content text."""
        self.content += text

    def add_tool_call(self, tool_call: dict):
        """Add a tool call."""
        self.tool_calls.append(tool_call)
        logger.debug(f"Tool call: {tool_call['name']}")

    def add_tool_result(self, tool_result: dict):
        """Add a tool result."""
        self.tool_results.append(tool_result)
        logger.debug(f"Tool result: {tool_result['tool_call_id']}")

    def add_raw_data(self, data: dict):
        """Add raw SSE data."""
        self.raw_data.append(data)

    def get_summary(self) -> str:
        """Get a summary of the response with tool usage."""
        summary = self.content or "Code generation completed."

        if self.tool_calls:
            summary += "\n\n**Tools Used:**\n"
            tool_names = set()
            for tc in self.tool_calls:
                if tc["name"] not in tool_names:
                    summary += f"- `{tc['name']}`\n"
                    tool_names.add(tc["name"])

        return summary


class NotebookAgent:
    """Client for external notebook agent service with SSE streaming support."""

    base_url = settings.NOTEBOOK_AGENT_BASE_URL

    def __init__(
        self,
        project_id: str,
        notebook_url: str,
        notebook_token: str,
        current_working_dir: str | None = None,
        session_id: str | None = None,
        current_notebook_file: str | None = None,
        current_kernel: str | None = None,
        knowledge_context: str | None = None,
        file_contents: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.notebook_url = notebook_url
        self.notebook_token = notebook_token
        self.current_working_dir = current_working_dir
        self.session_id = session_id
        self.current_notebook_file = current_notebook_file
        self.current_kernel = current_kernel
        self.knowledge_context = knowledge_context
        self.file_contents = file_contents

    def _parse_sse_chunk(self, chunk: str, response: NotebookAgentResponse):
        """Parse a single SSE chunk and update the response object."""
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                response.add_raw_data(data)

                data_type = data.get("type")

                if data_type == "ai_message" and data.get("content"):
                    response.add_content(data["content"])

                elif data_type == "tool_use":
                    response.add_tool_call(
                        {
                            "id": data.get("tool_use_id"),
                            "name": data.get("tool_name"),
                            "args": data.get("input", {}),
                            "type": "tool_call",
                        }
                    )

                elif data_type == "tool_result":
                    response.add_tool_result(
                        {
                            "tool_call_id": data.get("tool_use_id"),
                            "content": data.get("content", []),
                            "type": "tool_result",
                        }
                    )

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE chunk: {chunk[:100]}...")

        elif not chunk.startswith(": ping") and chunk.strip():
            response.add_content(chunk)

    async def generate_chat(self, query: str) -> AsyncGenerator[str, None]:
        """
        Stream raw SSE chunks from the external notebook agent service.

        Args:
            query: The user query to send to the notebook agent

        Yields:
            Streaming SSE chunks from the external service
        """
        try:

            payload = {
                "query": query,
                "current_working_dir": self.current_working_dir or "",
                "session_id": self.session_id or self.project_id,
                "project_id": self.project_id,
                "current_notebook_file": self.current_notebook_file or "Untitled.ipynb",
                "jupyter_url": self.notebook_url,
                "jupyter_token": self.notebook_token,
                "current_kernel": self.current_kernel or "python3",
                "knowledge_context": (
                    json.dumps(self.knowledge_context) if self.knowledge_context else ""
                ),
                "file_contents": (
                    json.dumps(self.file_contents) if self.file_contents else ""
                ),
            }

            logger.debug(f"[NB] Sending Message {payload}")

            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/agent/chat",
                    json=payload,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk

        except httpx.TimeoutException as e:
            error_msg = f"Timeout error connecting to notebook agent: {str(e)}"
            yield f"Error: {error_msg}"

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from notebook agent: {e.response.status_code}"
            yield f"Error: {error_msg}"

        except httpx.RequestError as e:
            error_msg = f"Request error connecting to notebook agent: {str(e)}"
            yield f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error in notebook agent: {str(e)}"
            yield f"Error: {error_msg}"
