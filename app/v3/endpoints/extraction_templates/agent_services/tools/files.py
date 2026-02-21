from mimetypes import guess_type
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.integrations.backend.projects import get_project_details
from app.logging import logger
from app.utils.download import async_download_files_from_flag_id
from app.utils.files import create_file_input
from app.v3.endpoints.extraction_templates.agent_services.schemas import (
    AgentState,
)
from app.v3.endpoints.extraction_templates.configs import settings as template_settings

DESCRIBE_PROJECT_TOOL_DESCRIPTION = (
    """Describes the current project user is working on."""
)

LIST_PROJECT_FILES_TOOL_DESCRIPTION = """Lists project files with pagination using limit and offset parameters.

supported files: PDFs, images, and text files.

Parameters:
- limit: Number of files to return (e.g., 10 to get 10 files)
- offset: Number of files to skip from the beginning (e.g., 0 for first page, 10 for second page, 20 for third page)

Example usage:
- To get files 1-10: limit=10, offset=0
- To get files 11-20: limit=10, offset=10
- To get files 21-30: limit=10, offset=20
"""  # noqa: E501


READ_FILE_TOOL_DESCRIPTION = """Read a file from the project with file_id.

Parameters:
- file_id: ID of the file to read
- offset: Starting line number for text files (default: 0)
- limit: Number of lines to read for text files (default: 200)
- read_all: Read entire file content for text files, ignoring offset/limit (default: False)
  WARNING: Use read_all=True only for small files (<1MB) or when complete content is needed
"""  # noqa: E501


def _is_binary_file(file_path: Path) -> bool:
    """Check if a file is a binary file like images, pdf, etc."""
    mime_type, _ = guess_type(file_path)

    if not mime_type:
        return False

    return mime_type.startswith("image") or mime_type.startswith("application/pdf")


def _format_line(line: str, line_number: int, n_lines_digits: int) -> str:
    return f"{line_number:0{n_lines_digits}d}: {line}"


@tool(description=DESCRIBE_PROJECT_TOOL_DESCRIPTION)
async def describe_project(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str | Command:
    project_id = state.get("project_id")

    project_details = await get_project_details(project_id, state.get("user_token"))

    if project_details.get("error"):
        if project_details.get("status_code") == "403":
            return "The user does not have permission to access this project."
        else:
            logger.warning(
                "An unknown error occurred while fetching the "
                f"project details: {project_details}"
            )
            return "An unknown error occurred while fetching the project details."

    project_files = []
    flag_to_file_id_map = {}
    for idx, file in enumerate(project_details.get("files")):
        flag_id = file.get("flagId")
        file_id = flag_to_file_id_map.get(flag_id, str(idx + 1))
        flag_to_file_id_map[flag_id] = file_id

        file_details = {
            "file_name": file.get("fileName"),
            "title": file.get("title"),
            "summary": file.get("summary"),
            "file_id": file_id,
        }
        project_files.append(file_details)

    files_to_display = project_files[:10] if len(project_files) > 10 else project_files

    content_parts = [
        f"Project Name: {project_details.get('name')}\n"
        f"Project Description: {project_details.get('description')}\n\n"
        f"Project Files: {files_to_display}"
    ]

    if len(project_files) > 10:
        content_parts.append(
            f"\n\nNote: Showing details for 10 out of {len(project_files)} files. "
            "Use other available tools to get details about specific files."
        )

    message = ToolMessage(
        content="".join(content_parts),
        tool_call_id=tool_call_id,
        name="describe_project",
    )

    sorted_project_files = sorted(
        project_files,
        key=lambda x: x.get("file_id"),
    )
    file_id_to_flag_map = {
        v if isinstance(v, str) else str(v): k for k, v in flag_to_file_id_map.items()
    }

    return Command(
        update={
            "messages": [message],
            "flag_to_file_id_map": flag_to_file_id_map,
            "file_id_to_flag_map": file_id_to_flag_map,
            "file_details_list": sorted_project_files,
        }
    )


@tool(description=LIST_PROJECT_FILES_TOOL_DESCRIPTION)
def list_project_files(
    state: Annotated[AgentState, InjectedState],
    offset: int = 0,
    limit: int = 15,
) -> str:
    """List project files with pagination using limit and offset.

    Args:
        limit: Number of files to return
        offset: Number of files to skip from the beginning

    Returns:
        String with paginated file list
    """
    file_details_list = state.get("file_details_list", [])

    if not file_details_list:
        return (
            "No files found. You must use the describe_project tool "
            "first to load project files."
        )

    if offset < 0:
        offset = 0
    if limit < 1:
        limit = 10

    total_files = len(file_details_list)

    start_idx = offset
    end_idx = min(offset + limit, total_files)

    if start_idx >= total_files:
        return (
            f"Offset {offset} is beyond the total number of files ({total_files}). "
            f"Please use an offset less than {total_files}."
        )

    paginated_files = file_details_list[start_idx:end_idx]

    content = (
        f"Files {start_idx + 1} to {end_idx} out of {total_files} files:\n\n"
        f"{paginated_files}\n\n"
        f"To get more files, use offset={end_idx} with your desired limit."
    )

    return content


@tool(description=READ_FILE_TOOL_DESCRIPTION)
async def read_file(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    file_id: int,
    offset: int = 0,
    limit: int = 200,
    read_all: bool = False,
) -> str | ToolMessage:
    """Read a file from the project with file_id."""
    file_id_to_flag_map = state.get("file_id_to_flag_map", {})
    file_id = str(file_id)

    if file_id not in file_id_to_flag_map:
        return (
            f"File with id {file_id} not found. "
            "Please first use the describe_project tool to get the project files. "
            "Then if needed, use the list_project_files tool to get the file_id."
        )

    flag_id = file_id_to_flag_map[file_id]
    file_path = await async_download_files_from_flag_id(flag_id, return_type="path")
    if file_path is None:
        return f"Failed to download file with id {file_id}. Please try again later."

    if _is_binary_file(file_path):
        return ToolMessage(
            content=[create_file_input(file_path, template_settings.MAIN_AGENT)],
            tool_call_id=tool_call_id,
            name="read_file",
            artifact={"file_path": str(file_path)},
        )

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if read_all and file_size_mb > 1:
        return (
            f"Warning: File size is {file_size_mb:.2f} MB\n\n"
            f"Reading large files with read_all=True may be slow and token-intensive.\n"
            f"Recommendation: Use offset/limit parameters for pagination.\n\n"
            f"To proceed anyway, this is a safety check - file NOT read."
        )

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        return (
            f"Error: Failed to read file: {file_path}\n\n"
            f"The file is not valid UTF-8 encoded text.\n"
            f"System error: {e}\n\n"
            f"Troubleshooting:\n"
            f"1. File may be binary or use a different encoding\n"
            f"2. Check if the file type is supported by the tool"
        )
    except Exception as e:
        return (
            f"Error: Failed to read file: {file_path}\n\n"
            f"System error: {e}\n\n"
            f"Troubleshooting:\n"
            f"1. Check file permissions\n"
            f"2. Verify the file is not locked by another process\n"
            f"3. Ensure the file is not corrupted"
        )

    n_lines = len(lines)
    n_lines_digits = len(str(n_lines))

    if offset >= n_lines:
        return (
            f"Error: Offset {offset} is beyond file length\n\n"
            f"File has {n_lines} lines (line numbers 1-{n_lines}).\n"
            f"Your offset starts at line {offset + 1}.\n\n"
            f"Troubleshooting:\n"
            f"1. Use offset=0 to read from the beginning\n"
            f"2. Use offset={max(0, n_lines - limit)} to read the last {limit} lines\n"
            f"3. Use read_all=True to read the entire file"
        )

    if read_all:
        result_lines = [
            _format_line(line, i + 1, n_lines_digits) for i, line in enumerate(lines)
        ]
        return "".join(result_lines)

    result_lines = []
    for i in range(offset, min(offset + limit, n_lines)):
        result_lines.append(_format_line(lines[i], i + 1, n_lines_digits))

    if offset + limit < n_lines:
        remaining = n_lines - (offset + limit)
        result_lines.append(
            f"\n... ({remaining} more lines available. "
            f"Use offset={offset + limit} to continue.)"
        )

    return "".join(result_lines)
