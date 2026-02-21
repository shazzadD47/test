from logging import getLogger
from typing import Annotated, Any

from langchain_core.tools import tool

from app.utils.repl import PythonREPL
from app.v3.endpoints.agent_chat.jupyter.context_manager import (
    jupyter_context_manager,
)
from app.v3.endpoints.agent_chat.schema import ContextDocs
from app.v3.endpoints.agent_chat.utils.image_analyze import get_image_context
from app.v3.endpoints.agent_chat.utils.retriever import retrieve_context

# --- Helper function to manage client connections (Conceptual) ---
# In a more advanced setup, kernel and notebook clients might be managed per session
# rather than re-established (without stopping) in each tool.
# For now, we ensure they are started but NOT stopped within the tool.

# -- Helper functions REMOVED as client initialization should happen outside tools --
# async def _get_notebook_client(...): ...
# async def _get_kernel_client(...): ...

# --- Tools Definition ---


logger = getLogger(__name__)


@tool
def execute_python_code(code: str) -> str:
    """Execute Python code and return the output.
    Args:
        code: str Python code to execute.
    Returns: str python code execution result.
    """
    python_repl = PythonREPL()
    try:
        result = python_repl.run(code)
        return f"Execution result:\n{result}"
    except Exception as e:
        return f"Error executing code: {str(e)}"


@tool
def paper_context_retrieval(
    query: Annotated[str, "The query to search context from a paper"],
    flag_id: Annotated[
        str | None, "Analogous to a file id. It is a unique id for a research paper."
    ] = None,
    project_id: Annotated[
        str | None,
        "The id of the project in which the research paper belongs to.",
    ] = None,
) -> list[ContextDocs]:
    """Get information from research papers based on the flag_id and/or project_id.
    The information is stored in a vector store with contextual embeddings.

    flag_id is the id of the research paper like a file id.
    project_id is the id of the project in which the research paper belongs to.

    Give a `specific query` to search for the most relevant information.
    """
    return retrieve_context(query, flag_id, project_id, "document")


@tool
def code_context_retrieval(
    query: Annotated[str, "The query to search context from a code"],
    flag_id: Annotated[
        str | None, "Analogous to a file id. It is a unique id for a code file."
    ] = None,
    project_id: Annotated[
        str | None, "The id of the project in which the code file belongs to."
    ] = None,
) -> list[ContextDocs]:
    """Get information from code files based on the flag_id and/or project_id.
    The information is stored in a vector store with contextual embeddings.

    flag_id is the id of the code file like a file id.
    project_id is the id of the project in which the code file belongs to.

    Give a `specific query` to search for the most relevant information.
    """
    return retrieve_context(query, flag_id, project_id, "code")


# --- Notebook Tools (Updated to use JupyterContextManager) ---


@tool
async def create_markdown_cell(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
    # Arguments related to the action itself
    markdown_content: Annotated[str, "The markdown content for the new cell"],
    after_cell_index: Annotated[
        int | None, "Index of the cell after which to insert. Appends if None."
    ] = None,
) -> dict[str, Any]:
    """Creates a new markdown cell in the project's notebook.
    Returns a dictionary with the cell_id_or_index and status.
    """
    try:
        # Get the notebook client from the context manager
        if project_id not in jupyter_context_manager._notebook_clients:
            return {"error": f"No notebook client found for project_id: {project_id}"}

        notebook_client = jupyter_context_manager._notebook_clients[project_id]

        # Create the cell
        cell_index = notebook_client.add_markdown_cell(markdown_content)
        # Consider if notebook_client.save() is needed here.
        return {"cell_id_or_index": cell_index, "status": "created"}
    except Exception as e:
        return {"error": f"Error creating markdown cell: {str(e)}"}


@tool
async def modify_cell_content(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
    # Arguments related to the action itself
    cell_id_or_index: Annotated[int, "The index of the cell to modify."],
    new_content: Annotated[str, "The new content for the cell"],
    cell_type: Annotated[str, "Type of the cell ('code' or 'markdown')"] = "code",
) -> dict[str, Any]:
    """Modifies the content of an existing cell in the project's notebook.
    Returns a dictionary with the status.
    """
    try:
        # Get the notebook client from the context manager
        if project_id not in jupyter_context_manager._notebook_clients:
            return {"error": f"No notebook client found for project_id: {project_id}"}

        notebook_client = jupyter_context_manager._notebook_clients[project_id]

        # Modify the cell based on type
        if hasattr(notebook_client, "update_cell"):
            # If a general update_cell method exists
            notebook_client.update_cell(cell_id_or_index, new_content, cell_type)
        else:
            # Fallback: Try to use type-specific methods
            if cell_type.lower() == "markdown":
                notebook_client.update_markdown_cell(cell_id_or_index, new_content)
            else:
                notebook_client.update_code_cell(cell_id_or_index, new_content)

        return {"cell_id_or_index": cell_id_or_index, "status": "modified"}
    except Exception as e:
        return {"error": f"Error modifying cell: {str(e)}"}


@tool
async def get_cell_info(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
    # Arguments related to the action itself
    cell_id_or_index: Annotated[int, "The index of the cell to get information about."],
) -> dict[str, Any]:
    """
    Gets information about a specific cell in the project's notebook.
    Returns a dictionary with the cell contents, type,
    and other available metadata.
    """
    try:
        if project_id not in jupyter_context_manager._notebook_clients:
            return {"error": f"No notebook client found for project_id: {project_id}"}

        notebook_client = jupyter_context_manager._notebook_clients[project_id]

        if hasattr(notebook_client, "get_cell_by_index"):
            cell_info = notebook_client.get_cell_by_index(cell_id_or_index)
            return cell_info
        else:
            return {
                "error": "get_cell_by_index method not implemented in notebook client"
            }
    except Exception as e:
        return {"error": f"Error getting cell info: {str(e)}"}


@tool
async def get_notebook_structure(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
) -> dict[str, Any]:
    """
    Gets the structure of the project's notebook,
    including the number of cells and their types.
    Returns a dictionary with the notebook structure.
    """
    try:
        # Get the notebook client from the context manager
        if project_id not in jupyter_context_manager._notebook_clients:
            return {"error": f"No notebook client found for project_id: {project_id}"}

        notebook_client = jupyter_context_manager._notebook_clients[project_id]

        notebook_metadata = notebook_client.metadata

        notebook_language = notebook_metadata.get("kernelspec", {}).get(
            "language", "python"
        )
        notebook_language_version = notebook_metadata.get("language_info", {}).get(
            "version", "unknown"
        )

        return {
            "language": notebook_language,
            "version": notebook_language_version,
        }

    except Exception as e:
        return {"error": f"Error getting notebook structure: {str(e)}"}


@tool
async def delete_cell(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
    # Arguments related to the action itself
    cell_index: Annotated[int, "The index of the cell to delete."],
) -> dict[str, Any]:
    """Deletes a cell from the project's notebook by its index.
    Returns a dictionary with the status of the deletion.
    """
    try:
        # Get the notebook client from the context manager
        if project_id not in jupyter_context_manager._notebook_clients:
            return {"error": f"No notebook client found for project_id: {project_id}"}

        notebook_client = jupyter_context_manager._notebook_clients[project_id]

        # Check if the cell index is valid
        if cell_index < 0 or cell_index >= len(notebook_client):
            return {
                "error": f"Invalid cell index: {cell_index}."
                f"Notebook has {len(notebook_client)} cells."
            }

        # Store notebook length before deletion
        length_before = len(notebook_client)

        # Store cell info before deletion for confirmation
        try:
            cell_info = notebook_client[cell_index]
            cell_type = cell_info.get("cell_type", "unknown")
        except Exception:
            cell_type = "unknown"

        # Delete the cell with error handling for the 'to_py' issue
        try:
            del notebook_client[cell_index]

            # Sync the notebook to ensure changes are persisted
            try:
                if hasattr(notebook_client, "wait_until_synced"):
                    await notebook_client.wait_until_synced()
                elif hasattr(notebook_client, "sync"):
                    await notebook_client.sync()
            except Exception as sync_error:
                logger.warning(f"Failed to sync notebook after deletion: {sync_error}")
                # Don't fail the operation if sync fails

        except Exception as delete_error:
            # Handle the specific 'to_py' error gracefully
            if "'dict' object has no attribute 'to_py'" in str(delete_error):
                # Check if deletion actually worked by comparing lengths
                length_after = len(notebook_client)
                if length_after < length_before:
                    # Deletion succeeded despite the internal error
                    # Try to sync even in error case
                    try:
                        if hasattr(notebook_client, "wait_until_synced"):
                            await notebook_client.wait_until_synced()
                    except Exception as sync_error:
                        logger.warning(
                            "Failed to sync notebook after deletion (error case):"
                            f"{sync_error}"
                        )
                else:
                    return {"error": f"Cell deletion failed: {str(delete_error)}"}
            else:
                return {"error": f"Error deleting cell: {str(delete_error)}"}

        return {
            "cell_index": cell_index,
            "cell_type": cell_type,
            "status": "deleted",
            "message": f"Successfully deleted {cell_type} cell at index {cell_index}",
        }
    except Exception as e:
        return {"error": f"Error deleting cell: {str(e)}"}


@tool
async def add_and_execute_code(
    # Updated to use project_id
    project_id: Annotated[str, "The project ID to identify the proper clients"],
    # Arguments related to the action itself
    code: Annotated[str, "The code to add and execute"],
) -> dict[str, Any]:
    """
    Adds a new code cell to the project's notebook and executes it.
    """
    try:
        cell_index, result = await jupyter_context_manager.add_and_execute_code(
            project_id=project_id,
            code=code,
        )
        # Format the result for better output
        output_text = ""
        error_text = ""
        if result.get("status") == "ok":
            for output_item in result.get("outputs", []):
                if output_item.get("output_type") == "stream":
                    output_text += output_item.get("text", "")
                elif output_item.get("output_type") == "execute_result":
                    output_text += output_item.get("data", {}).get("text/plain", "")
                elif output_item.get("output_type") == "display_data":
                    if "image/png" in output_item.get("data", {}):
                        # Get image analysis
                        try:
                            analysis = await get_image_context(
                                query="What is the main plot of the image?",
                                base64_image=output_item["data"]["image/png"],
                            )
                            output_text += f"\nImage Analysis: {analysis}\n"
                        except Exception as e:
                            logger.error(f"Error analyzing image: {str(e)}")
                            output_text += "\nImage Analysis: [Error analyzing image]\n"
                    else:
                        output_text += output_item.get("data", {}).get(
                            "text/plain", "[display_data]"
                        )
        elif result.get("status") == "error":
            error_name = result.get("ename", "Error")
            error_value = result.get("evalue", "")
            error_text = f"{error_name}: {error_value}"

        response = {
            "status": result.get("status", "unknown"),
            "output": output_text.strip(),
            "execution_count": result.get("execution_count"),
            "problem_cell_index": (
                cell_index if result.get("status") == "error" else None
            ),
        }

        if error_text:
            response["error"] = error_text
        logger.info(f"add_and_execute_code response: {response}")
        return response
    except KeyError as ke:
        return {"error": str(ke)}
    except Exception as e:
        return {"error": f"Error adding and executing code: {str(e)}"}
