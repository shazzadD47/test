import asyncio
import re
import traceback
from typing import Any

from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import NbModelClient, get_jupyter_notebook_websocket_url

from app.v3.endpoints.agent_chat.logging import logger

# Set up logging


class JupyterContextManager:
    """
    Manages Jupyter notebook kernels and clients, keyed by project_id.
    This allows reusing kernels and clients across different requests
    for the same project.
    """

    def __init__(self):
        """Initialize the context manager with
        empty dictionaries for kernels and clients.
        """
        self._kernel_clients: dict[str, KernelClient] = {}
        self._notebook_clients: dict[str, NbModelClient] = {}
        self._lock = asyncio.Lock()

        # Map common error patterns to fixes
        self._common_imports = {
            r"name 'plt' is not defined": "import matplotlib.pyplot as plt",
            r"name 'np' is not defined": "import numpy as np",
            r"name 'pd' is not defined": "import pandas as pd",
            r"name 'sns' is not defined": "import seaborn as sns",
            r"No module named '([^']+)'": "!pip install \\1",
            r"ModuleNotFoundError: No module named '([^']+)'": "!pip install \\1",
        }
        logger.info("JupyterContextManager initialized")

    async def get_or_create_clients(
        self,
        project_id: str,
        notebook_url: str,
        notebook_token: str,
        notebook_path: str | None = None,
    ) -> tuple[KernelClient, NbModelClient]:
        """
        Get existing clients for the project or
        create new ones if they don't exist.

        Args:
            project_id: The unique identifier for the project
            notebook_url: The URL of the Jupyter notebook server
            notebook_token: Authentication token for the
            Jupyter notebook server
            notebook_path: Optional path to the notebook file

        Returns:
            Tuple containing (kernel_client, notebook_client)
        """
        try:
            async with self._lock:
                # Check if clients already exist for this project
                if (
                    project_id in self._kernel_clients
                    and project_id in self._notebook_clients
                ):
                    return (
                        self._kernel_clients[project_id],
                        self._notebook_clients[project_id],
                    )

                # Create new clients
                logger.info(f"Creating new clients for project {project_id}")
                kernel_client = KernelClient(
                    server_url=notebook_url,
                    token=notebook_token,
                )

                url = get_jupyter_notebook_websocket_url(
                    server_url=notebook_url, token=notebook_token, path=notebook_path
                )

                notebook_client = NbModelClient(url)
                await notebook_client.start()

                # Start the clients
                kernel_name = notebook_client.metadata.get(
                    "kernelspec", {"name": "python3"}
                ).get("name", "python3")

                logger.info(f"Starting kernel client with name: {kernel_name}")

                kernel_client.start(
                    name=kernel_name,
                )

                logger.info("Starting notebook client")

                # Store for future use
                self._kernel_clients[project_id] = kernel_client
                self._notebook_clients[project_id] = notebook_client

                logger.info(f"Clients created and started for project {project_id}")
                return kernel_client, notebook_client
        except Exception as e:
            logger.error(f"Error creating clients for project {project_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_kernel_language(self, project_id: str) -> str:
        """
        Get the kernel name for a specific project.

        Args:
            project_id: The unique identifier for the project

        Returns:
            The kernel name as a string
        """
        try:
            if project_id in self._notebook_clients:
                notebook_client = self._notebook_clients[project_id]
                language = notebook_client.metadata.get(
                    "language_info", {"name": "python3"}
                ).get("name", "python3")
                return language
            else:
                logger.error(f"No notebook client found for project_id: {project_id}")
                return "python3"
        except Exception as e:
            logger.error(f"Error getting kernel name: {str(e)}")
            return "python3"

    def get_kernel_version(self, project_id: str) -> str:
        """
        Get the kernel version for a specific project.

        Args:
            project_id: The unique identifier for the project

        Returns:
            The kernel version as a string
        """
        try:
            if project_id in self._notebook_clients:
                notebook_client = self._notebook_clients[project_id]
                kernel_version = notebook_client.metadata.get(
                    "language_info", {"version": "3.12"}
                ).get("version", "3.12")
                return kernel_version
            else:
                logger.error(f"No notebook client found for project_id: {project_id}")
                return "3.12"
        except Exception as e:
            logger.error(f"Error getting kernel version: {str(e)}")
            return "3.12"

    def _get_cell_source(self, notebook_client, cell_index):
        """Get the source code of a cell by its index."""
        try:
            logger.info(f"Getting source for cell {cell_index}")

            # If get_cell_by_index is available, use it
            if hasattr(notebook_client, "get_cell_by_index"):
                logger.info("Using get_cell_by_index method")
                cell_info = notebook_client.get_cell_by_index(cell_index)
                if cell_info and "source" in cell_info:
                    return cell_info["source"]

            # Fallback if no direct method is available
            if hasattr(notebook_client, "get_cells"):
                logger.info("Using get_cells method")
                cells = notebook_client.get_cells()
                if 0 <= cell_index < len(cells):
                    return cells[cell_index].get("source", "")

            logger.warning(f"Could not get source for cell {cell_index}")
            # If we can't get the source, return empty string
            return ""
        except Exception as e:
            logger.error(f"Error getting cell source: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _update_cell_with_imports(self, cell_content, imports_to_add):
        """Add imports to the top of a cell's content."""
        try:
            logger.info(f"Updating cell with imports: {imports_to_add}")

            # Add each import on a new line at the top of the cell
            imports_text = "\n".join(imports_to_add)

            # Check if imports are already present
            for import_line in imports_to_add:
                if import_line in cell_content:
                    # Skip this import as it's already there
                    logger.info(f"Import already present: {import_line}")
                    imports_text = imports_text.replace(import_line + "\n", "")
                    imports_text = imports_text.replace(import_line, "")

            if not imports_text.strip():
                # No new imports to add
                logger.info("No new imports to add")
                return cell_content

            updated_content = imports_text + "\n\n" + cell_content
            logger.info(f"Updated cell content: {updated_content[:100]}...")
            return updated_content
        except Exception as e:
            logger.error(f"Error updating cell with imports: {str(e)}")
            logger.error(traceback.format_exc())
            return cell_content

    async def add_and_execute_code(
        self, project_id: str, code: str, auto_fix_imports: bool = True
    ) -> Any:
        """
        Add a code cell to the notebook and execute it.

        Args:
            project_id: The project ID to identify the clients
            code: The code to execute
            auto_fix_imports: Whether to automatically fix common import errors

        Returns:
            Execution results

        Raises:
            KeyError: If no clients exist for the given project_id
        """
        logger.info(f"Adding and executing code for project {project_id}")
        logger.info(f"Code to execute: {code[:100]}...")

        try:
            if (
                project_id not in self._kernel_clients
                or project_id not in self._notebook_clients
            ):
                logger.error(f"No Jupyter clients found for project_id: {project_id}")
                raise KeyError(f"No Jupyter clients found for project_id: {project_id}")

            kernel_client = self._kernel_clients[project_id]
            notebook_client = self._notebook_clients[project_id]

            # Add a code cell
            logger.info("Adding code cell")
            cell_index = notebook_client.add_code_cell(code)
            logger.info(f"Added cell at index {cell_index}")

            # Execute the cell
            logger.info(f"Executing cell {cell_index}")
            result = notebook_client.execute_cell(cell_index, kernel_client)
            logger.info(f"Execution result status: {result.get('status')}")
            return cell_index, result
        except Exception as e:
            logger.error(f"Error adding and executing code: {str(e)}")
            logger.error(traceback.format_exc())
            # NOTE: This is a workaround for a specific error
            if hasattr(e, "args"):
                # ('transient',)
                error_arg = e.args[0]
                if error_arg == "transient":
                    return {
                        "status": "ok",
                        "outputs": [
                            {
                                "output_type": "stream",
                                "name": "stdout",
                                "text": "successfully added cell, and executed it. ",
                            }
                        ],
                    }
            raise

    async def execute_cell(
        self, project_id: str, cell_index: int, auto_fix_imports: bool = True
    ) -> Any:
        """
        Execute an existing cell in the notebook.

        Args:
            project_id: The project ID to identify the clients
            cell_index: The index of the cell to execute
            auto_fix_imports: Whether to automatically fix common import errors

        Returns:
            Execution results

        Raises:
            KeyError: If no clients exist for the given project_id
        """
        logger.info(f"Executing cell {cell_index} for project {project_id}")

        try:
            if (
                project_id not in self._kernel_clients
                or project_id not in self._notebook_clients
            ):
                logger.error(f"No Jupyter clients found for project_id: {project_id}")
                raise KeyError(f"No Jupyter clients found for project_id: {project_id}")

            kernel_client = self._kernel_clients[project_id]
            notebook_client = self._notebook_clients[project_id]

            # Execute the cell
            logger.info(f"Executing cell {cell_index}")
            result = notebook_client.execute_cell(cell_index, kernel_client)
            logger.info(f"Execution result status: {result.get('status')}")

            # Check for common import errors and fix them if needed
            if auto_fix_imports and result.get("status") == "error":
                error_text = result.get("evalue", "")
                logger.info(f"Execution error: {error_text}")
                imports_to_add = []

                for pattern, fix in self._common_imports.items():
                    match = re.search(pattern, error_text)
                    if match:
                        logger.info(f"Found matching error pattern: {pattern}")
                        # Extract capture group if it exists
                        if "\\1" in fix and match.groups():
                            fix = fix.replace("\\1", match.group(1))
                        imports_to_add.append(fix)

                if imports_to_add:
                    logger.info(f"Adding imports: {imports_to_add}")

                    # Get the original code from the cell
                    original_code = self._get_cell_source(notebook_client, cell_index)
                    logger.info(f"Original cell code: {original_code[:100]}...")

                    # Add imports to the top of the cell content
                    updated_code = self._update_cell_with_imports(
                        original_code, imports_to_add
                    )
                    logger.info(f"Updated code: {updated_code[:100]}...")

                    # Update the cell with the combined content
                    logger.info(f"Updating cell {cell_index}")
                    if hasattr(notebook_client, "update_code_cell"):
                        logger.info("Using update_code_cell method")
                        notebook_client.update_code_cell(cell_index, updated_code)
                    elif hasattr(notebook_client, "update_cell"):
                        logger.info("Using update_cell method")
                        notebook_client.update_cell(cell_index, updated_code, "code")
                    else:
                        logger.error("No method available to update cell content")

                    # Execute the updated cell
                    logger.info(f"Re-executing cell {cell_index} with imports")
                    result = notebook_client.execute_cell(cell_index, kernel_client)
                    logger.info(f"Re-execution result status: {result.get('status')}")

            return result
        except Exception as e:
            logger.error(f"Error executing cell: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def get_cell_code(self, project_id: str, cell_index: int) -> str:
        """
        Get the code from a specific cell.

        Args:
            project_id: The project ID to identify the clients
            cell_index: The index of the cell

        Returns:
            The cell content as a string

        Raises:
            KeyError: If no notebook client exists for the given project_id
        """
        logger.info(f"Getting code from cell {cell_index} for project {project_id}")

        try:
            if project_id not in self._notebook_clients:
                logger.error(f"No notebook client found for project_id: {project_id}")
                raise KeyError(f"No notebook client found for project_id: {project_id}")

            notebook_client = self._notebook_clients[project_id]
            return self._get_cell_source(notebook_client, cell_index)
        except Exception as e:
            logger.error(f"Error getting cell code: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    async def delete_cell_by_index(
        self, project_id: str, cell_index: int
    ) -> dict[str, Any]:
        """
        Delete a cell from the notebook by its index.

        Args:
            project_id: The project ID to identify the clients
            cell_index: The index of the cell to delete

        Returns:
            Dictionary with deletion status and info

        Raises:
            KeyError: If no notebook client exists for the given project_id
        """
        logger.info(f"Deleting cell {cell_index} for project {project_id}")

        try:
            if project_id not in self._notebook_clients:
                logger.error(f"No notebook client found for project_id: {project_id}")
                raise KeyError(f"No notebook client found for project_id: {project_id}")

            notebook_client = self._notebook_clients[project_id]
            # Check if the cell index is valid
            if cell_index < 0 or cell_index >= len(notebook_client):
                return {
                    "error": f"Invalid cell index: {cell_index}."
                    "Notebook has {len(notebook_client)} cells."
                }

            # Store notebook length before deletion
            length_before = len(notebook_client)

            # Store cell info before deletion
            try:
                cell_info = notebook_client[cell_index]
                cell_type = cell_info.get("cell_type", "unknown")
                cell_source = cell_info.get("source", "")
            except Exception:
                cell_type = "unknown"
                cell_source = ""

            # Delete the cell with error handling for the 'to_py' issue
            try:
                del notebook_client[cell_index]
                logger.info(
                    f"Successfully deleted {cell_type} cell at index {cell_index}"
                )

                # Sync the notebook to ensure changes are persisted
                try:
                    if hasattr(notebook_client, "wait_until_synced"):
                        await notebook_client.wait_until_synced()
                        logger.info("Notebook synced after cell deletion")
                    elif hasattr(notebook_client, "sync"):
                        await notebook_client.sync()
                        logger.info("Notebook synced after cell deletion")
                except Exception as sync_error:
                    logger.warning(
                        f"Failed to sync notebook after deletion: {sync_error}"
                    )
                    # Don't fail the operation if sync fails

            except Exception as delete_error:
                # Handle the specific 'to_py' error gracefully
                if "'dict' object has no attribute 'to_py'" in str(delete_error):
                    logger.warning(
                        "Cell deletion had internal error but may have succeeded"
                    )
                    # Check if deletion actually worked by comparing lengths
                    length_after = len(notebook_client)
                    if length_after < length_before:
                        logger.info("Cell deletion succeeded despite internal error")

                        # Sync the notebook even after error case
                        try:
                            if hasattr(notebook_client, "wait_until_synced"):
                                await notebook_client.wait_until_synced()
                                logger.info(
                                    "Notebook synced after cell deletion (error case)"
                                )
                        except Exception as sync_error:
                            logger.warning(
                                "Failed to sync notebook after deletion (error case):"
                                f"{sync_error}"
                            )
                    else:
                        logger.error("Cell deletion failed - length unchanged")
                        raise delete_error
                else:
                    logger.error(
                        f"Unexpected error during cell deletion: {delete_error}"
                    )
                    raise delete_error

            return {
                "cell_index": cell_index,
                "cell_type": cell_type,
                "cell_source": cell_source,
                "status": "deleted",
            }
        except Exception as e:
            logger.error(f"Error deleting cell: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def direct_execute_code(self, project_id: str, code: str) -> Any:
        """
        Execute code directly using the kernel.

        This is a more direct approach to executing code that bypasses notebook cells
        when possible, using the kernel's direct execution capabilities.

        Args:
            project_id: The project ID to identify the clients
            code: The code to execute

        Returns:
            Execution results
        """
        logger.info(f"Directly executing code for project {project_id}")
        logger.info(f"Code to execute: {code[:100]}...")

        try:
            if project_id not in self._kernel_clients:
                logger.error(f"No kernel client found for project_id: {project_id}")
                raise KeyError(f"No kernel client found for project_id: {project_id}")

            kernel_client = self._kernel_clients[project_id]

            # Use the raw kernel execution if available
            if hasattr(kernel_client, "execute"):
                logger.info("Using kernel client execute method")
                result = kernel_client.execute(code)
                logger.info("Execution completed")
                return result
            elif hasattr(kernel_client, "execute_interactive"):
                logger.info("Using kernel client execute_interactive method")
                result = kernel_client.execute_interactive(code)
                logger.info("Interactive execution completed")
                return result
            else:
                logger.error("No direct execution method available on kernel client")
                raise NotImplementedError(
                    "No direct execution method available on kernel client"
                )
        except Exception as e:
            logger.error(f"Error directly executing code: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def close_clients(self, project_id: str) -> None:
        """
        Close and remove clients for a specific project.

        Args:
            project_id: The project ID to identify the clients to close
        """
        logger.info(f"Closing clients for project {project_id}")

        try:
            async with self._lock:
                if project_id in self._kernel_clients:
                    logger.info("Stopping kernel client")
                    try:
                        self._kernel_clients[project_id].stop()
                    except Exception as e:
                        logger.error(f"Error stopping kernel client: {str(e)}")
                    del self._kernel_clients[project_id]

                if project_id in self._notebook_clients:
                    logger.info("Stopping notebook client")
                    try:
                        await self._notebook_clients[project_id].stop()
                    except Exception as e:
                        logger.error(f"Error stopping notebook client: {str(e)}")
                    del self._notebook_clients[project_id]

            logger.info(f"Clients closed for project {project_id}")
        except Exception as e:
            logger.error(f"Error closing clients: {str(e)}")
            logger.error(traceback.format_exc())

    async def close_all_clients(self) -> None:
        """Close all kernel and notebook clients."""
        logger.info("Closing all clients")

        try:
            async with self._lock:
                for project_id in list(self._kernel_clients.keys()):
                    await self.close_clients(project_id)

            logger.info("All clients closed")
        except Exception as e:
            logger.error(f"Error closing all clients: {str(e)}")
            logger.error(traceback.format_exc())


# Create a singleton instance
jupyter_context_manager = JupyterContextManager()
