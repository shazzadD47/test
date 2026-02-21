"""
Notebook Chat Agent Integration

This module provides integration with an external notebook agent service
that can execute code and interact with Jupyter notebooks via SSE streaming.
"""

from app.v3.endpoints.agent_chat.notebook_chat.agent import (
    NotebookAgent,
    NotebookAgentResponse,
)

__all__ = ["NotebookAgent", "NotebookAgentResponse"]
