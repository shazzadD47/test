from uuid import uuid4

from langfuse import get_client
from langfuse.langchain import CallbackHandler


def setup_langfuse_handler(langfuse_session_id: str = None, name: str = None):
    if not langfuse_session_id:
        langfuse_session_id = uuid4().hex
    langfuse = get_client()
    langfuse.update_current_trace(session_id=langfuse_session_id, name=name)
    langfuse_handler = CallbackHandler()
    return langfuse_handler
