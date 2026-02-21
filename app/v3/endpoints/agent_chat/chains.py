import threading

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from app.core.auto import AutoChatModel
from app.v3.endpoints.agent_chat.configs import settings as agent_chat_settings
from app.v3.endpoints.agent_chat.langchain_schema import DataFileChoice
from app.v3.endpoints.agent_chat.prompts import DATA_FILE_CHOICE_PROMPT

# Lazy-initialized to speed up app startup
_code_llm = None
_code_llm_lock = threading.Lock()


def get_code_llm():
    """Get or create the code LLM instance (thread-safe)."""
    global _code_llm
    if _code_llm is None:
        with _code_llm_lock:
            # Double-check pattern to avoid race conditions
            if _code_llm is None:
                _code_llm = AutoChatModel.from_model_name(
                    agent_chat_settings.CODE_LLM, temperature=0.2
                )
    return _code_llm


parser = PydanticOutputParser(pydantic_object=DataFileChoice)
output_instructions = parser.get_format_instructions()


def data_file_chooser_chain() -> RunnableSerializable:
    prompt = PromptTemplate.from_template(DATA_FILE_CHOICE_PROMPT)
    prompt = prompt.partial(output_instructions=output_instructions)

    chain = prompt | get_code_llm() | parser | (lambda x: x.file_path)

    return chain
