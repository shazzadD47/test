import threading
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.auto import AutoChatModel
from app.v3.endpoints.agent_chat.configs import settings as agent_chat_settings
from app.v3.endpoints.agent_chat.constants import Agents
from app.v3.endpoints.agent_chat.prompts import (
    CODE_GENERATOR_SYSTEM_PROMPT_V1,
    CODE_GENERATOR_SYSTEM_PROMPT_V2,
    DEEP_AGENT_SYSTEM_PROMPT,
    RAG_AGENT_SYSTEM_PROMPT,
)
from app.v3.endpoints.agent_chat.services.tools import (
    add_and_execute_code,
    code_context_retrieval,
    create_markdown_cell,
    delete_cell,
    execute_python_code,
    get_cell_info,
    get_notebook_structure,
    modify_cell_content,
    paper_context_retrieval,
)

from ..schema import routeResponse

# Lazy-initialized LLM instances to speed up app startup
_llm = None
_code_llm = None
_llm_lock = threading.Lock()
_code_llm_lock = threading.Lock()


def get_llm():
    """Get or create the main LLM instance (thread-safe)."""
    global _llm
    if _llm is None:
        with _llm_lock:
            # Double-check pattern to avoid race conditions
            if _llm is None:
                _llm = AutoChatModel.from_model_name(
                    agent_chat_settings.CHAT_LLM, temperature=0.2
                )
    return _llm


def get_code_llm():
    """Get or create the code LLM instance (thread-safe)."""
    global _code_llm
    if _code_llm is None:
        with _code_llm_lock:
            # Double-check pattern to avoid race conditions
            if _code_llm is None:
                _code_llm = AutoChatModel.from_model_name(
                    agent_chat_settings.CODE_LLM, temperature=0.2, max_tokens=8192
                )
    return _code_llm


agent_names = {Agents.MAIN_AGENT.value, Agents.CODE_GENERATOR.value}


async def main_agent():
    tools = [routeResponse]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(agent_names=list(agent_names - {Agents.MAIN_AGENT.value}))
    prompt = prompt.partial(
        tools=[f"{tool.name}: {tool.description}" for tool in tools[:-1]]
    )

    today = datetime.now().strftime("%Y-%m-%d %A")
    prompt = prompt.partial(date=today)

    chain = prompt | get_llm().bind_tools(tools)
    return chain


async def gemini_agent(cache_name: str | None = None):
    tools = []

    messages = []
    if not cache_name:
        messages.append(("system", DEEP_AGENT_SYSTEM_PROMPT))

    messages.append(MessagesPlaceholder(variable_name="messages"))

    prompt = ChatPromptTemplate.from_messages(messages)

    today = datetime.now().strftime("%Y-%m-%d %A")
    prompt = prompt.partial(date=today)

    gemini_llm = AutoChatModel.from_model_name(
        agent_chat_settings.CHAT_GEMINI_MODEL,
        cached_content=cache_name,
        temperature=0.2,
    )

    if not cache_name:
        tool_names = [f"{tool.name}: {tool.description}" for tool in tools[:-1]]
        tool_message = (
            f"You have access to the following tools:\n<tools>\n{tool_names}\n</tools>"
        )

        prompt = prompt.partial(tools=tool_message)
        gemini_llm = gemini_llm.bind_tools(tools)

    chain = prompt | gemini_llm

    return chain


async def deep_chat_fallback_agent():
    tools = []

    messages = []
    messages.append(("system", DEEP_AGENT_SYSTEM_PROMPT))

    messages.append(MessagesPlaceholder(variable_name="messages"))

    prompt = ChatPromptTemplate.from_messages(messages)

    today = datetime.now().strftime("%Y-%m-%d %A")
    prompt = prompt.partial(date=today)

    deep_chat_llm = AutoChatModel.from_model_name(
        agent_chat_settings.DEEP_CHAT_FALLBACK_LLM,
        temperature=0.2,
    )

    tool_names = [f"{tool.name}: {tool.description}" for tool in tools[:-1]]
    tool_message = (
        f"You have access to the following tools:\n<tools>\n{tool_names}\n</tools>"
    )

    prompt = prompt.partial(tools=tool_message)
    deep_chat_llm = deep_chat_llm.bind_tools(tools)

    chain = prompt | deep_chat_llm

    return chain


async def code_generator_agent_v2(
    language: str = "python",
    language_version: str = "3.12",
):
    tools = [
        paper_context_retrieval,
        code_context_retrieval,
        execute_python_code,
        create_markdown_cell,
        modify_cell_content,
        get_cell_info,
        get_notebook_structure,
        add_and_execute_code,
        delete_cell,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CODE_GENERATOR_SYSTEM_PROMPT_V2),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(
        tools=[f"{tool.name}: {tool.description}" for tool in tools],
        language=language,
        version=language_version,
    )

    chain = prompt | get_code_llm().bind_tools(tools)
    return chain


async def code_generator_agent_v1(file_contents):
    tools = [paper_context_retrieval]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CODE_GENERATOR_SYSTEM_PROMPT_V1),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(
        tools=[f"{tool.name}: {tool.description}" for tool in tools],
        file_contents=file_contents,
    )

    return prompt | get_code_llm().bind_tools(tools)
