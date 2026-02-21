from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.auto import AutoChatModel
from app.v3.endpoints.report_generator.configs import settings as report_settings
from app.v3.endpoints.report_generator.prompts import (
    REPORT_AI_ASSISTANT_SYSTEM_PROMPT,
    REPORT_AI_EDIT_SYSTEM_PROMPT,
    REPORT_AI_INSIGHTS_SYSTEM_PROMPT,
)
from app.v3.endpoints.report_generator.services.tools import (
    report_code_context_retrieval,
    report_paper_context_retrieval,
)


async def report_assistant_agent():
    """Agent for AI Assistant - longer content generation for reports."""
    tools = [
        report_paper_context_retrieval,
        report_code_context_retrieval,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_AI_ASSISTANT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    today = datetime.now().strftime("%Y-%m-%d %A")
    prompt = prompt.partial(date=today)

    llm = AutoChatModel.from_model_name(
        report_settings.ASSISTANT_LLM,
        temperature=0.3,
    )

    chain = prompt | llm.bind_tools(tools)
    return chain


async def report_edit_agent():
    """Agent for AI Edit - shorter, focused edits."""
    tools = []

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_AI_EDIT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    llm = AutoChatModel.from_model_name(
        report_settings.EDIT_LLM,
        temperature=0.1,
    )

    chain = prompt | llm.bind_tools(tools)
    return chain


async def report_image_edit_agent():
    """Agent for AI Edit - shorter, focused edits."""
    tools = []

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_AI_INSIGHTS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    llm = AutoChatModel.from_model_name(
        report_settings.IMAGE_EDIT_LLM,
        temperature=0.2,
    )

    chain = prompt | llm.bind_tools(tools)
    return chain
