from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.auto import AutoChatModel
from app.v3.endpoints.extraction_templates.agent_services.prompts import (
    MAIN_AGENT_PROMPT,
)
from app.v3.endpoints.extraction_templates.agent_services.tools import (
    add_input_row,
    add_output_row,
    delete_input_row,
    delete_output_row,
    describe_project,
    list_project_files,
    read_current_extraction_schema,
    read_file,
    suggest_actions_to_user,
    update_input_row,
    update_output_row,
    update_table_info,
)
from app.v3.endpoints.extraction_templates.configs import settings

agent_tools = [
    describe_project,
    read_current_extraction_schema,
    add_input_row,
    add_output_row,
    delete_input_row,
    delete_output_row,
    update_input_row,
    update_output_row,
    update_table_info,
    suggest_actions_to_user,
    read_file,
    list_project_files,
]


def get_main_agent():
    agent = AutoChatModel.from_model_name(settings.MAIN_AGENT, temperature=0.5)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MAIN_AGENT_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    if agent_tools:
        agent = agent.bind_tools(agent_tools)

    return prompt | agent
