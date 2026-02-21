import asyncio
import copy
import os
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.configs import settings
from app.utils.download import async_download_files_from_flag_id
from app.utils.files import create_file_input
from app.utils.llms import ainvoke_chain_with_retry
from app.utils.texts import convert_data_to_string
from app.v3.endpoints.agent_chat.configs import settings as chat_settings
from app.v3.endpoints.agent_chat.constants import Agents
from app.v3.endpoints.agent_chat.logging import logger
from app.v3.endpoints.agent_chat.schema import AgentState
from app.v3.endpoints.agent_chat.services.agents import (
    code_generator_agent_v1,
    code_generator_agent_v2,
    deep_chat_fallback_agent,
    gemini_agent,
    main_agent,
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
from app.v3.endpoints.agent_chat.utils.files import (
    get_code_file_contents,
    get_data_file_contents,
    get_relevant_data_file_paths,
    is_pdf_content,
)
from app.v3.endpoints.agent_chat.utils.retriever import (
    group_contexts_by_flag_id,
    retrieve_context,
)
from app.v3.endpoints.agent_chat.utils.utils import excel_to_csv


async def _download_file(flag_id: str) -> dict[str, str | None]:
    file_path = await async_download_files_from_flag_id(
        flag_id, return_supplementaries=True
    )

    if not file_path:
        return {"flag_id": flag_id, "file_path": file_path}

    if isinstance(file_path, dict):
        path_for_check = file_path.get("main_file")

        if not path_for_check:
            return {"flag_id": flag_id, "file_path": file_path}
    else:
        path_for_check = file_path

    suffix = Path(path_for_check).suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        csv_file_paths = excel_to_csv(path_for_check)
        if csv_file_paths:
            file_path = {
                "main_file": csv_file_paths[0],
                "supplementaries": (
                    csv_file_paths[1:] if len(csv_file_paths) > 1 else []
                ),
            }

    return {"flag_id": flag_id, "file_path": file_path}


async def setup_node(state: AgentState):
    state["messages"] = []

    if state.get("agent_chat_type", "chat") == "chat":
        state["next_agent"] = Agents.MAIN_AGENT

        logger.debug("Routing chat to main agent (RAG)")
        return state
    elif state.get("agent_chat_type") == "deep_chat":
        state["next_agent"] = Agents.DEEP_AGENT

        logger.debug("Continuing chat with deep agent (RAG)")
    elif state.get("agent_chat_type") == "agent":
        state["next_agent"] = Agents.CODE_GENERATOR

        logger.debug("Routing chat to code generator")
        return state

    flag_ids = state.get("flag_id")
    if not flag_ids and not state.get("file_contents") and state.get("project_id"):
        logger.debug("No flag IDs found, Retrieving contexts from project")

        state["sources"] = await retrieve_context(
            state.get("user_query"),
            project_id=state.get("project_id"),
            k=50,
        )

        return state

    if not flag_ids:
        flag_ids = []

    if isinstance(flag_ids, str):
        flag_ids = [flag_ids]

    if flag_ids and (
        state.get("files_for_flag_ids", set()) | state.get("failed_flag_ids", set())
    ) == set(flag_ids):
        logger.debug("Files for flag IDs are already in context, skipping setup")

        state["sources"] = []
        return state

    invokes = []
    for flag_id in flag_ids:
        invoke = _download_file(flag_id)
        invokes.append(invoke)

    results = await asyncio.gather(*invokes)

    failed_flag_ids = set()
    contents = [
        {
            "type": "text",
            "text": """
            You must use the following files to answer the user's question for now
            and next messages. If the user's question is not related to the files,
            you must say that you do not have the information to answer the question.
            Cite the content and file against any information you use. Use this format
            to cite the content:

            <citation><flag_id>...</flag_id><page_no>...</page_no><content>...</content></citation>

            You must strictly follow the above citation format.
            Do not make any changes to this citation format while citing the content.
            Do not use any other format to cite the content.

            If one or more files are empty, then do not make up information
            or hallucinate.
            """,
        }
    ]

    for result in results:
        flag_id = result["flag_id"]
        file_path = result["file_path"]

        if not file_path["main_file"]:
            failed_flag_ids.add(flag_id)
            continue

        main_file_path = file_path["main_file"]

        contents.append(
            {
                "type": "text",
                "text": f"""
                \n\n Following is the file for the file/flag id: {flag_id}.
                You must refer to this flag id in your citations.
                But you must never use this flag id anywhere outside
                citations (<citation>...</citation> block) in your answer.
                Use the name/title of the file to refer to this file in your answer.
                You must never use the flag id to refer to this file in your answer.
                Only use the flag id in your citations.

                If the file is empty, ignore this file.
                """,
            }
        )

        contents.append(
            create_file_input(main_file_path, chat_settings.CHAT_GEMINI_MODEL)
        )

        supplementary_files = file_path["supplementaries"]
        if len(supplementary_files) > 0:
            contents.append(
                {
                    "type": "text",
                    "text": f"""
                    \n\n You are given additional supplementary files for
                    the file/flag id: {flag_id}. Each supplementary file has
                    a unique flag ID. You must refer to the specific supplementary
                    file's flag id in your citations.
                    But you must never use this flag id anywhere outside
                    citations (<citation>...</citation> block) in your answer.
                    Use the name/title of the file to refer to this file in your answer.
                    You must never use the flag id to refer to this file in your answer.
                    Only use the flag id in your citations.
                    """,
                }
            )

            for suppl_file_path in supplementary_files:
                suppl_file_path = Path(suppl_file_path)
                contents.append(
                    {
                        "type": "text",
                        "text": f"""
                        \n\n Following is the supplementary file with flag id:
                        {suppl_file_path.name.removesuffix('.pdf')}.
                        You must refer to the supplementary file's flag id
                        in your citations. But you must never use the
                        supplementary file's flag id anywhere outside
                        citations (<citation>...</citation> block) in your answer.
                        Use the name/title of the file to refer to this file
                        in your answer. You must never use the flag id to
                        refer to this file in your answer. Only use the flag
                        id in your citations.

                        If the file is empty, ignore this file.
                        """,
                    }
                )

                contents.append(
                    create_file_input(suppl_file_path, chat_settings.CHAT_GEMINI_MODEL)
                )

    # add file contents if present
    file_contents = state.get("file_contents")

    if file_contents and isinstance(file_contents, list):
        for content in file_contents:
            contents.append(
                {
                    "type": "text",
                    "text": f"""
                    \n\n Following is a file stored in path: {content.get('path')} which
                    does not have any flag id. You must use the path of this
                    file inside the <content>...</content> block to cite this file
                    if you are using the content of this file in the answer.

                    For example, if the path of the file is 'src/controller/controller.py',
                    you should cite the file in the following way:
                    <citation><flag_id>null</flag_id><page_no>null</page_no><content>src/controller/controller.py</content></citation>
                    Notice that the flag_id and page_no inside citation tag will be null in this case.

                    You must strictly follow the above citation format to cite files
                    without flag ids.

                    If the file is empty, ignore this file.
                    """,  # noqa: E501
                }
            )

            # Process file content - extract text from PDFs to reduce token usage
            file_content_raw = content.get("content")
            file_path = content.get("path", "unknown")
            is_stash_file = content.get("is_stash_file", False)

            if (
                is_stash_file
                and isinstance(file_content_raw, bytes)
                and is_pdf_content(file_content_raw)
            ):
                # save the data to cache for temporary use
                save_path = settings.PDF_CACHE_DIR / f"{uuid4()}.pdf"
                # check if file is a base64 encoded string
                with open(save_path, "wb") as f:
                    f.write(file_content_raw)
                contents.append(
                    create_file_input(save_path, chat_settings.CHAT_GEMINI_MODEL)
                )
                if os.path.exists(save_path):
                    os.remove(save_path)
            else:
                # Use standard conversion for non-stash files or non-PDF content
                file_text = convert_data_to_string(file_content_raw)
                contents.append(
                    {
                        "type": "text",
                        "text": f"File content: \n\n {file_text}",
                    }
                )

    state["messages"] = [HumanMessage(content=contents)]
    state["files_for_flag_ids"] = set(flag_ids)
    state["failed_flag_ids"] = failed_flag_ids
    state["sources"] = []

    logger.debug("Gemini files created")
    return state


async def deep_agent_node(
    state: AgentState,
):
    agent = await gemini_agent(state.get("gemini_cache"))

    messages = copy.deepcopy(state["messages"])

    if state.get("failed_flag_ids"):
        contexts = await retrieve_context(
            state.get("user_query"),
            project_id=state.get("project_id"),
            flag_id=list(state.get("failed_flag_ids")),
            k=max(75, 2 * len(state.get("failed_flag_ids"))),
        )
        state["sources"].extend(contexts)

    if state.get("sources"):
        grouped_contexts = group_contexts_by_flag_id(state.get("sources"))

        contexts = []
        for flag_id, file_contexts in grouped_contexts.items():
            contexts.append(f"## Following are the contexts for the flag ID: {flag_id}")
            contexts.append(
                "---\n\n".join([context.page_content for context in file_contexts])
            )
            contexts.append("\n\n")

        contexts.append(
            "Remember to use the flag IDs only for citations and NEVER use this flag "
            "ID in your answer. Use these contexts along with the other raw files to "
            "answer the user's question."
        )
        contexts = "\n\n".join(contexts)
        messages.append(HumanMessage(content=contexts))

    if state.get("tool_results"):
        if not state.get("tool_request_message"):
            raise ValueError("Tool request message is required.")

        messages += [state.get("tool_request_message"), *state.get("tool_results")]

    # Pass all relevant parameters to the agent
    invoke_params = {
        "messages": messages,
    }

    try:
        response = await ainvoke_chain_with_retry(agent, invoke_params)
    except Exception as e:
        logger.error(f"Error in Gemini agent: {e}")
        fallback_agent = await deep_chat_fallback_agent()
        response = await ainvoke_chain_with_retry(fallback_agent, invoke_params)

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = Agents.DEEP_AGENT
    state["tool_results"] = []

    logger.info(f"Main agent response: {response.usage_metadata}")

    return state


async def rag_agent_node(state: AgentState):
    agent = await main_agent()
    messages = copy.deepcopy(state["messages"])

    last_human_message = None
    for message in messages[::-1]:
        if isinstance(message, HumanMessage):
            last_human_message = message
            break

    context_objects = await retrieve_context(
        last_human_message.content,
        flag_id=state.get("flag_id"),
        project_id=state.get("project_id"),
    )
    state["sources"] = context_objects

    contexts = [context.page_content for context in context_objects]
    contexts = "---\n\n".join(contexts)

    if state.get("tool_results"):
        if not state.get("tool_request_message"):
            raise ValueError("Tool request message is required.")

        messages += [state.get("tool_request_message"), *state.get("tool_results")]

    # Pass all relevant parameters to the agent
    invoke_params = {
        "messages": messages,
        "contexts": contexts,
        "current_notebook_path": state.get("current_notebook_path"),
        "file_contents": state.get("file_contents"),
        "project_id": state.get("project_id"),
    }

    response = await agent.ainvoke(invoke_params)

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = Agents.MAIN_AGENT
    state["tool_results"] = []

    return state


async def code_generator_node(state: AgentState):
    """
    Code generator node that routes to either:
    - External NotebookAgent API (for agent_chat_type="agent" with notebook params)
    - Internal LangChain agent (for agent_chat_type="chat" or fallback)
    """
    # Check if we should use the external NotebookAgent API
    use_notebook_agent = (
        state.get("agent_chat_type") == "agent"
        and state.get("notebook_url")
        and state.get("notebook_token")
        and state.get("project_id")
    )

    if use_notebook_agent:
        logger.info("Using external NotebookAgent API for code generation")
        return await _code_generator_with_notebook_agent(state)
    else:
        logger.info("Using internal LangChain agent for code generation")
        return await _code_generator_with_langchain(state)


async def _code_generator_with_notebook_agent(state: AgentState):
    """
    Use the external NotebookAgent API to generate code.

    This sets a flag in state to tell chat.py to handle streaming directly,
    since we need to stream responses in real-time rather than waiting for completion.
    """
    # Set a flag indicating that chat.py should handle the notebook agent streaming
    state["use_notebook_agent_streaming"] = True
    state["sender"] = Agents.CODE_GENERATOR

    # Create a placeholder message
    ai_message = AIMessage(
        content="Starting notebook agent...",
        additional_kwargs={"source": "external_notebook_agent_placeholder"},
    )

    state["messages"] = [ai_message]
    state["tool_request_message"] = None
    state["tool_calls"] = []
    state["tool_results"] = []
    state["sources"] = []

    return state


async def _code_generator_with_langchain(state: AgentState):
    """
    Use the internal LangChain agent for code generation
    (original implementation).
    """
    relevant_paths = []
    data_content, code_content = "", ""

    if state.get("file_contents"):
        relevant_paths = get_relevant_data_file_paths(
            state["user_query"],
            state["file_contents"],
            state.get("user_selected_context"),
        )

        data_content = get_data_file_contents(state["file_contents"], relevant_paths)
        code_content = get_code_file_contents(
            state["file_contents"], state["current_notebook_path"]
        )

    if state.get("agent_chat_type") == "chat":
        agent = await code_generator_agent_v1(
            file_contents=state.get("file_contents"),
        )
    elif state.get("agent_chat_type") == "agent":
        agent = await code_generator_agent_v2(
            language=state.get("language", "python3"),
            language_version=state.get("language_version", "3.12"),
        )
    else:
        raise ValueError("Invalid agent chat type")

    messages = copy.deepcopy(state["messages"])

    if state.get("tool_results"):
        if not state.get("tool_request_message"):
            raise ValueError("Tool request message is required.")

        messages += [state.get("tool_request_message"), *state.get("tool_results")]

    response = await agent.ainvoke(
        {
            "notebook": code_content,
            "data": data_content,
            "query": state["user_query"],
            "messages": messages,
            "flag_id": state.get("flag_id"),
            "project_id": state.get("project_id"),
            "current_notebook_path": state.get("current_notebook_path"),
        }
    )

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = Agents.CODE_GENERATOR
    state["tool_results"] = []
    state["sources"] = []

    return state


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
tools_by_name = {tool.name: tool for tool in tools}


async def tool_node(state: AgentState):
    results = []
    tool_calls = state.get("tool_calls", [])

    # Get project_id from state
    project_id = state.get("project_id")

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        if tool_name not in tools_by_name:
            results.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                )
            )
            continue

        tool_to_call = tools_by_name[tool_name]
        tool_args = tool_call.get("args", {})

        final_args = tool_args.copy()

        # For JupyterContextManager-based tools, ensure project_id is included
        if (
            tool_name
            in [
                create_markdown_cell.name,
                modify_cell_content.name,
                get_cell_info.name,
                get_notebook_structure.name,
                add_and_execute_code.name,
                delete_cell.name,
            ]
            and project_id
            and "project_id" not in final_args
        ):
            # Add project_id to the arguments
            final_args["project_id"] = project_id

        try:
            # Call the tool with the final arguments map
            if hasattr(tool_to_call, "ainvoke"):
                result = await tool_to_call.ainvoke(final_args)
            else:
                result = tool_to_call.invoke(final_args)
        except Exception as e:
            results.append(
                ToolMessage(
                    content=f"Error executing tool {tool_name}: {str(e)}",
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                )
            )
            continue

        if tool_call["name"] in [
            code_context_retrieval.name,
            paper_context_retrieval.name,
        ]:
            awaited_result = await result
            state["sources"] = awaited_result
            contents = [context.page_content for context in awaited_result]
            result = "---\n\n".join(contents)

        results.append(
            ToolMessage(
                content=result, tool_call_id=tool_call["id"], name=tool_call["name"]
            )
        )

    state["tool_results"] = results
    state["messages"] = []

    return state
