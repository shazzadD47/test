from langchain_core.messages import HumanMessage, ToolMessage

from app.utils.llms import ainvoke_chain_with_retry
from app.utils.texts import convert_data_to_string
from app.v3.endpoints.agent_chat.utils.retriever import (
    retrieve_context,
)
from app.v3.endpoints.report_generator.configs import settings
from app.v3.endpoints.report_generator.constants import ReportAgents
from app.v3.endpoints.report_generator.helpers import (
    encode_image_from_url,
    get_paper_summaries_by_flag_id,
    parse_html_content,
)
from app.v3.endpoints.report_generator.logging import logger
from app.v3.endpoints.report_generator.schema import ReportContextDocs, ReportState
from app.v3.endpoints.report_generator.services.agents import (
    report_assistant_agent,
    report_edit_agent,
    report_image_edit_agent,
)
from app.v3.endpoints.report_generator.services.tools import (
    report_code_context_retrieval,
    report_paper_context_retrieval,
)
from app.v3.endpoints.report_generator.utils.compression import (
    ensure_file_contents_decompressed,
    is_compressed_file_contents,
)
from app.v3.endpoints.report_generator.utils.token_management import (
    count_tokens_in_messages,
    count_tokens_in_text,
    truncate_contexts_to_fit_limit,
    truncate_text_to_tokens,
)


async def report_assistant_node(state: ReportState, report_type="general"):
    """Node for handling AI Assistant requests (longer content generation)."""
    agent = await report_assistant_agent()
    messages = [HumanMessage(content=state["user_query"])]

    base_tokens = count_tokens_in_messages(messages)
    available_tokens = settings.GEMINI_25_FLASH_INPUT_LIMIT - base_tokens - 1000

    logger.info(
        f"Base message tokens: {base_tokens}, Available for context: {available_tokens}"
    )

    if state.get("flag_id") and not state.get("user_selected_context"):

        context_objects = await retrieve_context(
            state.get("user_query"),
            flag_id=state.get("flag_id"),
            project_id=state.get("project_id"),
            k=20,
        )
        logger.debug(f"Retrieved {len(context_objects)} context objects")

        report_context_objects = [
            ReportContextDocs(
                page_content=ctx.page_content, flag_id=ctx.flag_id, title=ctx.title
            )
            for ctx in context_objects
        ]
        state["sources"] = report_context_objects
        flag_ids = state.get("flag_id")
        summary_map = {}
        summaries_collected = get_paper_summaries_by_flag_id(flag_ids)
        for summary in summaries_collected:
            if summary["paper_summary"] not in [None, ""]:
                summary_map[str(summary["flag_id"])] = summary["paper_summary"]
        if context_objects:
            from app.v3.endpoints.agent_chat.utils.retriever import (
                group_contexts_by_flag_id,
            )

            grouped_contexts = group_contexts_by_flag_id(context_objects)

            context_parts = []
            for flag_id, file_contexts in grouped_contexts.items():
                context_parts.append(f"## Research contexts for flag ID: {flag_id}")
                context_parts.append(
                    "---\n\n".join([context.page_content for context in file_contexts])
                )

                if flag_id in summary_map:
                    context_parts.append(
                        f"Paper Summary Supplement context:\n{summary_map[flag_id]}\n\n"
                    )

                context_parts.append("\n\n")

            context_parts.append(
                "Remember to use the flag ids only for citations under citation tags."
                "But you must never use these flag ids anywhere outside"
                "citations (<citation>...</citation> block) in your answer."
                "NEVER PROVIDE THE flag ids OUTSIDE THE <citation>"
                "TAGS UNDER ANY CIRCUMSTANCES."
                "If a query is asked then only answer the query"
                "with the relevant context in citation."
                "If user wants to generate report instead then give comprehensive"
                "coverage with detailed analysis in multiple sections"
                "(Background, Methods, Results, Conclusions)."
            )

            context_token_budget = int(available_tokens * 0.7)
            truncated_contexts = truncate_contexts_to_fit_limit(
                context_parts, context_token_budget
            )

            contexts = "\n\n".join(truncated_contexts)

            context_message = f"""
            <retrieved_contexts>
            {contexts}
            </retrieved_contexts>

            Use the above contexts along with any provided file contents to generate
            professional report content. Remember to cite sources appropriately.
            If users asks a query instead, then only answer
            the query using the above contexts.
            The flag ids must only be used inside the <citation>
            tags in your response.
            """

            messages.append(HumanMessage(content=context_message))

            context_tokens = count_tokens_in_messages(
                [HumanMessage(content=context_message)]
            )
            available_tokens -= context_tokens
            logger.info(
                f"Context tokens used: {context_tokens}, Remaining: {available_tokens}"
            )

    if state.get("file_contents") and available_tokens > 100:
        try:
            if is_compressed_file_contents(state.get("file_contents")):
                logger.debug("Decompressing file contents in AI service...")
                decompressed_files = ensure_file_contents_decompressed(
                    state.get("file_contents")
                )
                state["file_contents"] = decompressed_files
                logger.debug(
                    f"Successfully decompressed {len(decompressed_files)} files"
                )

            file_contents = ensure_file_contents_decompressed(
                state.get("file_contents")
            )

            if file_contents:
                file_token_budget = int(available_tokens * 0.7)

                file_context_parts = []
                current_file_tokens = 0

                for content in file_contents:
                    file_part = f"""
                File: {content.get('path')}
                Content:
                ```
                {convert_data_to_string(content.get('content'))}
                ```
                """
                    part_tokens = count_tokens_in_text(file_part)

                    if current_file_tokens + part_tokens <= file_token_budget:
                        file_context_parts.append(file_part)
                        current_file_tokens += part_tokens
                    else:
                        remaining_budget = file_token_budget - current_file_tokens
                        if remaining_budget > 200:
                            content_str = convert_data_to_string(content.get("content"))
                            truncated_content = truncate_text_to_tokens(
                                content_str,
                                remaining_budget - 100,
                            )
                            truncated_part = f"""
                File: {content.get('path')} [TRUNCATED]
                Content:
                ```
                {truncated_content}
                ```
                """
                            file_context_parts.append(truncated_part)
                        break

                if file_context_parts:
                    file_context_message = f"""
                    <file_contents>
                    {chr(10).join(file_context_parts)}
                    </file_contents>

                    The above files are provided for context.
                    Use them to inform your response.
                    """
                    messages.append(HumanMessage(content=file_context_message))

                    file_tokens = count_tokens_in_messages(
                        [HumanMessage(content=file_context_message)]
                    )
                    available_tokens -= file_tokens
                    logger.info(
                        f"File content tokens used: {file_tokens}, "
                        f"Remaining: {available_tokens}"
                    )

        except Exception as e:
            logger.error(f"Error processing file contents: {str(e)}")

    if state.get("report_context") and available_tokens > 100:
        report_context = state.get("report_context")
        report_context_tokens = count_tokens_in_text(report_context)

        if report_context_tokens > available_tokens * 0.5:
            report_context = truncate_text_to_tokens(
                report_context, int(available_tokens * 0.5)
            )
            report_context += "... [TRUNCATED]"

        report_contexts = []
        report_contexts.append({"type": "text", "text": "<selected_report_context>"})
        parsed_result = parse_html_content(report_context)

        for result in parsed_result:
            if result["type"] == "image":
                encoded_image = encode_image_from_url(result["value"])
                if encoded_image:
                    report_contexts.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded_image}",
                        }
                    )
            else:
                report_contexts.append({"type": "text", "text": result["value"]})
        report_contexts.append({"type": "text", "text": "<selected_report_context>"})
        report_msg = """
        The above is the current report content. Generate new content that flows
        naturally with the existing content and maintains consistency
        in style and tone.
        If a query is asked based on the selected report context,
        then only answer the query.
        """
        report_contexts.append(
            {
                "type": "text",
                "text": report_msg,
            }
        )

        messages.append(HumanMessage(content=report_contexts))

    if state.get("tool_results"):
        if not state.get("tool_request_message"):
            raise ValueError("Tool request message is required.")
        messages += [state.get("tool_request_message"), *state.get("tool_results")]

    final_token_count = count_tokens_in_messages(messages)
    logger.info(f"Final token count before sending to model: {final_token_count}")

    if final_token_count > settings.GEMINI_25_FLASH_INPUT_LIMIT:
        logger.warning(
            f"Token count ({final_token_count}) still exceeds limit. "
            f"Emergency truncation required."
        )
        messages = [
            messages[0],
            HumanMessage(
                content=(
                    "Generate a response based on the user query above. "
                    "Context was too large to include."
                )
            ),
        ]

    invoke_params = {"messages": messages}
    response = await ainvoke_chain_with_retry(agent, invoke_params)

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = ReportAgents.ASSISTANT_AGENT
    state["tool_results"] = []

    logger.info("Report assistant agent response generated")
    return state


async def report_edit_node(state: ReportState):
    """Node for handling AI Edit requests (shorter, focused edits)."""
    if state.get("selected_report_context"):
        agent = await report_image_edit_agent()
    else:
        agent = await report_edit_agent()
    messages = [HumanMessage(content=state["user_query"])]
    if state.get("selected_text"):
        edit_context = f"""

        <selected_text_to_edit>
        {state.get("selected_text")}
        </selected_text_to_edit>

        Please provide an improved version of the selected text above.
        Focus on enhancing clarity, grammar, and professional tone while
        maintaining the original meaning.
        """
        messages.append(HumanMessage(content=edit_context))

    if state.get("selected_report_context"):
        report_context = state.get("selected_report_context")
        report_contexts = []
        report_contexts.append({"type": "text", "text": "<current_report_context>"})
        parsed_result = parse_html_content(report_context)

        for result in parsed_result:
            if result["type"] == "image":
                encoded_image = encode_image_from_url(result["value"])
                if encoded_image:
                    report_contexts.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded_image}",
                        }
                    )
            else:
                report_contexts.append({"type": "text", "text": result["value"]})
        report_contexts.append({"type": "text", "text": "<current_report_context>"})
        report_context_msg = """
            The above is the current report content.
            Generate new content that flows naturally with the existing content
            and maintains consistency in style and tone.
            If asked a query on this content then only answer the query
            """
        report_contexts.append(
            {
                "type": "text",
                "text": report_context_msg,
            }
        )

        messages.append(HumanMessage(content=report_contexts))

    if state.get("file_contents"):
        try:
            if is_compressed_file_contents(state.get("file_contents")):
                logger.debug("Decompressing file contents in AI service...")
                decompressed_files = ensure_file_contents_decompressed(
                    state.get("file_contents")
                )
                state["file_contents"] = decompressed_files
                logger.debug(
                    f"Successfully decompressed {len(decompressed_files)} files"
                )

            file_contents = ensure_file_contents_decompressed(
                state.get("file_contents")
            )

            if file_contents:
                file_context_parts = []
                for content in file_contents:
                    file_context_parts.append(
                        f"""
                File: {content.get('path')}
                Content:
                ```
                {convert_data_to_string(content.get('content'))}
                ```
                """
                    )

                file_context_message = (
                    f"\n\n        <file_contents>\n"
                    f"        {chr(10).join(file_context_parts)}\n"
                    f"        </file_contents>\n\n"
                    f"        The above files are provided for context. Consider them "
                    f"when providing your response.\n        "
                )
                messages.append(HumanMessage(content=file_context_message))

        except Exception as e:
            logger.error(f"Error processing file contents: {str(e)}")

    if state.get("report_context"):
        context_msg = f"""

        <surrounding_report_context>
        {state.get("report_context")}
        </surrounding_report_context>

        Consider the above context to ensure your edit fits naturally
        with the surrounding content.
        """
        messages.append(HumanMessage(content=context_msg))

    invoke_params = {"messages": messages}
    response = await ainvoke_chain_with_retry(agent, invoke_params)

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = ReportAgents.EDIT_AGENT
    state["tool_results"] = []

    logger.info("Report edit agent response generated")
    return state


async def report_insights_node(state: ReportState):
    agent = await report_image_edit_agent()
    logger.debug("Using report image edit agent for insights generation")
    messages = [HumanMessage(content=state["user_query"])]
    state["sources"] = []
    if state.get("selected_report_context"):
        report_context = state.get("selected_report_context")

        report_contexts = []
        report_contexts.append({"type": "text", "text": "<current_report_context>"})
        parsed_result = parse_html_content(report_context)

        for result in parsed_result:
            if result["type"] == "image":
                encoded_image = encode_image_from_url(result["value"])
                report_contexts.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{encoded_image}",
                    }
                )
            else:
                report_contexts.append({"type": "text", "text": result["value"]})
        report_contexts.append({"type": "text", "text": "<current_report_context>"})
        report_context_msg = """
            The above is the selected report context.
            Generate insights that are relevant, clear,
            and focused on the given context
            while maintaining consistency in style and tone.
            Insights should be relevant to the user query.
            If asked a query on this content then only answer the query
            """
        report_contexts.append(
            {
                "type": "text",
                "text": report_context_msg,
            }
        )

        messages.append(HumanMessage(content=report_contexts))
    if state.get("report_context"):
        context_msg = f"""

            <surrounding_report_context>
            {state.get("report_context")}
            </surrounding_report_context>

            Consider the above context to ensure your edit fits naturally
            with the surrounding content.
            """
        messages.append(HumanMessage(content=context_msg))

    invoke_params = {"messages": messages}
    response = await ainvoke_chain_with_retry(agent, invoke_params)

    if response.tool_calls:
        state["messages"] = []
        state["tool_request_message"] = response
        state["tool_calls"] = response.tool_calls
    else:
        state["messages"] = [response]
        state["tool_request_message"] = None
        state["tool_calls"] = []

    state["sender"] = ReportAgents.INSIGHT_AGENT
    state["tool_results"] = []

    logger.info("Report edit agent response generated")
    return state


async def report_tool_node(state: ReportState):
    """Execute tools for report generation."""
    results = []
    tool_calls = state.get("tool_calls", [])

    tools_by_name = {
        report_paper_context_retrieval.name: report_paper_context_retrieval,
        report_code_context_retrieval.name: report_code_context_retrieval,
    }

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

        try:
            if hasattr(tool_to_call, "ainvoke"):
                result = await tool_to_call.ainvoke(tool_args)
            else:
                result = tool_to_call.invoke(tool_args)

            if tool_name in [
                report_paper_context_retrieval.name,
                report_code_context_retrieval.name,
            ]:
                state["sources"] = result
                contents = [context.page_content for context in result]
                result = "---\n\n".join(contents)

        except Exception as e:
            results.append(
                ToolMessage(
                    content=f"Error executing tool {tool_name}: {str(e)}",
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                )
            )
            continue

        results.append(
            ToolMessage(
                content=result, tool_call_id=tool_call["id"], name=tool_call["name"]
            )
        )

    state["tool_results"] = results
    state["messages"] = []

    return state
