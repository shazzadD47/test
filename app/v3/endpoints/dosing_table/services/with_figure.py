from uuid import uuid4

import json_repair
import pandas as pd
from langchain_core.messages import HumanMessage
from langfuse import observe

from app.exceptions.system import AnthropicInternalServerError
from app.utils.image import convert_image_to_base64, get_image_from_url
from app.utils.llms import get_message_text, invoke_llm_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints import Status
from app.v3.endpoints.dosing_table.chains import (
    llm_claude,
    llm_gpt,
    prepare_table_chain,
)
from app.v3.endpoints.dosing_table.constants import (
    DOSING_TABLE_COLUMN_ORDER,
    chain_configs_with_figure,
)
from app.v3.endpoints.dosing_table.langchain_schemas import Table
from app.v3.endpoints.dosing_table.logging import logger
from app.v3.endpoints.dosing_table.prompts.with_figure import (
    DOSE_CALCULATION_PROMPT,
    IMAGE_EXPLANATION_PROMPT_CLAUDE,
    INFORMATION_EXTRACTION_PROMPT,
)
from app.v3.endpoints.dosing_table.services.context_helpers import (
    get_arms_routes_and_doses_contexts,
    process_pdf_file,
)
from app.v3.endpoints.dosing_table.utils import (
    create_empty_table,
    fix_arm_time_starting_from_one,
)


@observe(name="dosing with figure")
def prepare_dosing_table(
    project_id: str, paper_id: str, image_url: str, metadata: dict = None
) -> list[dict]:
    if metadata is None:
        metadata = {}
    session_id = str(uuid4())
    langfuse_handler = setup_langfuse_handler(
        session_id, name="dosing_table_with_figure"
    )

    chain_configs_with_figure["metadata"] = {
        "langfuse_session_id": session_id,
        "project_id": project_id,
        "flag_id": paper_id,
    }
    chain_configs_with_figure["callbacks"] = [langfuse_handler]

    image, media_type = get_image_from_url(image_url, return_media_type=True)
    image = convert_image_to_base64(image)

    pdf_cache_name = f"dosing_table_pdf_{str(uuid4())}.pdf"
    file_details = process_pdf_file(
        flag_id=paper_id,
        cache_name=pdf_cache_name,
    )
    contexts = get_arms_routes_and_doses_contexts(
        project_id=project_id,
        flag_id=paper_id,
        file_details=file_details,
        chain_configs=chain_configs_with_figure,
    )

    prompt = IMAGE_EXPLANATION_PROMPT_CLAUDE.format(contexts=contexts)
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )

    try:
        basic_info = invoke_llm_with_retry(
            llm_claude,
            [message],
            config=chain_configs_with_figure,
        )
    except AnthropicInternalServerError:
        metadata["message"] = "Anthropic server failed or busy."
        metadata["status"] = Status.FAILED.value

        message = {
            "payload": {},
            "metadata": metadata,
        }
        return message

    except Exception:
        logger.exception("Failed to get basic info.")

        metadata["message"] = "Failed to get basic info."
        metadata["status"] = Status.FAILED.value

        message = {
            "payload": {},
            "metadata": metadata,
        }
        return message

    logger.debug(f"Basic info: {get_message_text(basic_info)}\n{'=' * 88}")

    prompt = DOSE_CALCULATION_PROMPT.format(contexts=get_message_text(basic_info))
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )

    counts = invoke_llm_with_retry(
        llm_gpt,
        [message],
        config=chain_configs_with_figure,
    )

    logger.debug(f"Counts: {get_message_text(counts)}\n{'=' * 88}")

    chain = prepare_table_chain(INFORMATION_EXTRACTION_PROMPT, Table)

    try:
        table: Table = chain.invoke(
            {
                "contexts": get_message_text(counts),
            },
            config=chain_configs_with_figure,
        )
    except Exception:
        logger.exception("Failed to prepare dosing table.")

        metadata["message"] = "Failed to prepare dosing table."
        metadata["status"] = Status.FAILED.value

        message = {
            "payload": {},
            "metadata": metadata,
        }
        return message

    table = table.model_dump().get("rows")

    # Handle empty or invalid table data
    if not table or (isinstance(table, list) and len(table) == 0) or table == "":
        logger.warning("No table data extracted from the figure.")

        # Create empty table with proper default values
        final_table = create_empty_table()
        final_table = pd.DataFrame(final_table)
        final_table["ARM_TIME"] = final_table["ARM_START_TIME"].copy()
        final_table.drop(columns=["ARM_START_TIME", "ARM_END_TIME"], inplace=True)
        final_table = final_table.to_json(orient="records")
        final_table = json_repair.loads(final_table)

        metadata["message"] = "No dosing table data could be extracted from the figure."
        metadata["status"] = Status.SUCCESS.value

        message = {
            "payload": final_table,
            "metadata": metadata,
        }

        return message

    table = pd.DataFrame(table)
    table["ARM_TIME"] = table["ARM_START_TIME"].copy()
    table.drop(columns=["ARM_START_TIME", "ARM_END_TIME"], inplace=True)
    table = fix_arm_time_starting_from_one(table)
    table = table[DOSING_TABLE_COLUMN_ORDER]
    table = table.to_json(orient="records")
    final_table = json_repair.loads(table)

    metadata["message"] = "Dosing table prepared successfully"
    metadata["status"] = Status.SUCCESS.value

    message = {
        "payload": final_table,
        "metadata": metadata,
    }

    return message
