from uuid import uuid4

import json_repair
import pandas as pd
from celery.utils.log import get_task_logger
from langfuse import observe

from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints import Status
from app.v3.endpoints.dosing_table.chains import (
    dose_calculation_chain,
    prepare_table_chain,
)
from app.v3.endpoints.dosing_table.constants import (
    DOSING_TABLE_COLUMN_ORDER,
    chain_configs_without_figure,
)
from app.v3.endpoints.dosing_table.langchain_schemas import Table
from app.v3.endpoints.dosing_table.prompts.with_figure import (
    INFORMATION_EXTRACTION_PROMPT,
)
from app.v3.endpoints.dosing_table.services.context_helpers import (
    get_study_design_contexts,
    process_pdf_file,
)
from app.v3.endpoints.dosing_table.utils import (
    create_empty_table,
    fix_arm_time_starting_from_one,
)

logger = get_task_logger("dosing_table")


@observe(name="dosing table no figure")
def prepare_dosing_table_no_figure(
    project_id: str,
    flag_id: str,
    metadata: dict,
) -> list[dict]:
    if metadata is None:
        metadata = {}
    session_id = str(uuid4())
    langfuse_handler = setup_langfuse_handler(
        session_id, name="dosing_table_without_figure"
    )

    chain_configs_without_figure["metadata"] = {
        "langfuse_session_id": session_id,
        "project_id": project_id,
        "flag_id": flag_id,
    }
    chain_configs_without_figure["callbacks"] = [langfuse_handler]
    pdf_cache_name = f"dosing_table_pdf_{str(uuid4())}.pdf"
    file_details = process_pdf_file(
        flag_id,
        cache_name=pdf_cache_name,
    )
    study_design = get_study_design_contexts(
        project_id=project_id,
        flag_id=flag_id,
        file_details=file_details,
        chain_configs=chain_configs_without_figure,
    )

    logger.info(f"Study design: {study_design}")

    dose_calculator = dose_calculation_chain()
    dose_calculation = dose_calculator.invoke(
        {"contexts": study_design}, config=chain_configs_without_figure
    )
    logger.info(f"Dose calculation: {dose_calculation}")

    chain = prepare_table_chain(INFORMATION_EXTRACTION_PROMPT, Table)

    try:
        table = chain.invoke(
            {"contexts": dose_calculation}, config=chain_configs_without_figure
        )
    except Exception:
        # TODO: Handle this error
        logger.exception("Failed to prepare dosing table.")

        metadata["message"] = "Failed to prepare dosing table."
        metadata["status"] = Status.FAILED.value

        message = {
            "payload": {},
            "metadata": metadata,
        }

        return message

    logger.info(f"Dosing table: {table}")

    # standardize the units. Convert the unit to the highest
    # possbile unit so that every number is a natural number
    # and not in fractions. If fraction is inevitable, then
    # keep it as the lowest unit : hour
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

        metadata["message"] = "No dosing table data could be extracted from the paper."
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

    metadata["message"] = "Dosing table no figure process completed"
    metadata["status"] = Status.SUCCESS.value

    message = {
        "payload": final_table,
        "metadata": metadata,
    }

    return message
