import time
from uuid import uuid4

import pandas as pd
from langfuse import observe

from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.standardization.helpers import (
    double_check_standardized_data,
    prepare_final_error_log,
    standardize_using_regex_llm,
)


@observe()
@track_all_llm_costs
def run_standardization(
    unstandardized_dataframe: pd.DataFrame,
    number_columns: list[str] = None,
    statistical_columns: list[str] = None,
    string_columns: list[str] = None,
) -> (pd.DataFrame, dict):
    try:
        langfuse_session_id = uuid4().hex
        setup_langfuse_handler(langfuse_session_id, name="standardization")
        # Summarize unique values
        # summary_table = summarize_unique_values(unstandardized_dataframe,
        # COLUMNS_TO_CHECK)
        start_time = time.time()
        standardized_dataframe, error_log = standardize_using_regex_llm(
            unstandardized_dataframe,
            number_columns=number_columns,
            statistical_columns=statistical_columns,
            string_columns=string_columns,
            langfuse_session_id=langfuse_session_id,
        )
        end_time = time.time()
        logger.debug(f"Time standardization: {end_time - start_time} seconds")
        # summary_table_standardized = summarize_unique_values(
        #     standardized_dataframe, COLUMNS_TO_CHECK
        # )
        # double check saved standardized data
        # with open("standardization_error_log.json", "w") as f:
        #     json.dump(error_log, f, indent=4)
        error_log = double_check_standardized_data(standardized_dataframe, error_log)
        final_error_log = prepare_final_error_log(error_log)
        return standardized_dataframe, final_error_log
    except Exception as e:
        error_message = "Error standardizing data."
        error_message += f" Error message: {str(e)}"
        raise Exception(error_message)
