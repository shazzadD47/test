import re
import traceback
from io import StringIO
from uuid import uuid4

import pandas as pd

from app.utils.download import download_file_from_url
from app.v3.endpoints.merging.constants import TableNames
from app.v3.endpoints.merging.group_qc import run_group_qc
from app.v3.endpoints.merging.load_tables import load_and_parse_tables
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.merge import run_whole_merge
from app.v3.endpoints.merging.merge_flow import run_merge_flow
from app.v3.endpoints.merging.schemas import MergeResponse, SingleError
from app.v3.endpoints.merging.standardization import run_standardization
from app.v3.endpoints.merging.utils import (
    check_if_null,
    clean_numerical_columns,
    expand_plot_points,
)


def remove_empty_errors(error_log: dict) -> dict:
    updated_error_log = {}
    for key, value in error_log.items():
        if value == "" or check_if_null(value):
            continue
        else:
            value_without_brackets = re.sub(r"[()\[\]{}]", "", value).strip()
            if value_without_brackets == "":
                continue
            else:
                updated_error_log[key] = value
    return updated_error_log


def prepare_error_log(error_log: dict) -> list[dict]:
    error_log_list = []
    for key, value in error_log.items():
        error_log_list.append({"error_name": key, "error_message": value})
    return error_log_list


def run_merge(table_dict: dict) -> MergeResponse:
    """
    V1 merge flow: load tables from payload URLs, run merge pipeline, return CSV.

    Before (input):
        table_dict = {
            "project_id": "...", "flag_id": "...", "version": "v1",
            "tables": [
                {"table_name": "R-Observation", "table_type": "Observation",
                 "table_url": "https://...", "table_structure": [...]},
                ...
            ]
        }

    After (output):
        MergeResponse(
            final_df="FILE_NAME,STU_NUMBER,...\\n...",  # CSV string
            errors=[], status="success", metadata={}
        )
    """
    errors: list[SingleError] = []
    try:
        # Step 1: Load and parse each table from table_url -> tables_by_type, errors
        tables_by_type, errors = load_and_parse_tables(table_dict)
        n_loaded = (
            len(tables_by_type.get("plot") or [])
            + len(tables_by_type.get("dosing") or [])
            + len(tables_by_type.get("covariate") or [])
            + len(tables_by_type.get("observation_table") or [])
            + len(tables_by_type.get("paper_labels") or [])
        )
        # Step 2: If no tables could be loaded -> return failed
        if n_loaded == 0:
            if not errors:
                errors.append(
                    {
                        "error_name": "No tables loaded",
                        "error_message": "No tables could be downloaded or parsed.",
                    }
                )
            return MergeResponse(
                final_df="",
                errors=errors,
                status="failed",
                metadata={},
            )
        # Step 3: Run merge pipeline (transform, event_data, merge cov, backfill)
        final_data, errors = run_merge_flow(tables_by_type, errors)
        if final_data is None or final_data.empty:
            return MergeResponse(
                final_df="",
                errors=errors,
                status="failed",
                metadata={},
            )
        # Step 4: Serialize to CSV (use na_rep so Int64/other dtypes are not filled
        # with string "NA" in-memory, which would raise; NaN is written as "NA" in CSV)
        buffer = StringIO()
        final_data.to_csv(buffer, index=False, na_rep="NA")
        csv_string = buffer.getvalue()
        return MergeResponse(
            final_df=csv_string,
            errors=errors,
            status="success",
            metadata={},
        )
    except Exception as e:
        logger.exception(f"Merge flow failed: {e}")
        errors.append(
            {
                "error_name": "Merge failed",
                "error_message": str(e),
            }
        )
        return MergeResponse(
            final_df="",
            errors=errors,
            status="failed",
            metadata={},
        )


def run_merge_and_qc(table_dict: dict) -> MergeResponse:
    """
    V0 merge+QC flow: download tables, group QC, run_whole_merge, standardization,
    then return MergeResponse with CSV and optional cost metadata.
    """
    error_log = []
    try:
        tables_df, updated_table_info = {}, []
        for table_info in table_dict["tables"]:
            table_url = table_info["table_url"]
            logger.info(f"Downloading table from {table_info['table_type']}")
            _, file_content = download_file_from_url(table_url)
            try:
                df = pd.read_csv(StringIO(file_content.decode("utf-8")))
            except Exception as e:
                error_log.append(
                    {
                        "error_name": (
                            f"Error reading table {table_info['table_name']}"
                        ),
                        "error_message": (
                            f"Check the content of the table "
                            f"{table_info['table_name']} "
                            "and ensure that there is proper data "
                            "in the table. \n "
                            f"Error message: {e}"
                        ),
                    }
                )
                continue
            if "index" in df.columns:
                df.drop(columns=["index"], inplace=True)
            else:
                df.reset_index(drop=True, inplace=True)
            if table_info["table_type"] in tables_df and not check_if_null(
                table_info["table_type"]
            ):
                if table_info["table_type"] == TableNames.OBSERVATION.value:
                    df = expand_plot_points(df)
                tables_df[table_info["table_type"]] = pd.concat(
                    [tables_df[table_info["table_type"]], df], ignore_index=True
                )
            elif not check_if_null(table_info["table_type"]):
                if table_info["table_type"] == TableNames.OBSERVATION.value:
                    df = expand_plot_points(df)
                tables_df[table_info["table_type"]] = df
            else:
                if table_info["table_name"] in tables_df:
                    table_name = table_info["table_name"]
                    table_name_unique = table_name + str(uuid4().hex)
                    table_info["table_name"] = table_name_unique
                    tables_df[table_info["table_name"]] = df
                else:
                    tables_df[table_info["table_name"]] = df

            updated_table_info.append(table_info)

        table_dict["tables"] = updated_table_info

        table_structure_dictionary = {}
        for table_info in table_dict["tables"]:
            table_type = table_info["table_type"]
            table_name = table_info["table_name"]
            if table_type:
                table_structure_dictionary[table_type] = table_info["table_structure"]
            else:
                table_structure_dictionary[table_name] = table_info["table_structure"]

        for table_name, df in tables_df.items():
            numerical_columns = [
                column["name"]
                for column in table_structure_dictionary[table_name]
                if column["d_type"] in ["float", "integer", "number"]
            ]
            numerical_columns = list(set(numerical_columns))
            df_cleaned = clean_numerical_columns(df, numerical_columns)
            tables_df[table_name] = df_cleaned

        group_dataframes = {
            table_name: df
            for table_name, df in tables_df.items()
            if table_name
            in [
                TableNames.COVARIATE.value,
                TableNames.OBSERVATION.value,
                TableNames.DOSING.value,
            ]
        }
        group_dataframes, group_error_log = run_group_qc(group_dataframes)

        if len(group_error_log) > 0:
            error_log.extend(group_error_log)
            response = MergeResponse(
                **{
                    "final_df": "",
                    "errors": error_log,
                    "status": "success",
                }
            )
            return response

        for table_name, df in tables_df.items():
            if table_name not in [
                TableNames.COVARIATE.value,
                TableNames.OBSERVATION.value,
                TableNames.DOSING.value,
            ]:
                group_dataframes[table_name] = df

        final_df, merge_error_log = run_whole_merge(
            group_dataframes, table_structure_dictionary
        )
        error_log.extend(merge_error_log)
        if final_df is None:
            return MergeResponse(
                final_df="",
                errors=error_log,
                status="success",
                metadata={},
            )

        for table_name, table_structure in table_structure_dictionary.items():
            updated_table_structure = []
            for column in table_structure:
                if column["name"] == "LLOQ":
                    continue
                updated_table_structure.append(column)
            table_structure_dictionary[table_name] = updated_table_structure

        numerical_columns = [
            column["name"]
            for table_name in table_structure_dictionary
            for column in table_structure_dictionary[table_name]
            if column["d_type"] in ["float", "integer", "number"]
        ]
        numerical_columns = list(set(numerical_columns))
        for column_name in final_df.columns:
            if (
                column_name not in numerical_columns
                and pd.api.types.is_numeric_dtype(final_df[column_name])
            ) or column_name in ["DV_UCI", "DV_LCI"]:
                numerical_columns.append(column_name)

        final_df = clean_numerical_columns(final_df, numerical_columns)
        combined_table_structure = []
        column_names_added = []
        for _table_name, table_structure in table_structure_dictionary.items():
            for column in table_structure:
                if column["name"] not in column_names_added:
                    column_names_added.append(column["name"])
                    combined_table_structure.append(column)

        standardization_result = run_standardization(
            final_df,
            number_columns=numerical_columns,
        )
        final_df, standardization_error_log = standardization_result["result"]
        error_log.extend(
            prepare_error_log(remove_empty_errors(standardization_error_log))
        )
        cost_metadata = standardization_result["metadata"]["ai_metadata"][
            "cost_metadata"
        ]
        final_df = final_df.fillna("NA")
        buffer = StringIO()
        final_df.to_csv(buffer, index=False)
        final_df_string = buffer.getvalue()
        return MergeResponse(
            final_df=final_df_string,
            errors=error_log,
            status="success",
            metadata={
                "response_type": "csv/s3",
                "ai_metadata": {"cost_metadata": cost_metadata},
            },
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        return MergeResponse(
            final_df="",
            errors=[
                {
                    "error_name": "Quality Check Failed",
                    "error_message": (
                        f"Quality Check Failed. Error message: {str(e)}\n\n"
                    ),
                }
            ],
            status="failed",
            metadata={},
        )


def run_standardization_only(df: dict, table_structure: list[dict]) -> dict:
    try:
        df = pd.DataFrame(df)
        numerical_columns = [
            column["name"]
            for column in table_structure
            if column["d_type"] in ["float", "integer", "number"]
        ]
        numerical_columns = list(set(numerical_columns))
        df = clean_numerical_columns(df, numerical_columns)
        standardization_result = run_standardization(df)
        df, _ = standardization_result["result"]
        cost_metadata = standardization_result["metadata"]["ai_metadata"][
            "cost_metadata"
        ]
        df = df.fillna("NA")
        return {
            "final_df": df.to_dict(orient="records"),
            "status": "success",
            "metadata": {
                "ai_metadata": {
                    "cost_metadata": cost_metadata,
                },
            },
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        error_message = "Error standardizing data."
        error_message += f" Error message: {str(e)}"
        raise Exception(error_message)
