import re
from uuid import uuid4

import pandas as pd
from langfuse import observe

from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.standardization.constants import (
    COLUMNS_NOT_TO_CHECK,
    DV_STAT_CONSTANTS,
    DV_UNIT_CONSTANTS,
    DV_VAR_STAT_CONSTANTS,
    FOUND_ERROR_KEY,
    REPLACED_VALUE_KEY,
    ROUTE_MAPPINGS,
    SIMILAR_REGIMEN_MAPPING,
    STATISTICAL_ABBREVIATIONS,
    SUBSCRIPT_MAP,
    SUPERSCRIPT_MAP,
    UNIT_MAPPINGS,
)
from app.v3.endpoints.merging.standardization.llm_normalization import (
    standardize_dv_values_llm,
    standardize_regimen_llm,
    standardize_unit_llm,
    standardize_values_llm,
)
from app.v3.endpoints.merging.utils import (
    check_if_null,
    check_if_number_in_string,
    check_if_string_in_number,
    check_if_unit_starts_with_number,
    clean_numerical_columns,
    preprocess_string,
    replace_nans_with_na,
    return_range_indices,
    standardize_percentage,
    transform_filenames,
)

logger = logger.getChild("standardization")


def double_check_standardized_data(
    data: pd.DataFrame,
    error_log: dict = None,
) -> dict:
    if error_log is None:
        error_log = {REPLACED_VALUE_KEY: {}, FOUND_ERROR_KEY: {}}
    else:
        error_log = error_log.copy()
    # check arm number and number of arms
    number_of_arms = []
    if "ARM_NUMBER" in data.columns:
        arm_negative_indices = []
        arm_greater_than_10_indices = []
        for index, arm_number in enumerate(data["ARM_NUMBER"].values):
            try:
                if float(arm_number) < 1:
                    arm_negative_indices.append(index + 1)
                elif float(arm_number) > 10:
                    arm_greater_than_10_indices.append(index + 1)
            except Exception as e:
                logger.error(
                    f"Failed converting arm number to float."
                    f"arm number: {arm_number}, error: {e}"
                )
                continue

        if len(arm_negative_indices) > 0:
            if "ARM_NUMBER" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["ARM_NUMBER"] = []
            arm_number_error_message = "Minimum `ARM_NUMBER` must be greater than 0."
            arm_number_error_message += f"\n Underflowed rows: {arm_negative_indices}"
            error_log[FOUND_ERROR_KEY]["ARM_NUMBER"].append(arm_number_error_message)

        if len(arm_greater_than_10_indices) > 0:
            if "ARM_NUMBER" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["ARM_NUMBER"] = []
            arm_number_error_message = "Maximum `ARM_NUMBER` should be less than 10."
            arm_number_error_message += (
                f"\n Overflowed rows: {arm_greater_than_10_indices}"
            )
            error_log[FOUND_ERROR_KEY]["ARM_NUMBER"].append(arm_number_error_message)

        for _index, arm_number in enumerate(data["ARM_NUMBER"].values):
            try:
                arm_number = float(arm_number)
                number_of_arms.append(arm_number)
            except Exception as e:
                logger.error(
                    f"Failed converting arm number to float."
                    f"arm number: {arm_number}, error: {e}"
                )
                continue

    if "NUMBER_OF_ARMS" in data.columns and number_of_arms:
        number_of_arms = len(set(number_of_arms))
        number_of_arms_in_data = float(data["NUMBER_OF_ARMS"].unique()[0])
        if number_of_arms != number_of_arms_in_data:
            if "NUMBER_OF_ARMS" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["NUMBER_OF_ARMS"] = []
            number_of_arms_error_message = (
                "`NUMBER_OF_ARMS`!= number of unique `ARM_NUMBER` values."
            )
            number_of_arms_error_message += (
                f"\n Number of unique `ARM_NUMBER` values: {number_of_arms}"
            )
            number_of_arms_error_message += (
                f"\n `NUMBER_OF_ARMS` value: {number_of_arms_in_data}"
            )
            error_log[FOUND_ERROR_KEY]["NUMBER_OF_ARMS"].append(
                number_of_arms_error_message
            )

    # check dvid
    if "DVID" in data.columns and "DV" in data.columns:
        unqiue_dvids = []
        negative_dvids = []
        greater_than_10_dvids = []
        for index, dvid in enumerate(data["DVID"].values):
            try:
                dvid = float(dvid)
                unqiue_dvids.append(dvid)
                if dvid < 1:
                    negative_dvids.append(index + 1)
                elif dvid > 10:
                    greater_than_10_dvids.append(index + 1)
            except Exception as e:
                logger.error(
                    f"Failed converting dvid to float. dvid: {dvid}, error: {e}"
                )
                continue

        if len(negative_dvids) > 0:
            if "DVID" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["DVID"] = []
            dvid_error_message = "`DVID` should be greater than 0."
            dvid_error_message += f"\n Check these rows: {negative_dvids}"
            error_log[FOUND_ERROR_KEY]["DVID"].append(dvid_error_message)
        if len(greater_than_10_dvids) > 0:
            if "DVID" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["DVID"] = []
            dvid_error_message = "`DVID` should be less than 10."
            dvid_error_message += f"\n Check these rows: {greater_than_10_dvids}"
            error_log[FOUND_ERROR_KEY]["DVID"].append(dvid_error_message)

        dvid_4_error_indices = []
        dvid_3_error_indices = []
        dv_for_dvid_4 = data[
            ((data["DVID"] == "4.0") | (data["DVID"] == "4") | (data["DVID"] == 4))
        ]["DV"]
        dv_for_dvid_3 = data[
            ((data["DVID"] == "3.0") | (data["DVID"] == "3") | (data["DVID"] == 3))
        ]["DV"]

        for i, dv_value in enumerate(dv_for_dvid_4):
            if check_if_null(dv_value):
                continue
            if dv_value < 0 or dv_value > 300:
                dvid_4_error_indices.append(i + 1)
        for i, dv_value in enumerate(dv_for_dvid_3):
            if check_if_null(dv_value):
                continue
            if dv_value < 0:
                dvid_3_error_indices.append(i + 1)

        if len(dvid_4_error_indices) > 0:
            if "DVID" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["DVID"] = []
            dvid_error_message = (
                "`DV` should be greater than 0 and less than 300 when `DVID` is 4."
            )
            dvid_error_message += f"\n Check these rows: {dvid_4_error_indices}"
            error_log[FOUND_ERROR_KEY]["DVID"].append(dvid_error_message)
        if len(dvid_3_error_indices) > 0:
            if "DVID" not in error_log[FOUND_ERROR_KEY]:
                error_log[FOUND_ERROR_KEY]["DVID"] = []
            dvid_error_message = "`DV` should be greater than 0 when `DVID` is 3."
            dvid_error_message += f"\n Check these rows: {dvid_3_error_indices}"
            error_log[FOUND_ERROR_KEY]["DVID"].append(dvid_error_message)
    return error_log


def standardize_unit(value: str) -> str:
    # Remove superscript/subscript characters by
    # replacing them with regular characters
    for sup, reg in SUPERSCRIPT_MAP.items():
        value = value.replace(sup, reg)
    for sub, reg in SUBSCRIPT_MAP.items():
        value = value.replace(sub, reg)

    # Replace full unit names with their abbreviations
    if value in UNIT_MAPPINGS:
        value = re.sub(r"\b" + re.escape(value) + r"\b", UNIT_MAPPINGS[value], value)
    elif value.strip().lower() in UNIT_MAPPINGS:
        value = re.sub(
            r"\b" + re.escape(value) + r"\b",
            UNIT_MAPPINGS[value.strip().lower()],
            value,
        )

    # replace plural units with singular ones except for time units
    for unit in UNIT_MAPPINGS:
        if unit + "s" == value or unit + "s" == value.strip().lower():
            value = re.sub(r"\b" + re.escape(value) + r"\b", UNIT_MAPPINGS[unit], value)

    # Replace variations of "per" and "with respect to" with forward slash
    value = re.sub(
        r"\b(per|with\s+respect\s+to|with\s+respect|w\.r\.t\.)\b",
        "/",
        value,
        flags=re.IGNORECASE,
    )

    # Remove any spaces in the middle of the string
    value = re.sub(r"\s+", "", value)
    # Normalize percentage
    value = standardize_percentage(value)
    return value


def standardize_statistical_values(value: str) -> str:
    replaced_value = value
    if value in STATISTICAL_ABBREVIATIONS:
        replaced_value = re.sub(
            r"\b" + re.escape(value) + r"\b", STATISTICAL_ABBREVIATIONS[value], value
        )
    elif value.strip().lower() in STATISTICAL_ABBREVIATIONS:
        replaced_value = re.sub(
            r"\b" + re.escape(value) + r"\b",
            STATISTICAL_ABBREVIATIONS[value.strip().lower()],
            value,
        )
    elif value.endswith("s"):
        singular_value = re.sub(r"s$", "", value)
        if singular_value in STATISTICAL_ABBREVIATIONS:
            replaced_value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                STATISTICAL_ABBREVIATIONS[singular_value],
                value,
            )
        elif singular_value.strip().lower() in STATISTICAL_ABBREVIATIONS:
            replaced_value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                STATISTICAL_ABBREVIATIONS[singular_value.strip().lower()],
                value,
            )
    abbreviations = list(STATISTICAL_ABBREVIATIONS.values())
    if replaced_value.upper() in abbreviations:
        replaced_value = replaced_value.upper()
    return replaced_value


def standardize_route(value: str) -> str:
    if value in ROUTE_MAPPINGS:
        return re.sub(r"\b" + re.escape(value) + r"\b", ROUTE_MAPPINGS[value], value)
    elif value.strip().lower() in ROUTE_MAPPINGS:
        return re.sub(
            r"\b" + re.escape(value) + r"\b",
            ROUTE_MAPPINGS[value.strip().lower()],
            value,
        )
    elif value.endswith("s"):
        singular_value = re.sub(r"s$", "", value)
        if singular_value in ROUTE_MAPPINGS:
            value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                ROUTE_MAPPINGS[singular_value],
                value,
            )
        elif singular_value.strip().lower() in ROUTE_MAPPINGS:
            value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                ROUTE_MAPPINGS[singular_value.strip().lower()],
                value,
            )
        return value
    else:
        return value


def standardize_regimen(value: str) -> str:
    if value in SIMILAR_REGIMEN_MAPPING:
        value = re.sub(
            r"\b" + re.escape(value) + r"\b", SIMILAR_REGIMEN_MAPPING[value], value
        )
    elif value.strip().lower() in SIMILAR_REGIMEN_MAPPING:
        value = re.sub(
            r"\b" + re.escape(value) + r"\b",
            SIMILAR_REGIMEN_MAPPING[value],
            value,
        )
    elif value.endswith("s"):
        singular_value = re.sub(r"s$", "", value)
        if singular_value in SIMILAR_REGIMEN_MAPPING:
            value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                SIMILAR_REGIMEN_MAPPING[singular_value],
                value,
            )
        elif singular_value.strip().lower() in SIMILAR_REGIMEN_MAPPING:
            value = re.sub(
                r"\b" + re.escape(value) + r"\b",
                SIMILAR_REGIMEN_MAPPING[singular_value.strip().lower()],
                value,
            )
        return value
    value = re.sub(r"(?<=[qtbo])[1](?=[hdmw])", "", value, flags=re.IGNORECASE)
    return value


def replace_range(value: str) -> str:
    range_patterns = [
        r"^\d*\.?\d+\s*-\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*:\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s+to\s+\d*\.?\d+$",
        r"^from\s+\d*\.?\d+\s+to\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*till\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*until\s+\d*\.?\d+$",
        r"^\d*\.?\d+\s*–\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*—\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2010\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2011\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2012\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2013\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2014\s*\d*\.?\d+$",
        r"^\d*\.?\d+\s*\u2015\s*\d*\.?\d+$",
    ]

    for pattern in range_patterns:
        if re.match(pattern, value):
            try:
                all_values = list(map(int, re.findall(r"\d+", value)))
                min_val = min(all_values)
                max_val = max(all_values)
                return str((max_val - min_val) / 4)
            except Exception:
                return value
    return value


def replace_same_strings_with_first(values: list[str]) -> list[str]:
    unique_values_dict = {}
    for value in values:
        if preprocess_string(value) not in unique_values_dict:
            unique_values_dict[preprocess_string(value)] = value

    normalized_values = []
    for value in values:
        preprocessed_value = preprocess_string(value)
        if preprocessed_value in unique_values_dict:
            normalized_values.append(unique_values_dict[preprocessed_value])
        else:
            normalized_values.append(value)
    if len(normalized_values) != len(values):  # nosec B101
        raise ValueError(f"Length mismatch: {len(normalized_values)} != {len(values)}")
    return normalized_values


def hardcoded_unit_mapping(value: str) -> str:
    value = value.strip().lower()
    if value == "kg/m3":
        value = "kg/m2"
    if (
        re.match(r"ml/min/1\.73m(?!2)", value)
        or re.match(r"ml/min/1\.73(?!2)", value)
        or value == "ml/min/1.73"
    ):
        value = "ml/min/1.73m2"
    return value


def standardize_unit_columns(
    df: pd.DataFrame, columns_to_check: list[str], langfuse_session_id: str = None
) -> pd.DataFrame:
    unique_units = []
    for column in columns_to_check:
        if column in df.columns and column.strip().lower().endswith("_unit"):
            unique_units.extend(df[column].unique())
    unique_units = list(set(unique_units))
    unique_units = [value for value in unique_units if value != "NA"]
    normalized_unique_units = standardize_unit_llm(
        unique_units, langfuse_session_id=langfuse_session_id
    )
    normalized_units_dict = dict(zip(unique_units, normalized_unique_units))
    for column in columns_to_check:
        if column in df.columns and column.strip().lower().endswith("_unit"):
            df[column] = df[column].apply(
                lambda x: normalized_units_dict[x] if x in normalized_units_dict else x
            )
            df[column] = df[column].apply(lambda x: standardize_unit(x))
            if column.strip().lower() in ["bmi_bl_unit", "egfr_bl_unit"]:
                df[column] = df[column].apply(lambda x: hardcoded_unit_mapping(x))
    return df


def standardize_regimen_columns(
    df: pd.DataFrame,
    columns_to_check: list[str],
    langfuse_session_id: str = None,
) -> pd.DataFrame:
    unique_regimens = []
    for column in columns_to_check:
        if column in df.columns and "regimen" in column.strip().lower():
            unique_regimens.extend(df[column].unique())
    unique_regimens = list(set(unique_regimens))
    unique_regimens = [value for value in unique_regimens if value != "NA"]
    normalized_unique_regimens = standardize_regimen_llm(
        unique_regimens, langfuse_session_id=langfuse_session_id
    )
    normalized_regimens_dict = dict(zip(unique_regimens, normalized_unique_regimens))

    # renormalize the regimens after llm processsing to
    # fix any errors by llm
    for column in columns_to_check:
        if column in df.columns and "regimen" in column.strip().lower():
            df[column] = df[column].apply(
                lambda x: (
                    normalized_regimens_dict[x] if x in normalized_regimens_dict else x
                )
            )
            df[column] = df[column].apply(lambda x: standardize_regimen(x))
    return df


def standardize_dv_columns(
    df: pd.DataFrame,
    langfuse_session_id: str = None,
) -> pd.DataFrame:
    if "DV_UNIT" in df.columns:
        unique_units = df["DV_UNIT"].unique()
        unique_units = [value for value in unique_units if value != "NA"]
        normalized_units = standardize_dv_values_llm(
            unique_units,
            constant_values=DV_UNIT_CONSTANTS,
            langfuse_session_id=langfuse_session_id,
        )
        normalized_units_dict = dict(zip(unique_units, normalized_units))
        df["DV_UNIT"] = df["DV_UNIT"].apply(
            lambda x: normalized_units_dict[x] if x in normalized_units_dict else x
        )

    if "DV_STAT" in df.columns:
        unique_stats = df["DV_STAT"].unique()
        unique_stats = [value for value in unique_stats if value != "NA"]
        normalized_stats = standardize_dv_values_llm(
            unique_stats,
            constant_values=DV_STAT_CONSTANTS,
            langfuse_session_id=langfuse_session_id,
        )
        normalized_stats_dict = dict(zip(unique_stats, normalized_stats))
        df["DV_STAT"] = df["DV_STAT"].apply(
            lambda x: normalized_stats_dict[x] if x in normalized_stats_dict else x
        )

    if "DV_VAR_STAT" in df.columns:
        unique_var_stats = df["DV_VAR_STAT"].unique()
        unique_var_stats = [value for value in unique_var_stats if value != "NA"]
        normalized_var_stats = standardize_dv_values_llm(
            unique_var_stats,
            constant_values=DV_VAR_STAT_CONSTANTS,
            langfuse_session_id=langfuse_session_id,
        )
        normalized_var_stats_dict = dict(zip(unique_var_stats, normalized_var_stats))
        df["DV_VAR_STAT"] = df["DV_VAR_STAT"].apply(
            lambda x: (
                normalized_var_stats_dict[x] if x in normalized_var_stats_dict else x
            )
        )
    return df


def standardize_string_columns(
    df: pd.DataFrame,
    number_columns: list[str],
    statistical_columns: list[str],
    columns_to_check: list[str],
    split_size: int = 100,
    number_of_passes: int = 1,
    langfuse_session_id: str = None,
) -> pd.DataFrame:
    excluded_columns = number_columns.copy()
    excluded_columns.extend(
        [
            column
            for column in columns_to_check
            if (
                column in df.columns
                and "regimen" in column.strip().lower()
                or "_unit" in column.strip().lower()
            )
        ]
    )
    columns_to_normalize = [
        column
        for column in columns_to_check
        if column not in excluded_columns and column in df.columns
    ]

    for _pass_number in range(number_of_passes):
        all_column_values = []
        for column in columns_to_normalize:
            all_column_values.extend(df[column].unique())

        all_column_values = list(set(all_column_values))
        all_column_values = [value for value in all_column_values if value != "NA"]
        all_column_values_normalized = []
        for i in range(0, len(all_column_values), split_size):
            all_column_values_split = all_column_values[i : i + split_size]
            all_column_values_normalized_split = standardize_values_llm(
                all_column_values_split,
                langfuse_session_id=langfuse_session_id,
            )
            all_column_values_normalized.extend(all_column_values_normalized_split)

        normalized_values_dict = dict(
            zip(all_column_values, all_column_values_normalized)
        )

        for column in columns_to_normalize:
            df[column] = df[column].apply(
                lambda x: (
                    normalized_values_dict[x]  # noqa: B023
                    if x in normalized_values_dict  # noqa: B023
                    else x
                )
            )

    for column in columns_to_normalize:
        if column in df.columns and "route" in column.strip().lower():
            df[column] = df[column].apply(lambda x: standardize_route(x))

        # if column is a statistical column, normalize the statistical values
        if column in statistical_columns:
            df[column] = df[column].apply(lambda x: standardize_statistical_values(x))

    # if column is a treatment column, capitalize
    # first letter of the treatment values
    for column in [
        "ARM_TRT",
        "STD_TRT",
        "ARM_TRT_CLASS",
        "STD_TRT_CLASS",
    ]:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda x: x.capitalize() if x and x[0].islower() else x
            )

    return df


def get_changed_values_dict(
    df_before_standardization: pd.DataFrame,
    df_after_standardization: pd.DataFrame,
) -> dict:
    """
    During standardization, if a value is replaced by a fixed value,
    it is logged in the changed_values_dict.
    """
    changed_values_dict = {}
    for column in df_after_standardization.columns:
        if (
            column not in df_before_standardization.columns
            or column.strip().lower().endswith("_error")
        ):
            continue
        for i, (before, after) in enumerate(
            zip(df_before_standardization[column], df_after_standardization[column])
        ):
            if check_if_null(before) or check_if_null(after):
                continue
            try:
                if before != after:
                    if "FILE_NAME" in df_after_standardization.columns:
                        filename = df_after_standardization.at[i, "FILE_NAME"]
                        if filename not in changed_values_dict:
                            changed_values_dict[filename] = {}
                        if column not in changed_values_dict[filename]:
                            changed_values_dict[filename][column] = {}

                        changed_values_dict[filename][column][before] = after
                    else:
                        if column not in changed_values_dict:
                            changed_values_dict[column] = {}
                        changed_values_dict[column][before] = after
            except Exception as e:
                logger.error(
                    f"Error comparing values for column {column} at index {i}: {str(e)}"
                )
                continue

    return changed_values_dict


@observe()
def standardize_using_regex_llm(
    df: pd.DataFrame,
    number_columns: list[str] = None,
    statistical_columns: list[str] = None,
    string_columns: list[str] = None,
    error_log: dict = None,
    langfuse_session_id: str = None,
) -> (pd.DataFrame, dict):
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    setup_langfuse_handler(langfuse_session_id)

    if error_log is None:
        error_log = {REPLACED_VALUE_KEY: {}, FOUND_ERROR_KEY: {}}
    else:
        error_log = error_log.copy()
    df_before_standardization = df.copy(deep=True)
    columns_to_check = [
        column for column in df.columns if column not in COLUMNS_NOT_TO_CHECK
    ]
    # number columns are columns that contain numbers
    # and not strings like tirzepatide
    if not number_columns or not statistical_columns:
        if not number_columns:
            add_number_columns = True
            number_columns = []
        else:
            add_number_columns = False
        if not statistical_columns:
            add_statistical_columns = True
            statistical_columns = []
        else:
            add_statistical_columns = False

        for column in columns_to_check:
            if add_number_columns and (
                column in df.columns
                and (
                    column.strip().lower().endswith("_bl_var")
                    or column.strip().lower().endswith("_bl")
                    or column.strip().lower().endswith("_perc")
                )
                and not (
                    column.strip().lower().endswith("_var_stat")
                    or column.strip().lower().endswith("_stat")
                    or column.strip().lower().endswith("_unit")
                )
            ):
                number_columns.append(column)
            if add_statistical_columns and (
                column.strip().lower().endswith("_stat")
                or column.strip().lower().endswith("_var_stat")
            ):
                statistical_columns.append(column)

    if not string_columns:
        string_columns = [
            column
            for column in columns_to_check
            if column not in number_columns
            and "regimen" not in column.strip().lower()
            and "_unit" not in column.strip().lower()
        ]

    for column in columns_to_check:
        if column not in df.columns:
            continue
        # if column is not a number column,
        # preprocess the string like removing spaces,
        # lowercasing, normalizing, etc.
        if column not in number_columns:
            df[column] = df[column].apply(lambda x: preprocess_string(x))

        # replace all nans, nulls with NA
        df[column] = df[column].apply(lambda x: replace_nans_with_na(x))

        if column not in number_columns and not column.strip().lower().endswith(
            "_unit"
        ):
            df[column + "_string_error"] = df[column].apply(
                lambda x: check_if_number_in_string(x),
            )

        # check if the value is a number
        if column in number_columns:
            # change logic to var_stat = range when var is range
            # df[column] = df[column].apply(replace_range)
            # check if the value is a number
            range_indices = return_range_indices(df[column].values)
            column_field_name = "_".join(column.split("_")[:-1])
            if column_field_name + "_VAR_STAT" in df.columns:
                for index in range_indices:
                    stat_value = df.at[index, column_field_name + "_VAR_STAT"]
                    if stat_value not in ["range", "IQR"]:
                        df.at[index, column_field_name + "_VAR_STAT"] = "range"

            df[column + "_number_error"] = df[column].apply(
                lambda x: check_if_string_in_number(x)
            )

        # if column is a unit column, normalize the unit
        if column.strip().lower().endswith("_unit") and column not in number_columns:
            df[column] = df[column].apply(lambda x: standardize_unit(x))
            df[column + "_unit_error"] = df[column].apply(
                lambda x: check_if_unit_starts_with_number(x)
            )

        if column.strip().lower() in ["bmi_bl_unit", "egfr_bl_unit"]:
            df[column] = df[column].apply(lambda x: hardcoded_unit_mapping(x))

        # if column is a route column, normalize the route
        if "route" in column.strip().lower():
            df[column] = df[column].apply(lambda x: standardize_route(x))

        # if column is a regimen column, normalize the regimen
        if "regimen" in column.strip().lower():
            df[column] = df[column].apply(lambda x: standardize_regimen(x))

        # if column is a statistical column, normalize the statistical values
        if column in statistical_columns:
            df[column] = df[column].apply(lambda x: standardize_statistical_values(x))

        # if column is a treatment column, capitalize
        # first letter of the treatment values
        if (
            column
            in [
                "ARM_TRT",
                "STD_TRT",
                "ARM_TRT_CLASS",
                "STD_TRT_CLASS",
            ]
            and column in df.columns
        ):
            df[column] = df[column].apply(
                lambda x: x.capitalize() if x and x[0].islower() else x
            )

    df_normalized_llm = df.copy()
    # replace the same strings with the first string in the column
    # to fix any errors by llm
    for column in columns_to_check:
        if column not in df.columns:
            continue
        df_normalized_llm[column] = replace_same_strings_with_first(
            df_normalized_llm[column]
        )

    df_normalized_llm = standardize_string_columns(
        df_normalized_llm,
        number_of_passes=2,
        number_columns=number_columns,
        statistical_columns=statistical_columns,
        columns_to_check=columns_to_check,
        langfuse_session_id=langfuse_session_id,
    )
    df_normalized_llm = standardize_unit_columns(
        df_normalized_llm,
        columns_to_check=columns_to_check,
        langfuse_session_id=langfuse_session_id,
    )
    df_normalized_llm = standardize_regimen_columns(
        df_normalized_llm,
        columns_to_check=columns_to_check,
        langfuse_session_id=langfuse_session_id,
    )
    df_normalized_llm = standardize_dv_columns(
        df_normalized_llm,
        langfuse_session_id=langfuse_session_id,
    )

    # renormalize the percentage values after llm processing
    # to fix any errors by llm
    for column in columns_to_check:
        if column in df_normalized_llm.columns and column.strip().lower().endswith(
            "_unit"
        ):
            df_normalized_llm[column] = df_normalized_llm[column].apply(
                lambda x: standardize_percentage(x)
            )

    # replace the same strings with the first string in the column
    # to fix any errors by llm
    for column in df_normalized_llm.columns:
        df_normalized_llm[column] = df_normalized_llm[column].apply(
            lambda x: replace_nans_with_na(x)
        )
    df_normalized_llm = df_normalized_llm.fillna("NA")
    if "FILE_NAME" in df_normalized_llm.columns:
        df_normalized_llm["FILE_NAME"] = transform_filenames(
            list(df_normalized_llm["FILE_NAME"].values)
        )

    # Create ARM REGIMEN COLUMN by combining ARM_DOSE, DOSE_UNIT, and REGIMEN
    # TODO: not needed right now, removing
    # if (
    #     "ARM_DOSE" in df_normalized_llm.columns
    #     and "ARM_DOSE_UNIT" in df_normalized_llm.columns
    #     and "REGIMEN" in df_normalized_llm.columns
    # ):
    #     df_normalized_llm["ARM_REGIMEN"] = df_normalized_llm.apply(
    #         lambda x: f"{x['ARM_DOSE']} {x['ARM_DOSE_UNIT']} {x['REGIMEN']}", axis=1
    #     )
    # TODO: not needed right now, removing

    for column in columns_to_check:
        if column not in df_normalized_llm.columns:
            continue
        df_normalized_llm[column] = replace_same_strings_with_first(
            df_normalized_llm[column]
        )

    df_normalized_llm = clean_numerical_columns(df_normalized_llm, number_columns)
    changed_values = get_changed_values_dict(
        df_before_standardization,
        df_normalized_llm,
    )
    error_log[REPLACED_VALUE_KEY] = changed_values

    if "GROUP NAME" in df_normalized_llm.columns:
        df_normalized_llm.rename(
            columns={
                "GROUP NAME": "GROUP_NAME",
            },
            inplace=True,
        )

    found_error_log = {}
    error_columns_to_remove = []
    for column in df_normalized_llm.columns:
        if column.strip().lower().endswith("_error"):
            error_columns_to_remove.append(column)
            if column.strip().lower().endswith("_unit_error"):
                indices = [
                    i + 1 for i in df_normalized_llm[df_normalized_llm[column]].index
                ]
                if len(indices) > 0:
                    error_message = (
                        "Found invalid units. Unit should not start with a number."
                    )
                    error_message += f"Check these rows: {indices}"
                    actual_col_name = column.replace("_unit_error", "")
                    if actual_col_name not in found_error_log:
                        found_error_log[actual_col_name] = []
                    found_error_log[actual_col_name].append(error_message)
            elif column.strip().lower().endswith("_string_error"):
                indices = [
                    i + 1 for i in df_normalized_llm[df_normalized_llm[column]].index
                ]
                if len(indices) > 0:
                    error_message = (
                        "String labels are not supposed to be fully numbers.\n"
                        "Are you sure you do not want to set the label "
                        "data type as number?\n"
                    )
                    error_message += f"Check these rows: {indices}"
                    actual_col_name = column.replace("_string_error", "")
                    if actual_col_name not in found_error_log:
                        found_error_log[actual_col_name] = []
                    found_error_log[actual_col_name].append(error_message)
            elif column.strip().lower().endswith("_number_error"):
                indices = [
                    i + 1 for i in df_normalized_llm[df_normalized_llm[column]].index
                ]
                if len(indices) > 0:
                    error_message = (
                        "Found invalid numbers. Number should not be a string."
                    )
                    error_message += f"Check these rows: {indices}"
                    actual_col_name = column.replace("_number_error", "")
                    if actual_col_name not in found_error_log:
                        found_error_log[actual_col_name] = []
                    found_error_log[actual_col_name].append(error_message)

    error_log[FOUND_ERROR_KEY] = found_error_log
    df_normalized_llm = df_normalized_llm.drop(columns=error_columns_to_remove)
    return df_normalized_llm, error_log


def summarize_unique_values(data, columns):
    """
    Summarize unique values in the dataframe in specified columns
    """
    summary_table = []
    for column in columns:
        if column in data.columns:
            unique_vals = data[column].unique()
            n_unique = len(unique_vals)
            summary_table.append(
                {
                    "Column": column,
                    "Number_of_Unique_Values": n_unique,
                    "Unique_Values": ", ".join(map(str, unique_vals)),
                }
            )
        else:
            summary_table.append(
                {
                    "Column": column,
                    "Number_of_Unique_Values": None,
                    "Unique_Values": "Column not found in dataframe",
                }
            )
    return pd.DataFrame(summary_table)


def seperate_data_and_save(df, xlsx_files, file_rows):
    start = 0
    for i, num_rows in enumerate(file_rows):
        end = start + num_rows
        df_subset = df.iloc[start:end]
        dest_path = xlsx_files[i].replace("final_merge", "final_standardized")
        df_subset.to_excel(dest_path, index=False, na_rep="NA")
        start = end


def prepare_final_error_log(error_log):
    final_error_log = {}
    for column_name, errors in error_log[FOUND_ERROR_KEY].items():
        if len(errors) > 0:
            joined_error_messages = "\n".join(errors)
            final_error_log["check_" + column_name] = joined_error_messages
    return final_error_log
