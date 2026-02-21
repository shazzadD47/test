import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from app.v3.endpoints.merging.constants import (
    COVARIATE_COLUMNS_TO_DROP,
    DOSING_COLUMNS_TO_DROP,
    FINAL_COLUMNS_TO_DROP,
    OBSERVATION_COLUMNS_TO_DROP,
    TableNames,
)
from app.v3.endpoints.merging.group_qc import find_group_name_column
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.utils import (
    break_comma_separated_numbers,
    check_if_comma_or_space_separated_numbers,
    check_if_null,
    check_if_range,
    clean_numerical_columns,
    is_var_column,
    validate_var_column_values,
)

# ## **Utility Functions**

"""-------------- UTILITY FUNCTIONS--------------------------------"""


# ### Formatting Functions
def format_figure_reference(original_reference):
    """
    Format a figure reference by removing specific prefixes and spaces.

    Args:
    original_reference (str): The original figure reference string.

    Returns:
    str: A formatted figure reference string.
    """
    # Define a pattern to match words starting with
    # 'F' or 'f' followed by optional spaces
    if check_if_null(original_reference):
        raise ValueError("Original reference is null")

    figure_prefix_pattern = r"\b[Ff]\w*\s*"

    # Remove figure prefixes (e.g., 'Figure', 'Fig') and
    # strip leading/trailing spaces
    reference_without_prefix = re.sub(
        figure_prefix_pattern, "", original_reference
    ).strip()

    # Remove all remaining spaces from the reference
    compact_reference = reference_without_prefix.replace(" ", "")

    # Construct the final formatted reference
    formatted_reference = f"Figure {compact_reference}"

    return formatted_reference


def get_conversion_factor(from_unit, to_unit):
    conversion_factors = {
        "minutes": {
            "minutes": 1,
            "hours": 1 / 60,
            "days": 1 / 1440,
            "weeks": 1 / 10080,
            "months": 1 / 43830,
        },
        "hours": {
            "minutes": 60,
            "hours": 1,
            "days": 1 / 24,
            "weeks": 1 / 168,
            "months": 1 / 730.5,
        },
        "days": {
            "minutes": 1440,
            "hours": 24,
            "days": 1,
            "weeks": 1 / 7,
            "months": 1 / 30.4375,
        },
        "weeks": {
            "minutes": 10080,
            "hours": 168,
            "days": 7,
            "weeks": 1,
            "months": 1 / 4.348125,
        },
        "months": {
            "minutes": 43830,
            "hours": 730.5,
            "days": 30.4375,
            "weeks": 4.348125,
            "months": 1,
        },
    }
    return conversion_factors[from_unit][to_unit]


# Function to define the rank of each time unit
def get_unit_rank(unit):
    unit_ranks = {"minutes": 1, "hours": 2, "days": 3, "weeks": 4, "months": 5}
    return unit_ranks[unit]


# Convert between units
def convert_time_row(org_time_col, org_time_unit, target_time_unit):

    # Use the conversion factor to convert to the lowest unit
    conversion_factor = get_conversion_factor(org_time_unit, target_time_unit)
    return org_time_col * conversion_factor


# endregion


"""-------------- PROCESSING FUNCTIONS --------------------------------"""

# region PROCESSING FUNCTIONS


# endregion


"""-------------- IMPORT DATA--------------------------------"""


# region IMPORT DATA
# ### Create Table Dictionary
# Finds all available files and matches it to table list
def create_table_file_dictionary(table_list, folder_path):
    """
    Create a dictionary mapping table names to their corresponding file paths.

    Args:
    table_list (list): List of table names to search for.
    folder_path (str): Relative path to the folder containing data files.

    Returns:
    dict: A dictionary with table names as keys and file paths as values.
    """
    table_file_dict = {}
    # absolute_folder_path = os.path.join(os.getcwd(), folder_path)

    current_folder = Path(__file__).parent.absolute()
    absolute_folder_path = os.path.join(current_folder, folder_path)

    for root, _, files in os.walk(absolute_folder_path):
        for file in files:
            for table in table_list:
                if match_table_to_file(table, file):
                    table_file_dict[table] = os.path.join(root, file)
                    break  # Match found, move to next file

    return table_file_dict


def match_table_to_file(table, file):
    """
    Check if a table name matches a file name.

    Args:
    table (str): Table name to match.
    file (str): File name to check against.

    Returns:
    bool: True if there's a match, False otherwise.
    """
    pattern = re.sub(r"[ -]", "[ -_]", table)
    return bool(re.search(pattern, file, re.IGNORECASE))


# *  Load in files and observation data from file path
def load_observation_data(
    dataframe_dictionary,
    table_structure_dictionary,
):
    # region LOAD IN FILES AND OBSERVATION DATA
    # FIX FOR MISALIGNED COLUMNS
    error_log = []
    try:
        cov_table_name = TableNames.COVARIATE.value
        if cov_table_name in dataframe_dictionary:
            cov_df = dataframe_dictionary[cov_table_name]
            if (
                "COV" in cov_df.columns
                and "COV_VAR" in cov_df.columns
                and "COV_MAX" in cov_df.columns
                and "COV_MIN" in cov_df.columns
                and cov_df["COV_VAR"].isnull().all()
            ):
                dataframe_dictionary[cov_table_name]["COV_VAR_STAT"] = cov_df["COV_MAX"]
                dataframe_dictionary[cov_table_name]["COV_VAR"] = cov_df["COV_MIN"]
                dataframe_dictionary[cov_table_name] = dataframe_dictionary[
                    cov_table_name
                ].drop("COV_MAX", axis=1)
                dataframe_dictionary[cov_table_name] = dataframe_dictionary[
                    cov_table_name
                ].drop("COV_MIN", axis=1)

        # * Convert columns to unified datatypesR
        # Columns to be dropped
        obs_columns_to_drop = OBSERVATION_COLUMNS_TO_DROP

        # Track columns to convert to string in table structure
        columns_to_convert_to_string = {}

        # Convert all the columns for every dataframe
        for key, df in dataframe_dictionary.items():
            logger.debug(key)
            table_structure = table_structure_dictionary.get(key, [])
            string_columns = [
                column["name"]
                for column in table_structure
                if column["d_type"] == "string"
            ]
            float_columns = [
                column["name"]
                for column in table_structure
                if column["d_type"]
                in [
                    "float",
                    "integer",
                ]
            ]
            if key == TableNames.OBSERVATION.value:
                columns_to_drop = obs_columns_to_drop
            else:
                columns_to_drop = []
            for column in df.columns:
                logger.info(f"Converting {column} to {df[column].dtype}")
                if column in columns_to_drop:
                    df.drop(column, axis=1, inplace=True)
                elif column in string_columns:
                    df[column] = df[column].fillna("NA")
                    df[column] = df[column].astype("string")
                elif column in float_columns:
                    try:
                        logger.info(f"Converting {column} to float")

                        # Special handling for _VAR columns in covariate table
                        # Don't break into rows, validate individual items
                        is_covariate_var_column = (
                            key == TableNames.COVARIATE.value and is_var_column(column)
                        )

                        # Check if any values have MULTIPLE comma/space separated
                        # numbers (not single numbers)
                        has_multiple_values = (
                            df[column]
                            .apply(
                                lambda x: (
                                    check_if_comma_or_space_separated_numbers(str(x))
                                    if not check_if_null(x)
                                    else False
                                )
                            )
                            .any()
                        )

                        has_range_values = (
                            df[column]
                            .apply(
                                lambda x: (
                                    check_if_range(str(x))
                                    if not check_if_null(x)
                                    else False
                                )
                            )
                            .any()
                        )

                        # For _VAR columns in covariate table with multiple values
                        # or ranges, validate but keep as string
                        if is_covariate_var_column and (
                            has_multiple_values or has_range_values
                        ):
                            logger.info(
                                f"Column {column} is a _VAR column with "
                                f"comma-separated or range values. "
                                f"Validating and keeping as string."
                            )
                            # Validate each individual item
                            for _, value in df[column].items():
                                if not check_if_null(value):
                                    is_valid, _ = validate_var_column_values(value)
                                    if not is_valid:
                                        error_log.append(
                                            {
                                                "error_name": (
                                                    f"Invalid value in column {column}"
                                                ),
                                                "error_message": (
                                                    f"Value '{value}' in column "
                                                    f"{column} contains invalid "
                                                    f"numbers. Each item in a "
                                                    f"comma-separated or range value "
                                                    f"must be a valid number."
                                                ),
                                            }
                                        )
                            # Convert to string and mark for table structure update
                            df[column] = df[column].apply(
                                lambda x: str(x) if not check_if_null(x) else "NA"
                            )
                            df[column] = df[column].astype("string")
                            # Track this column to update table structure
                            if key not in columns_to_convert_to_string:
                                columns_to_convert_to_string[key] = []
                            columns_to_convert_to_string[key].append(column)
                        elif has_multiple_values:
                            # Original behavior: break into multiple rows
                            # (for non-_VAR columns with multiple values)
                            new_rows = []

                            for idx, value in df[column].items():
                                if not check_if_null(
                                    value
                                ) and check_if_comma_or_space_separated_numbers(
                                    str(value)
                                ):
                                    # Get the list of numbers
                                    numbers = break_comma_separated_numbers(str(value))
                                    # Keep the first number in the original row
                                    prev_row = df.loc[idx].copy()
                                    prev_row[column] = numbers[0]
                                    new_rows.append(prev_row)
                                    # Create new rows for the rest
                                    for num in numbers[1:]:
                                        new_row = df.loc[idx].copy()
                                        new_row[column] = num
                                        new_rows.append(new_row)
                                else:
                                    new_rows.append(df.loc[idx].copy())

                            df = pd.DataFrame(new_rows).reset_index(drop=True)
                            df = clean_numerical_columns(df, [column])
                        else:
                            df = clean_numerical_columns(df, [column])
                    except Exception as e:
                        logger.error(f"Error converting {column} to float: {str(e)}")
                        df[column] = df[column].map(
                            lambda x: pd.NA if check_if_null(x) else x
                        )
                        try:
                            df[column] = df[column].astype(float)
                        except Exception as e:
                            logger.error(f"Could not convert {column} to float")
                            error_log.append(
                                {
                                    "error_name": (
                                        f"Error converting column {column} "
                                        f"to float",
                                    ),
                                    "error_message": (
                                        f"Error: {str(e)}.\n"
                                        f"Kindly recheck the observation data "
                                        f"for column {column}."
                                    ),
                                }
                            )
                            logger.info(f"traceback: {traceback.format_exc()}")
                            return None, None, error_log, table_structure_dictionary

                else:
                    logger.debug(f"  Column '{column}' unchanged")

            dataframe_dictionary[key] = df

        # Update table structure for columns converted to string
        for table_key, columns in columns_to_convert_to_string.items():
            if table_key in table_structure_dictionary:
                updated_structure = []
                for col_def in table_structure_dictionary[table_key]:
                    if col_def["name"] in columns:
                        # Create a copy and update d_type to string
                        updated_col_def = col_def.copy()
                        updated_col_def["d_type"] = "string"
                        updated_structure.append(updated_col_def)
                        logger.info(
                            f"Updated table structure for column {col_def['name']} "
                            f"in table {table_key} from {col_def['d_type']} to string"
                        )
                    else:
                        updated_structure.append(col_def)
                table_structure_dictionary[table_key] = updated_structure

        dose = dataframe_dictionary.get(TableNames.DOSING.value)
        return dataframe_dictionary, dose, error_log, table_structure_dictionary

    except Exception as e:
        error_log.append(
            {
                "error_name": "Error loading observation data",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck the observation data."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")

        return None, None, error_log, table_structure_dictionary


# * Merge Observation Tables
def merge_observation_data(dataframe_dictionary):
    error_log = []
    try:
        # * ### Merge Observation Tables
        observation_tables = [
            TableNames.OBSERVATION.value,
        ]

        observation_dataframe = pd.DataFrame()
        for key, _df in dataframe_dictionary.items():
            if key in observation_tables:
                logger.debug(key)
                observation_dataframe = pd.concat(
                    [observation_dataframe, dataframe_dictionary[key]]
                )

        obs = observation_dataframe.copy()

        if obs.shape[0] > 0:
            # Formats ARM_TIME units to one common unit
            def standardize_unit(unit):
                unit = unit.strip().lower()
                if unit in ["h", "hr", "hour", "hourly"]:
                    return "hours"
                elif unit in ["min", "m", "minute", "minutes"]:
                    return "minutes"
                elif unit in ["d", "day", "daily", "od", "Days"]:
                    return "days"
                elif unit in ["w", "week", "weekly"]:
                    return "weeks"
                elif unit in ["m", "month", "monthly"]:
                    return "months"
                elif unit in ["hours", "days", "weeks", "months", "minutes"]:
                    return unit
                else:
                    raise ValueError(f"Unknown unit: {unit}")

            # Applying the standardize_unit function to the relevant columns
            if "ARM_TIME_UNIT" in obs.columns:
                try:
                    obs["ARM_TIME_UNIT"] = obs["ARM_TIME_UNIT"].apply(standardize_unit)
                except Exception as e:
                    logger.error(f"Error standardizing unit: {e}")
                    error_log.append(
                        {
                            "error_name": "Unknown unit found in observation data",
                            "error_message": (
                                f"Error: {e}. "
                                "Kindly recheck the units in the "
                                "observation data for the column ARM_TIME_UNIT."
                            ),
                        }
                    )
                    logger.info(f"traceback: {traceback.format_exc()}")
                    return None, error_log

            # HOTFIX If DVID does not equal 2 then round to nearest whole number ,
            # if negative floors at 0
            if "DVID" in obs.columns and "ARM_TIME" in obs.columns:
                mask = obs["DVID"] != 2
                obs.loc[mask, "ARM_TIME"] = (
                    obs.loc[mask, "ARM_TIME"].clip(lower=0).round()
                )

            def get_conversion_factor(from_unit, to_unit):
                conversion_factors = {
                    "minutes": {
                        "minutes": 1,
                        "hours": 1 / 60,
                        "days": 1 / 1440,
                        "weeks": 1 / 10080,
                        "months": 1 / 43830,
                    },
                    "hours": {
                        "minutes": 60,
                        "hours": 1,
                        "days": 1 / 24,
                        "weeks": 1 / 168,
                        "months": 1 / 730.5,
                    },
                    "days": {
                        "minutes": 1440,
                        "hours": 24,
                        "days": 1,
                        "weeks": 1 / 7,
                        "months": 1 / 30.4375,
                    },
                    "weeks": {
                        "minutes": 10080,
                        "hours": 168,
                        "days": 7,
                        "weeks": 1,
                        "months": 1 / 4.348125,
                    },
                    "months": {
                        "minutes": 43830,
                        "hours": 730.5,
                        "days": 30.4375,
                        "weeks": 4.348125,
                        "months": 1,
                    },
                }
                return conversion_factors[from_unit][to_unit]

            # Function to define the rank of each time unit
            def get_unit_rank(unit):
                unit_ranks = {
                    "minutes": 1,
                    "hours": 2,
                    "days": 3,
                    "weeks": 4,
                    "months": 5,
                }
                return unit_ranks[unit]

            def convert_to_lowest_unit(comb, time_col, unit_col):
                # Find the lowest unit based on ranks
                obs["unit_rank"] = obs[unit_col].apply(get_unit_rank)
                lowest_unit_rank = obs["unit_rank"].min()
                lowest_unit = obs[obs["unit_rank"] == lowest_unit_rank][unit_col].iloc[
                    0
                ]

                # Convert all values to the lowest unit
                def convert_row(row):
                    from_unit = row[unit_col]
                    time_value = row[time_col]

                    # Use the conversion factor to convert to the lowest unit
                    conversion_factor = get_conversion_factor(from_unit, lowest_unit)
                    return time_value * conversion_factor

                obs[time_col] = obs.apply(lambda row: convert_row(row), axis=1)

                # Update the unit column to the lowest unit
                obs[unit_col] = lowest_unit

                # Drop the temporary 'unit_rank' column
                obs.drop(columns=["unit_rank"], inplace=True)

                return obs

            # Apply the conversion function
            if "ARM_TIME" in obs.columns and "ARM_TIME_UNIT" in obs.columns:
                obs = convert_to_lowest_unit(obs, "ARM_TIME", "ARM_TIME_UNIT")

            # replace DV_VAR with VAR if VAR exists
            if "VAR" in obs.columns:
                if "DV_VAR" in obs.columns:
                    obs = obs.drop(columns=["DV_VAR"])
                obs.rename(columns={"VAR": "DV_VAR"}, inplace=True)

            # ### Filling the observation values
            obs_group_column = find_group_name_column(obs)

            # Build sort columns that exist
            sort_cols = []
            if obs_group_column and obs_group_column in obs.columns:
                sort_cols.append(obs_group_column)
            if "ARM_TIME" in obs.columns:
                sort_cols.append("ARM_TIME")
            if sort_cols:
                obs = obs.sort_values(by=sort_cols)

            columns_to_exclude_from_fill = [
                obs_group_column,
                "ARM_TIME",
                "DV",
                "DVID",
                "UCI",
                "LCI",
                "ENDPOINT",
                "DV_VAR_STAT",
                "DV_STAT",
                "DV_VAR",
                "VAR",
                "VARU",
                "LLOQ",
                "BQL",
                "N_ARM",
                "N_STUDY",
            ]
            columns_to_fill = obs.columns.difference(columns_to_exclude_from_fill)
            if obs_group_column and obs_group_column in obs.columns:
                obs[columns_to_fill] = obs.groupby(obs_group_column)[
                    columns_to_fill
                ].transform(lambda group: group.ffill().bfill())

            # ## **Dose Merging**
            observation = obs
        else:
            observation = pd.DataFrame()

        return observation, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error merging observation data",
                "error_message": (
                    f"Error: {str(e)}.\n"
                    "Kindly recheck the observation data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Preprocess combined observation table
def observation_table_preprocessing(observation):
    error_log = []
    try:
        # region PREPARING OBSERVATION

        # *  ### Essential Changes in Observation Rows
        # Define a function to process each value in the 'DV' column
        def process_dv_value(x):

            # Try to convert the value to a numeric type return nan if NaN
            numeric_value = pd.to_numeric(x, errors="coerce")

            # If the conversion was successful (not NaN), return the numeric value
            if pd.notna(numeric_value):
                return numeric_value

            # If conversion failed, extract the first number from the string
            else:
                # Use regex to find the first number (integer or float) in the string
                match = re.search(r"([-+]?\d*\.?\d+)", str(x))
                # Convert the matched string to a float and return it
                return float(match.group())

        # Apply the processing function to the 'DV' column
        if "DV" in observation.columns:
            observation["DV"] = observation["DV"].apply(process_dv_value)

        # Rounding the Observation Column
        if "ARM_TIME" in observation.columns:
            observation["ARM_TIME"] = pd.to_numeric(observation["ARM_TIME"])

        # If DVID does not equal 2 then round to nearest whole number ,
        # if negative floors at 0
        if "DVID" in observation.columns and "ARM_TIME" in observation.columns:
            mask = observation["DVID"] != 2
            observation.loc[mask, "ARM_TIME"] = (
                observation.loc[mask, "ARM_TIME"].clip(lower=0).round()
            )

        # If DVID does equal 2 then any negative DV values will change to 0
        if (
            "DVID" in observation.columns
            and "DV" in observation.columns
            and "ARM_TIME" in observation.columns
        ):
            mask = observation["DVID"] == 2
            observation.loc[mask, "DV"] = observation.loc[mask, "DV"].clip(lower=0)
            observation.loc[mask, "ARM_TIME"] = observation.loc[mask, "ARM_TIME"].clip(
                lower=0
            )

        # If DVID equals 5, 6, or 7, DV should be rounded to whole
        # numbers and no negatives
        # mask = observation['DVID'].isin([5,6,7])
        # observation.loc[mask,'DV'] =
        # observation.loc[mask,'DV'].clip(lower=0).round(2)

        # Rounding adverse event measurements to whole numbers.
        """Possibly still need to turn negatives to 0"""

        # If value is below zero will be automatically set to 0
        if "DVID" in observation.columns and "DV" in observation.columns:
            observation["DV"] = np.where(
                observation["DVID"].isin([5, 6, 7]),
                round(observation["DV"].clip(lower=0)),
                observation["DV"],
            )

            # If DVID is 2 (Plasma Concentration) floor will be 0
            observation["DV"] = np.where(
                observation["DVID"].isin([2]),
                observation["DV"].clip(lower=0),
                observation["DV"],
            )

        # Capital Letters in STD.TRT, ARM.TRT, and ARM.TRTCLASS
        uppercase_columns = ["STD_TRT", "STD_TRT_CLASS", "ARM_TRT", "ARM_TRT_CLASS"]
        for column in uppercase_columns:
            if column in observation.columns:
                observation[column] = observation[column].str.upper()
        capitalize_columns = ["STD_TRT", "ARM_TRT"]
        for column in capitalize_columns:
            if column in observation.columns:
                observation[column] = observation[column].str.capitalize()

        # Small letters in REGIMEN
        lowercase_columns = ["REGIMEN"]
        for column in lowercase_columns:
            if column in observation.columns:
                observation[column] = observation[column].str.lower().str.strip()

        # Renaming UCI and LCI to DV_UCI and DV_LCI
        if "UCI" in observation.columns:
            observation = observation.rename(columns={"UCI": "DV_UCI"})
        if "LCI" in observation.columns:
            observation = observation.rename(columns={"LCI": "DV_LCI"})
        if "N_ARM" in observation.columns:
            observation = observation.rename(columns={"N_ARM": "N_ARM_TRIAL"})

        # Clearning out unwanted spaces in Line column
        obs_group_column = find_group_name_column(observation)
        if obs_group_column and obs_group_column in observation.columns:
            observation[obs_group_column] = observation[obs_group_column].str.strip()

        # remove duplicate rows
        observation = observation.drop_duplicates()
        return observation, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error preprocessing observation data",
                "error_message": (
                    f"Error: {str(e)}.\n"
                    "Kindly recheck the observation data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Dosing Table Preperation
def dosing_table_preprocessing(dose):
    error_log = []
    try:
        # *  Essential Changes Required in the Dose columns
        # region
        dose_group_column = find_group_name_column(dose)

        # Build list of columns that exist
        dose_columns_to_drop = DOSING_COLUMNS_TO_DROP
        dose = dose.drop(columns=dose_columns_to_drop, errors="ignore")
        dose = dose.assign(DV=np.nan, DVID=np.nan)

        if dose_group_column and dose_group_column in dose.columns:
            dose = dose.assign(LINE=dose[dose_group_column])
            dose["LINE"] = dose[
                "LINE"
            ].str.strip()  # stripping out extra spaces in LINE column

        dose_group_column = find_group_name_column(dose)

        # * ARM_DURR Addition
        # region

        # Initialize ARM_DURR column with NaN values
        # remove these columns from dose
        # dose["ARM_DUR"] = np.nan
        # dose["ARM_DUR_UNIT"] = dose["II_UNIT"].copy()

        # Ensure AMT column is float for calculation
        if "AMT" in dose.columns:
            dose["AMT"] = dose["AMT"].astype(float)

        # Iterate through each group in the dataset
        # arm duration is not used in the code, remove this logic
        # for group_name, group in dose.groupby(dose_group_column):
        # Initialize the total duration for this group
        # arm_durr = 0

        # amt_sum = group["AMT"].astype(float).sum()

        # Check if this is a placebo dose (AMT = 0 for entire group)
        # if group["AMT"].astype(float).sum() == 0:
        # if len(group) > 1:
        #     error_log.append(
        #         {
        #             "error_name": "More than 1 Placebo Dosing Row",
        #             "error_message": (
        #                 "Kindly add the correct number of "
        #                 "placebo dosing rows."
        #             ),
        #         }
        #     )
        #     logger.info(f"traceback: {traceback.format_exc()}")
        #     return None, error_log
        # Get first row of placebo, values should not change
        # row = group.iloc[0]

        # arm_durr = row["II"] * (row["ADDL"] + 1)

        # else:
        # Process each row in the current group
        # for _index, row in group.iterrows():
        # Calculate duration for current dose
        # II = Interdose Interval
        # ADDL = Additional Doses
        # Formula: Interval * (Additional doses + 1)

        # calculated_durr = row["II"] * (row["ADDL"] + 1)

        # if row["AMT"] == 0 and amt_sum != 0:
        #     calculated_durr = 0

        # logger.debug(
        #     f"Duration for dose in group {group_name}: {calculated_durr}"
        # )

        # Add to total duration for this group
        # arm_durr += calculated_durr

        # Update ARM_DURR for all rows in this group
        # arm_durr is not used in the code, remove this column from dose
        # dose.loc[dose[dose_group_column] == group_name, "ARM_DUR"] = arm_durr
        # dose["ARM_DUR"] = dose["ARM_DUR"].astype(float)
        # endregion

        return dose, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error preprocessing dosing table",
                "error_message": (
                    f"Error: {str(e)}.\n"
                    "Kindly recheck the dosing data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Combine observation and dosing
def combine_observation_and_dosing(observation, dose):
    error_log = []
    try:
        # region COMBINE OBSERVATION AND DOSING
        obs_group_column = find_group_name_column(observation)
        dose_group_column = find_group_name_column(dose)
        if obs_group_column != dose_group_column:
            if not check_if_null(obs_group_column):
                dose.rename(columns={dose_group_column: obs_group_column}, inplace=True)
                dose_group_column = obs_group_column
            else:
                observation.rename(
                    columns={obs_group_column: dose_group_column}, inplace=True
                )
                obs_group_column = dose_group_column

        merge_columns = [obs_group_column, "ARM_TIME", "ARM_TIME_UNIT", "DV", "DVID"]
        updated_merge_columns = []
        for column in merge_columns:
            if column in observation.columns or column in dose.columns:
                updated_merge_columns.append(column)
        merge_columns = updated_merge_columns

        if "DVID" in observation.columns:
            logger.info(f"dvid types: {set(observation['DVID'].unique())}")
        comb = pd.merge(
            observation,
            dose,
            how="outer",
            on=merge_columns,
            suffixes=("", "_dosing"),
        )
        # NEED to change lines below to UCI and LCI be NANs
        # if DV_STAT does not equal 90% CI (only will work in this case)
        if "DV_UCI" in comb.columns and "DV" in comb.columns:
            comb["DV_UCI"] = np.where(comb["DV"].notna(), comb["DV_UCI"], np.nan)
        if "DV_LCI" in comb.columns and "DV" in comb.columns:
            comb["DV_LCI"] = np.where(comb["DV"].notna(), comb["DV_LCI"], np.nan)

        # add evid and mdv columns
        if "AMT" in comb.columns:
            comb["EVID"] = comb["AMT"].apply(lambda x: 0 if check_if_null(x) else 1)
            comb["MDV"] = comb["AMT"].apply(lambda x: 0 if check_if_null(x) else 1)

        # ### Filling up the necessary columns after merging

        # Build sort columns that exist
        sort_cols_for_fill = [obs_group_column, "ARM_TIME", "AMT", "ENDPOINT"]
        existing_sort_cols = [
            col for col in sort_cols_for_fill if col and col in comb.columns
        ]
        if existing_sort_cols:
            comb = comb.sort_values(by=existing_sort_cols)

        # List of columns to exclude (those present in 'dose')
        columns_to_exclude_from_fill = [
            obs_group_column,
            "ARM_TIME",
            "DV",
            "DVID",
            "DV_UCI",
            "DV_LCI",
            "DV_VAR_STAT",
            "DV_STAT",
            "DV_VAR",
            "VAR",
            "VARU",
            "AMT",
            "II",
            "ADDL",
        ]
        columns_to_fill = comb.columns.difference(columns_to_exclude_from_fill)

        # Fill missing values upwards within each group for the relevant columns
        if obs_group_column and obs_group_column in comb.columns:
            comb[columns_to_fill] = comb.groupby(obs_group_column)[
                columns_to_fill
            ].transform(lambda group: group.bfill().ffill())

        # ### Observation rows are moved before the dosing rows when time is zero

        # Create the conditional column for sorting
        if "ARM_TIME" in comb.columns and "DV" in comb.columns:
            comb["sort_col"] = np.where(
                (comb["ARM_TIME"] == 0) & (comb["DV"].notna()), -0.5, 0
            )
        else:
            comb["sort_col"] = 0

        # Sort the DataFrame based on the specified columns
        intended_columns_to_sort = [
            obs_group_column,
            "ARM_NUMBER",
            "ARM_TIME",
            "sort_col",
        ]
        actual_columns_to_sort = [
            col for col in intended_columns_to_sort if col in comb.columns
        ]
        comb = comb.sort_values(by=actual_columns_to_sort)

        # Removing Sort Column
        comb = comb.drop(columns=["sort_col"])

        # endregion

        # for columns with suffixes, keep column from observation
        # and drop column from dosing. remove the suffix from
        # the observation column name.
        comb = comb.drop(columns=[col for col in comb.columns if "_dosing" in col])

        return comb, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error combining observation and dosing",
                "error_message": (
                    f"Error: {str(e)}.\n"
                    "Kindly recheck the observation and "
                    "dosing data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Standardize all units
def unit_standardization(comb):
    error_log = []
    try:
        # region Unit Standardization

        # Standardisation of Units
        def standardize_unit(unit):
            if isinstance(unit, str):
                unit = unit.strip().lower()
                if unit in ["h", "hr", "hour", "hourly"]:
                    return "hours"
                elif unit in ["min", "m", "minute", "minutes", "Minutes"]:
                    return "minutes"
                elif unit in ["d", "day", "daily", "od", "Days", "Day"]:
                    return "days"
                elif unit in ["w", "week", "weekly"]:
                    return "weeks"
                elif unit in ["m", "month", "monthly"]:
                    return "months"
                elif unit in ["hours", "days", "weeks", "months", "minutes"]:
                    return unit
                else:
                    raise ValueError(f"Unknown unit: {unit}")

            else:
                return unit

        # Applying the standardize_unit function to the relevant columns
        if "ARM_TIME_UNIT" in comb.columns:
            try:
                comb["ARM_TIME_UNIT"] = comb["ARM_TIME_UNIT"].apply(standardize_unit)
            except Exception as e:
                logger.error(f"Error standardizing unit: {e}")
                error_log.append(
                    {
                        "error_name": "Unknown unit found in combined data",
                        "error_message": (
                            f"Error: {e}. "
                            "Kindly recheck the units in the "
                            "combined data for the column ARM_TIME_UNIT."
                        ),
                    }
                )
                logger.info(f"traceback: {traceback.format_exc()}")
                return None, error_log

        # HOTFIX If DVID does not equal 2 then
        # round to nearest whole number , if negative floors at 0
        if "DVID" in comb.columns and "ARM_TIME" in comb.columns:
            mask = comb["DVID"] != 2
            comb.loc[mask, "ARM_TIME"] = (
                comb.loc[mask, "ARM_TIME"].clip(lower=0).round()
            )

        if "II_UNIT" in comb.columns:
            try:
                comb["II_UNIT"] = comb["II_UNIT"].apply(standardize_unit)
            except Exception as e:
                logger.error(f"Error standardizing unit: {e}")
                error_log.append(
                    {
                        "error_name": "Unknown unit found in combined data",
                        "error_message": (
                            f"Error: {e}. "
                            "Kindly recheck the units in the "
                            "combined data for the column II_UNIT."
                        ),
                    }
                )
                logger.info(f"traceback: {traceback.format_exc()}")
                return None, error_log
        # TODO: not needed right now, removing
        # comb["TIME_UNIT"] = comb["II_UNIT"]
        # TODO: not needed right now, removing

        # ### Formatting ARM_TIME units to one common unit
        def get_conversion_factor(from_unit, to_unit):
            conversion_factors = {
                "minutes": {
                    "minutes": 1,
                    "hours": 1 / 60,
                    "days": 1 / 1440,
                    "weeks": 1 / 10080,
                    "months": 1 / 43830,
                },
                "hours": {
                    "minutes": 60,
                    "hours": 1,
                    "days": 1 / 24,
                    "weeks": 1 / 168,
                    "months": 1 / 730.5,
                },
                "days": {
                    "minutes": 1440,
                    "hours": 24,
                    "days": 1,
                    "weeks": 1 / 7,
                    "months": 1 / 30.4375,
                },
                "weeks": {
                    "minutes": 10080,
                    "hours": 168,
                    "days": 7,
                    "weeks": 1,
                    "months": 1 / 4.348125,
                },
                "months": {
                    "minutes": 43830,
                    "hours": 730.5,
                    "days": 30.4375,
                    "weeks": 4.348125,
                    "months": 1,
                },
            }
            return conversion_factors[from_unit][to_unit]

        # Function to define the rank of each time unit
        def get_unit_rank(unit):
            unit_ranks = {"minutes": 1, "hours": 2, "days": 3, "weeks": 4, "months": 5}
            return unit_ranks[unit]

        def convert_to_lowest_unit(comb, time_col, unit_col):
            # Find the lowest unit based on ranks
            comb["unit_rank"] = comb[unit_col].apply(get_unit_rank)
            lowest_unit_rank = comb["unit_rank"].min()
            lowest_unit = comb[comb["unit_rank"] == lowest_unit_rank][unit_col].iloc[0]

            # Convert all values to the lowest unit
            def convert_row(row):
                from_unit = row[unit_col]
                time_value = row[time_col]

                # Use the conversion factor to convert to the lowest unit
                conversion_factor = get_conversion_factor(from_unit, lowest_unit)
                return time_value * conversion_factor

            comb[time_col] = comb.apply(lambda row: convert_row(row), axis=1)

            # Update the unit column to the lowest unit
            comb[unit_col] = lowest_unit

            # Drop the temporary 'unit_rank' column
            comb.drop(columns=["unit_rank"], inplace=True)

            return comb

        # Apply the conversion function
        if "ARM_TIME" in comb.columns and "ARM_TIME_UNIT" in comb.columns:
            comb = convert_to_lowest_unit(comb, "ARM_TIME", "ARM_TIME_UNIT")

        # Sort values by arm time again after adding dosing
        if "ARM_TIME" in comb.columns:
            comb["ARM_TIME"] = comb["ARM_TIME"].replace(-0.0, 0.0)

        comb_group_column = find_group_name_column(comb)
        intended_sort_columns = [comb_group_column, "ARM_NUMBER", "ARM_TIME"]
        actual_sort_columns = [
            col for col in intended_sort_columns if col in comb.columns
        ]
        comb = comb.sort_values(by=actual_sort_columns)

        # ## **Creating additional Essential NONMEM data set  columns**
        # TODO: not needed right now, removing
        # comb["EVID"] = comb.apply(
        #     lambda row: 0 if pd.notna(row["DV"]) else 1, axis=1
        # ).astype(int)

        # comb["MDV"] = comb["EVID"]

        # # ### Create Time_TSFD column #needs to be changed
        # # Group by 'comb_group_column' and calculate TIME_TSFD for comb
        # def calculate_time_tsfd(group):
        #     # Get the first non-missing 'AMT' entry and its corresponding 'ARM_TIME'
        #     first_non_missing_amt_time = group.loc[group["AMT"].notna(), "ARM_TIME"]
        #     first_non_missing_amt_time = first_non_missing_amt_time.iloc[0]

        #     # Subtract 'ARM_TIME' from the first non-missing 'AMT' entry's time
        #     group["TIME_TSFD"] = group["ARM_TIME"] - first_non_missing_amt_time
        #     return group

        # # Convert between units
        # def convert_time_row(org_time_col, org_time_unit, target_time_unit):

        #     # Use the conversion factor to convert to the lowest unit
        #     conversion_factor = get_conversion_factor(org_time_unit, target_time_unit)
        #     return org_time_col * conversion_factor

        # # Apply the function grouped by 'comb_group_column'
        # comb = (
        #     comb.groupby(comb_group_column)
        #     .apply(calculate_time_tsfd)
        #     .reset_index(level=[0, 1], drop=True)
        # )

        # comb["TIME_TSFD"] = np.where(comb["TIME_TSFD"] < 0, 0, comb["TIME_TSFD"])

        # # Convert units of TIME_TSFD to be consistent with TIME_UNITS
        # comb["TIME_TSFD"] = convert_time_row(
        #     comb["TIME_TSFD"],
        #     comb["ARM_TIME_UNIT"].iloc[0],
        #     comb["TIME_UNIT"].iloc[0]
        # ).astype(float)

        # TODO: not needed right now, removing
        # endregion

        return comb, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error standardizing units",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck the data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Temporary Column Creation
def temp_col_creation(comb):
    error_log = []
    try:
        # region TEMPORARY COLUMNS

        # * ### Creating temporary columns

        # Creating ADDL_FILLED column (ADDL+1)
        def fill_within_groups(df):
            if "ADDL" in df.columns:
                df["ADDL_filled"] = (df["ADDL"] + 1).ffill().bfill()
            return df

        comb_group_column = find_group_name_column(comb)
        if comb_group_column and comb_group_column in comb.columns:
            comb = (
                comb.groupby(comb_group_column)
                .apply(fill_within_groups)
                .reset_index(drop=True)
            )

        # Creating II_FILLED column (II_FILLED)
        if "II" in comb.columns:
            comb["II_filled"] = comb["II"].ffill().bfill()

        # creating STOP_TIME column
        # Dose stop time is represented by stop time colummn
        # formula = TIME_TSFD where addl entry  + ADDL value

        def fill_within_groups2(df):
            if "ADDL" in df.columns:
                df["ADDL_filled2"] = (df["ADDL"] + 1).ffill().bfill()
                if "II_filled" in df.columns:
                    df["ADDL_filled2"] = (
                        df["ADDL_filled2"] * df["II_filled"]
                    )  # by multipliying by ii this gives us the total
                    # number of doses that needs to be added to the time
                    # note here we need to multiply and not divide as addl
                    # indicates how many additional doses needed to be added,
            return df

        if "LINE" in comb.columns:
            comb = (
                comb.groupby("LINE").apply(fill_within_groups2).reset_index(drop=True)
            )

        # creating STOP_TIME
        # TODO: not needed right now, removing
        # comb["STOP_TIME"] = (
        #     comb["TIME_TSFD"].where(comb["ADDL"].notna()).ffill()
        #     + comb["ADDL_filled2"]
        # )

        # # creating STOP_DOSE
        # comb["STOP_DOSE"] = comb["STOP_TIME"] / comb["II_filled"]
        # TODO: not needed right now, removing

        # endregion

        # region TIME_TSLD COLUMN
        # * ### TIME_TSLD column

        # New TSLD and N_Dose Function
        # def find_closest_without_exceeding(lst, target):
        #     # Filter the list to only include values <= target
        #     valid_values = [x for x in lst if x <= target]

        #     # If there are no valid values, return None or handle as appropriate
        #     if not valid_values:
        #         return np.nan

        #     # Find the min value based on how close it is to the target
        #     return min(valid_values, key=lambda x: target - x)

        # def new_TSLD_function(group):
        #     # Calculate full list of times that doses occur
        #     # Check if required columns exist
        #     required_cols = [
        #         "II_UNIT", "ARM_TIME_UNIT", "II", "AMT", "ADDL", "ARM_TIME"
        #     ]
        #     if not all(col in group.columns for col in required_cols):
        #         return pd.Series([np.nan] * len(group), index=group.index)

        #     Dose_Times = []
        #     counter = 0
        #     for _idx, row in group.iterrows():
        #         # Convert II to same units as ARM_TIME
        #         II_rank = get_unit_rank(row["II_UNIT"])
        #         ARM_rank = get_unit_rank(row["ARM_TIME_UNIT"])

        #         # check this logic, is it greater than or equal to?
        #         if II_rank >= ARM_rank:
        #             II_converted = convert_time_row(
        #                 row["II"], row["II_UNIT"], row["ARM_TIME_UNIT"]
        #             )
        #             # II_converted= round(II_converted, 4)

        #             if not (check_if_null(row["AMT"])) and not (
        #                 check_if_null(row["ADDL"])
        #             ):
        #                 Dose_Times.append(row["ARM_TIME"])
        #                 counter += 1
        #                 for _i in range(int(row["ADDL"])):
        #                     Dose_Times.append(
        #                         round((Dose_Times[counter - 1] + II_converted), 4)
        #                     )
        #                     counter += 1
        #         else:
        #             ARM_TIME_converted = convert_time_row(
        #                 row["ARM_TIME"], row["ARM_TIME_UNIT"], row["II_UNIT"]
        #             )
        #             # Double check this formula
        #             if pd.notna(row["AMT"]):
        #                 Dose_Times.append(ARM_TIME_converted)
        #                 counter += 1
        #                 for _i in range(int(row["ADDL"])):
        #                     Dose_Times.append(Dose_Times[counter - 1] + row["II"])
        #                     counter += 1

        #     # Compare each ARM_TIME with dosing times to
        #     # find nearest dosing time (without exceeding)
        #     # NEED TO CHANGE THIS
        #     TSLD = []
        #     NDose = []
        #     if II_rank >= ARM_rank:
        #         for _idx, row in group.iterrows():
        #             closest_dose_time = find_closest_without_exceeding(
        #                 Dose_Times, row["ARM_TIME"]
        #             )

        #             if pd.notna(closest_dose_time):
        #                 TSLD.append(row["ARM_TIME"] - closest_dose_time)
        #                 number_of_dose = Dose_Times.index(closest_dose_time)
        #                 NDose.append(number_of_dose + 1)
        #             else:
        #                 TSLD.append(np.nan)
        #                 NDose.append(np.nan)
        #                 logger.debug("No Dose Times Before Arm Time")

        #     else:
        #         for _idx, row in group.iterrows():
        #             ARM_TIME_converted = convert_time_row(
        #                 row["ARM_TIME"], row["ARM_TIME_UNIT"], row["II_UNIT"]
        #             )

        #             closest_dose_time = find_closest_without_exceeding(
        #                 Dose_Times, ARM_TIME_converted
        #             )

        #             logger.debug(
        #                 f"""
        #                 ARMTIME: {row['ARM_TIME']}
        #                 ARMTIME CONVERTED: {ARM_TIME_converted}
        #                 Close Dose {closest_dose_time}
        #                 """
        #             )

        #             # ! ERROR
        #             if pd.notna(closest_dose_time):
        #                 TSLD.append(ARM_TIME_converted - closest_dose_time)
        #                 number_of_dose = Dose_Times.index(closest_dose_time)
        #                 NDose.append(number_of_dose + 1)
        #             else:
        #                 TSLD.append(np.nan)
        #                 NDose.append(np.nan)
        #                 logger.debug("No Dose Times Before Arm Time")
        # Convert final times so TIME_TSLD units = II units

        # return pd.Series(TSLD, index=group.index)

        # Run the new function, grouped by comb_group_column
        # TODO: not needed right now, removing
        # comb["TIME_TSLD"] = (
        #     comb.groupby([comb_group_column])
        #     .apply(new_TSLD_function)
        #     .squeeze()
        #     .reset_index(level=[0], drop=True)
        # )

        # # where it is not null, replace any negative TSLD values with 0
        # comb["TIME_TSLD"].loc[comb["TIME_TSLD"].notnull()] = np.where(
        #     comb["TIME_TSLD"].loc[comb["TIME_TSLD"].notnull()] < 0,
        #     0,
        #     comb["TIME_TSLD"].loc[comb["TIME_TSLD"].notnull()],
        # )

        # Convert so TIME_TSLD units = TIME units, TIME_TSLD units will
        # depend on rank of II_UNIT versus ARM_TIME_UNIT
        # II_rank = get_unit_rank(comb["II_UNIT"].iloc[0])
        # ARM_rank = get_unit_rank(comb["ARM_TIME_UNIT"].iloc[0])
        # if II_rank >= ARM_rank:
        #     # TSLD has ARM_TIME_UNITs, convert to TIME_UNIT
        #     comb["TIME_TSLD"] = convert_time_row(
        #         comb["TIME_TSLD"],
        #         comb["ARM_TIME_UNIT"].iloc[0],
        #         comb["TIME_UNIT"].iloc[0],
        #     ).astype(float)
        # else:
        #     # TSLD has II_UNITs, convert to TIME_UNIT
        #     # (just in case they are different)
        #     comb["TIME_TSLD"] = convert_time_row(
        #         comb["TIME_TSLD"],
        #         comb["II_UNIT"].iloc[0],
        #         comb["TIME_UNIT"].iloc[0]
        #     ).astype(float)
        # TODO: not needed right now, removing

        # endregion

        # region N_DOSE

        # N_Dose column
        # TODO: not needed right now, removing
        # def N_Dose_function(group):
        #     # Calculate full list of times that doses occur

        #     Dose_Times = []
        #     counter = 0
        #     for _idx, row in group.iterrows():
        #         # Convert II to same units as ARM_TIME
        #         II_rank = get_unit_rank(row["II_UNIT"])
        #         ARM_rank = get_unit_rank(row["ARM_TIME_UNIT"])

        #         # check this logic, is it greater than or equal to?
        #         if II_rank >= ARM_rank:
        #             II_converted = convert_time_row(
        #                 row["II"], row["II_UNIT"], row["ARM_TIME_UNIT"]
        #             )
        #             # II_converted= round(II_converted, 4)

        #             if not check_if_null(row["AMT"]) and not check_if_null(
        #                 row["ADDL"]
        #             ):
        #                 Dose_Times.append(row["ARM_TIME"])
        #                 counter += 1
        #                 for _i in range(int(row["ADDL"])):
        #                     Dose_Times.append(
        #                         round((Dose_Times[counter - 1] + II_converted), 4)
        #                     )
        #                     counter += 1
        #         else:
        #             ARM_TIME_converted = convert_time_row(
        #                 row["ARM_TIME"], row["ARM_TIME_UNIT"], row["II_UNIT"]
        #             )
        #             # Double check this formula
        #             if not check_if_null(row["AMT"]) and not check_if_null(
        #                 row["ADDL"]
        #             ):
        #                 Dose_Times.append(ARM_TIME_converted)
        #                 counter += 1
        #                 for _i in range(int(row["ADDL"])):
        #                     Dose_Times.append(Dose_Times[counter - 1] + row["II"])
        #                     counter += 1

        #     # Compare each ARM_TIME with dosing times to
        #     # find nearest dosing time (without exceeding)

        #     NDose = []
        #     if II_rank >= ARM_rank:
        #         for _idx, row in group.iterrows():
        #             closest_dose_time = find_closest_without_exceeding(
        #                 Dose_Times, row["ARM_TIME"]
        #             )

        #             if pd.notna(closest_dose_time):

        #                 number_of_dose = Dose_Times.index(closest_dose_time)
        #                 NDose.append(number_of_dose + 1)
        #             else:

        #                 NDose.append(np.nan)
        #                 logger.debug("No Dose Times Before Arm Time")

        #     else:
        #         for _idx, row in group.iterrows():
        #             ARM_TIME_converted = convert_time_row(
        #                 row["ARM_TIME"], row["ARM_TIME_UNIT"], row["II_UNIT"]
        #             )

        #             closest_dose_time = find_closest_without_exceeding(
        #                 Dose_Times, ARM_TIME_converted
        #             )

        #             logger.debug(
        #                 f"""
        #                 ARMTIME: {row['ARM_TIME']}
        #                 ARMTIME CONVERTED: {ARM_TIME_converted}
        #                 Close Dose {closest_dose_time}
        #                 """
        #             )

        #             if pd.notna(closest_dose_time):

        #                 number_of_dose = Dose_Times.index(closest_dose_time)
        #                 NDose.append(number_of_dose + 1)
        #             else:

        #                 NDose.append(np.nan)
        #                 logger.debug("No Dose Times Before Arm Time")

        #     return pd.Series(NDose, index=group.index)

        # # Run the new function, grouped by comb_group_column
        # comb["N_DOSE"] = (
        #     comb.groupby([comb_group_column])
        #     .apply(N_Dose_function)
        #     .squeeze()
        #     .reset_index(level=[0], drop=True)
        # ).astype(float)
        # TODO: not needed right now, removing
        # * ### Calculating N_DOSE column

        # endregion

        return comb, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error creating temporary columns",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck the data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Reordering dataframe columns
def reorder_dataframe_columns(comb):
    error_log = []
    try:
        # region REORDERING DATAFRAME COLUMNS
        comb_group_column = find_group_name_column(comb)
        # Define the desired order of columns
        desired_order = [
            "FILE_NAME",
            "STU_NUMBER",
            "LINE",
        ]
        if comb_group_column:
            desired_order.append(comb_group_column)
        desired_order.extend(
            [
                "AU",
                "TI",
                "JR",
                "PY",
                "VL",
                "IS",
                "PG",
                "PUBMEDID",
                "LA",
                "REGID",
                "REGNM",
                "TP",
                "TS",
                "DOI_URL",
                "CIT_URL",
                "STD_IND",
                "STD_TRT",
                "STD_TRT_CLASS",
                "SOURCE",
                "NUMBER_OF_ARMS",
                "ARM_NUMBER",
                "ARM_TRT",
                "ARM_TRT_CLASS",
                "ARM_DOSE",
                "ARM_DOSE_UNIT",
                "AMT",
                "AMT_UNIT",
                "REGIMEN",
                "N_STUDY",
                "N_ARM",
                "ROUTE",
                "ARM_TIME",
                "ARM_TIME_UNIT",
                "DVID",
                "ENDPOINT",
                "DV_STAT",
                "DV",
                "DV_UNIT",
                "DV_VAR_STAT",
                "DV_VAR",
                "DV_UCI",
                "DV_LCI",
                "LLOQ",
                "BQL",
                # "EVID", # TODO: not needed right now, removing
                # "MDV", # TODO: not needed right now, removing
                "ADDL",
                "II",
                "II_UNIT",
                # "TIME_TSFD", # TODO: not needed right now, removing
                # "TIME_TSLD", # TODO: not needed right now, removing
                # "TIME_UNIT", # TODO: not needed right now, removing
                # "N_DOSE", # TODO: not needed right now, removing
            ]
        )
        modified_desired_order = []
        for col in desired_order:
            if col and col in comb.columns:
                modified_desired_order.append(col)
        for col in comb.columns:
            if col not in desired_order:
                modified_desired_order.append(col)

        # Get columns present in the DataFrame
        present_cols = comb.columns.tolist()

        # Reorder columns based on the desired_order
        reordered_cols = [col for col in desired_order if col in present_cols]

        # Join columns not in desired_order at the end
        other_cols = [col for col in present_cols if col not in reordered_cols]
        final_order = reordered_cols + other_cols

        # Reorder the DataFrame
        comb = comb[final_order]

        # ### Dropping the temporary columns
        temporary_columns = [
            "ADDL_filled",
            "II_filled",
            "ADDL_filled2",
            "STOP_TIME",
            "STOP_DOSE",
        ]
        # Only drop columns that exist in the DataFrame
        columns_to_drop = [col for col in temporary_columns if col in comb.columns]
        if columns_to_drop:
            comb = comb.drop(columns=columns_to_drop)

        # endregion

        return comb, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error reordering dataframe columns",
                "error_message": str(e),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Dosing Rows QC
def dosing_rows_qc(comb):
    error_log = []
    try:
        # ## **QC for Dosing Rows**

        # region QC DOSING ROWS

        # nonmem dosing rows
        comb_group_column = find_group_name_column(comb)

        # Build list of columns that exist for nonmem_data
        nonmem_columns = [
            "LINE",
            comb_group_column,
            "ARM_TIME",
            "ARM_TIME_UNIT",
            "DV",
            "DVID",
            "ENDPOINT",
            "AMT",
            "AMT_UNIT",
            "ADDL",
            "II",
            "II_UNIT",
        ]
        existing_nonmem_columns = [col for col in nonmem_columns if col in comb.columns]

        nonmem_data = comb[existing_nonmem_columns]

        # NOTE: Removed no-op query that wasn't using its result
        # if "LINE" in nonmem_data.columns:
        #     nonmem_data.query('LINE == "placebo"')

        # only dosing rows
        # nonmem_data[nonmem_data.EVID == 1] # TODO: not needed right now, removing

        # Group by 'LINE' and perform the calculations
        if (
            "LINE" in nonmem_data.columns
            and "ADDL" in nonmem_data.columns
            and "AMT" in nonmem_data.columns
        ):
            dose_check = nonmem_data.groupby("LINE").agg(
                ADDL_sum=pd.NamedAgg(
                    column="ADDL", aggfunc=lambda x: x.sum(skipna=True)
                ),
                AMT_count=pd.NamedAgg(
                    column="AMT", aggfunc=lambda x: (~x.isna()).sum()
                ),
            )

            # Create the 'doses' column
            dose_check["doses"] = dose_check["ADDL_sum"] + dose_check["AMT_count"]

        # checking whether others is filled properly
        # other_cols = comb.drop(columns=nonmem_data.columns)
        # all columns should be false

        # comb["ENDPOINT"] = np.where(comb["EVID"] == 1, np.nan, comb["ENDPOINT"])
        # TODO: not needed right now, removing
        # endregion

        return comb, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error performing dosing rows QC",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck the data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Merge covariates to combined dosing and observation
def covariate_merge(dataframe_dictionary, comb, cov_table_structure: list[dict]):
    error_log = []
    try:
        # region COVARIATE MERGING
        # ## **Covariate merging** Step 3

        cov = dataframe_dictionary[TableNames.COVARIATE.value]
        cov_group_column = find_group_name_column(cov)
        # Adjust covariate handling based on type of covariate table
        cov_headers = cov.columns.to_list()

        if (
            "COV" in cov_headers
            and cov_group_column
            and cov_group_column in cov.columns
        ):

            # ### Checking Unique Enteries on Covariate Data
            if "TRIAL_ARM" in cov.columns:
                cov.groupby("TRIAL_ARM")["COV"].value_counts()

            # Build list of pivot values that exist
            pivot_values = [
                "COV_VAL",
                "COV_UNIT",
                "COV_STAT",
                "COV_VAR",
                "COV_VAR_STAT",
            ]
            existing_pivot_values = [col for col in pivot_values if col in cov.columns]

            cov_wide = cov.pivot(
                index=cov_group_column,
                columns="COV",
                values=existing_pivot_values,
            )

            # Flatten the MultiIndex columns and rename to desired format
            # (e.g., height_val, height_unit)
            cov_wide.columns = [
                f'{cov}_{var.split("_", 1)[1].upper()}' for var, cov in cov_wide.columns
            ]

            # Reset the index to make TRIAL_ARM a column
            cov_wide = cov_wide.reset_index()

            cov_wide.columns = cov_wide.columns.str.upper()

            cov_wide.columns = cov_wide.columns.str.replace(r"_VAL", "")
        else:
            cov_wide = cov

        # Check if all essential covariate columns are included
        # List of essential covariates:
        cov_list = [column["name"] for column in cov_table_structure]
        if cov_group_column and cov_group_column not in cov_list:
            cov_list.append(cov_group_column)

        # Compare lists

        # Find any covariates that are missing from our current set
        current_cov_list = cov_wide.columns.to_list()
        s = set(current_cov_list)
        missing_cov = [x for x in cov_list if x not in s]

        if missing_cov:
            raise ValueError(f"Covariates missing: {missing_cov}")

        # Find any covariates that are in our current
        # covariate set that are not essential, drop those

        s2 = set(cov_list)
        extra_cov = [x for x in current_cov_list if x not in s2]

        if extra_cov:
            cov_wide = cov_wide.drop(columns=extra_cov)
            logger.debug(f"Dropped extra covariates: {extra_cov}")

        # drop COVARIATE_COLUMNS_TO_DROP columns
        cov_wide = cov_wide.drop(columns=COVARIATE_COLUMNS_TO_DROP, errors="ignore")

        # Export the DataFrame to a CSV with the new filename
        comb_group_column = find_group_name_column(comb)
        if (
            comb_group_column
            and comb_group_column in comb.columns
            and comb_group_column in cov_wide.columns
        ):
            final = pd.merge(
                comb, cov_wide, on=comb_group_column, how="outer", suffixes=("", "_cov")
            )
            # remove columns with suffixes _cov
            final = final.drop(columns=[col for col in final.columns if "_cov" in col])
        else:
            # If no valid group column, just concatenate horizontally
            final = pd.concat([comb, cov_wide], axis=1)
        # endregion

        return final, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error merging covariates",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck the data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


# * Final Formatting
def final_formatting(final):
    error_log = []
    try:
        # ### Final Formatting before export
        final_group_column = find_group_name_column(final)

        # Make DV_LCI and DV_UCI be NAN if DV_VAR_STAT is blank
        if (
            "DV_VAR_STAT" in final.columns
            and "DV_LCI" in final.columns
            and "DV_UCI" in final.columns
        ):
            dv_var_stat_nan_mask = final["DV_VAR_STAT"].isna()
            final.loc[dv_var_stat_nan_mask, ["DV_LCI", "DV_UCI"]] = np.nan

        # Standardize case of some columns
        if "ENDPOINT" in final.columns:
            final["ENDPOINT"] = final["ENDPOINT"].astype(str).str.lower()
        if "ROUTE" in final.columns:
            final["ROUTE"] = final["ROUTE"].astype(str).str.lower().str.strip()

        # Replace Regimen naming of "OD" with "QD"
        if "REGIMEN" in final.columns:
            final["REGIMEN"] = final["REGIMEN"].astype(str).str.replace("OD", "qd")
            final["REGIMEN"] = final["REGIMEN"].astype(str).str.replace("od", "qd")

        # if ARM_TRT_CLASS is placebo or insulin, make it lowercase
        if "ARM_TRT_CLASS" in final.columns:
            final["ARM_TRT_CLASS"] = final["ARM_TRT_CLASS"].astype(str)
            mask = final["ARM_TRT_CLASS"].str.contains("PLACEBO", case=False, na=False)
            final.loc[mask, "ARM_TRT_CLASS"] = final.loc[
                mask, "ARM_TRT_CLASS"
            ].str.lower()

            mask = final["ARM_TRT_CLASS"].str.contains("INSULIN", case=False, na=False)
            final.loc[mask, "ARM_TRT_CLASS"] = final.loc[
                mask, "ARM_TRT_CLASS"
            ].str.lower()

        # If EVID is 1 (Dosing) then make LLOQ and DV_UNIT be nan
        # if "EVID" in final.columns:
        #     final.loc[final["EVID"] == 1, ["LLOQ", "DV_UNIT"]] = np.nan
        # TODO: not needed right now, removing

        # If DVID does not equal 2 then LLOQ should be NA
        if "DVID" in final.columns and "LLOQ" in final.columns:
            mask = final["DVID"] != 2
            final.loc[mask, "LLOQ"] = np.nan

        # Pick the first filename and use that consistently throughout
        if "FILE_NAME" in final.columns:
            first_file_name = final["FILE_NAME"].unique()[0]
            final["FILE_NAME"] = first_file_name

        # Apply some formatting to normalize naming of figure sources
        if "SOURCE" in final.columns:
            try:
                final["SOURCE"] = final["SOURCE"].apply(format_figure_reference)
            except Exception as e:
                error_log.append(
                    {
                        "error_name": "Original Figure reference is null",
                        "error_message": (
                            "Kindly recheck the data and rerun the merge. "
                            f"Error: {e}."
                        ),
                    }
                )

        # Replaces any instance of STD as a whole word to SD
        if "DV_VAR_STAT" in final.columns:
            final = final.map(
                lambda x: re.sub(r"\bSTD\b", "SD", x) if isinstance(x, str) else x
            )

        # Process DV_VAR_STAT: If 'C' present, assume
        # Confidence Interval (set DV_VAR to NaN);
        dv_var_stat_mask = None

        if "DV_VAR_STAT" in final.columns and "DV_VAR" in final.columns:
            dv_var_stat_col = final["DV_VAR_STAT"]
            # Handle duplicate columns - take the first column if DataFrame
            if isinstance(dv_var_stat_col, pd.DataFrame):
                dv_var_stat_col = dv_var_stat_col.iloc[:, 0]
            final["DV_VAR_STAT"] = dv_var_stat_col.astype(str)
            dv_var_stat_mask = final["DV_VAR_STAT"].str.contains(
                "[cCqQ]", case=False, na=False
            )
            # Ensure mask is a Series (not DataFrame) for proper boolean indexing
            if isinstance(dv_var_stat_mask, pd.DataFrame):
                dv_var_stat_mask = dv_var_stat_mask.iloc[:, 0]
            final.loc[dv_var_stat_mask, "DV_VAR"] = np.nan

        # Checks if the correct error bar option was turned on in the platform.
        # if confidence interval or interquartile range was stated,
        # UCI and LCI should not be nans
        if (
            "DV_UCI" in final.columns
            and dv_var_stat_mask is not None
            and final.loc[dv_var_stat_mask, "DV_UCI"].isnull().any()
        ):
            error_log.append(
                {
                    "error_name": (
                        "Wrong error bar option selected, "
                        "confidence interval columns blank"
                    ),
                    "error_message": "Kindly recheck the data and rerun the merge.",
                }
            )

        # else assume standard deviation (set DV_LCI and DV_UCI to NaN)
        if (
            "DV_UCI" in final.columns
            and "DV_LCI" in final.columns
            and dv_var_stat_mask is not None
        ):
            final.loc[~dv_var_stat_mask, ["DV_UCI", "DV_LCI"]] = np.nan

        # check to make sure standard deviation option was turned on
        if (
            "DV_VAR_STAT" in final.columns
            and "DV_VAR" in final.columns
            and dv_var_stat_mask is not None
        ):
            subset = final.loc[~dv_var_stat_mask]
            # Handle duplicate columns - take the first column if DataFrame
            dv_var_stat_vals = subset["DV_VAR_STAT"]
            dv_var_vals = subset["DV_VAR"]
            if isinstance(dv_var_stat_vals, pd.DataFrame):
                dv_var_stat_vals = dv_var_stat_vals.iloc[:, 0]
            if isinstance(dv_var_vals, pd.DataFrame):
                dv_var_vals = dv_var_vals.iloc[:, 0]
            condition = dv_var_stat_vals.notna().values & dv_var_vals.isna().values
            if condition.any():
                error_log.append(
                    {
                        "error_name": (
                            "Wrong error bar option selected, DV_VAR column is blank"
                        ),
                        "error_message": "Kindly recheck the data and rerun the merge.",
                    }
                )

        # HOTFIX If DVID does not equal 2 then round to nearest whole number ,
        # if negative floors at 0
        # NOTE: Must be done BEFORE fillna("NA") since DVID becomes string after
        if "DVID" in final.columns and "ARM_TIME" in final.columns:
            mask = final["DVID"] != 2
            final.loc[mask, "ARM_TIME"] = (
                final.loc[mask, "ARM_TIME"].clip(lower=0).round()
            )

        if (
            final_group_column
            and final_group_column in final.columns
            and "ARM_NUMBER" in final.columns
            and "ARM_TIME" in final.columns
        ):
            final = final.sort_values(by=[final_group_column, "ARM_NUMBER", "ARM_TIME"])

        if "DV_VAR_STAT" in final.columns:
            final["DV_VAR_STAT"].unique()

        # * edit the DVID and Endpoint to account for CFB versus absolute
        # NOTE: Must be done BEFORE fillna("NA") since DVID becomes string after
        # Create mapping dictionary
        dvid_mapping = {
            4: (8, "weight loss CFB"),
            3: (10, "hba1c CFB"),
            1: (9, "fasting plasma glucose CFB"),
        }

        if (
            "DV_STAT" in final.columns
            and "DVID" in final.columns
            and "ENDPOINT" in final.columns
        ):
            final["DV_STAT"] = final["DV_STAT"].astype(str)
            cfb_mask = final["DV_STAT"].str.contains("CFB", case=False, na=False)
            # Apply changes only to rows where cfb_mask is True
            for old_dvid, (new_dvid, new_endpoint) in dvid_mapping.items():
                condition = cfb_mask & (final["DVID"] == old_dvid)
                final.loc[condition, "DVID"] = new_dvid
                final.loc[condition, "ENDPOINT"] = new_endpoint

        # Replace all NaN values with 'NA'
        final = final.fillna("NA")

        # Drop any extra columns
        columns_to_drop = ["LINE", "LINE NAME", "LINE STATUS", "PUBMEDID"]
        for column in columns_to_drop:
            if column in final.columns:
                final = final.drop(columns=[column])

        # Move comment columns to the end of the dataframe
        comment_columns = ["PAPER COMMENTS", "COMMENTS", "EXTRACTION COMMENTS"]
        for comment_column in comment_columns:
            if comment_column in final.columns:
                final.insert(
                    len(final.columns) - 1, comment_column, final.pop(comment_column)
                )

        # Update PAPER COMMENTS column name and EXTRACTION COMMENTS TO HAVE _
        rename_dict = {}
        if "PAPER COMMENTS" in final.columns:
            rename_dict["PAPER COMMENTS"] = "PAPER_COMMENTS"
        if "EXTRACTION COMMENTS" in final.columns:
            rename_dict["EXTRACTION COMMENTS"] = "EXTRACTION_COMMENTS"
        if rename_dict:
            final.rename(columns=rename_dict, inplace=True)

        # Inserting fileID as study number
        # tracker_file_path = './roche_contract_delivery_tracker.xlsx'
        # file_tracker = pd.read_excel(tracker_file_path, 'merge_tracker')
        # mask= file_tracker['file_name']==first_file_name
        file_id = None
        if "FILEID" in final.columns:
            file_id = final["FILEID"].iloc[0]

        # Make Trial number if NA become 1
        if "TRIAL_NUMBER" in final.columns:
            final["TRIAL_NUMBER"] = (
                final["TRIAL_NUMBER"].replace("NA", np.nan).fillna(1)
            )

        # Concatenate FileID with Trial Number
        if (
            "FILEID" in final.columns
            and "TRIAL_NUMBER" in final.columns
            and file_id is not None
        ):
            final["STU_NUMBER"] = (
                str(file_id) + "_" + final["TRIAL_NUMBER"].astype(int).astype(str)
            )

        # Create ARM REGIMEN COLUMN by combining ARM_DOSE, DOSE_UNIT, and REGIMEN
        # TODO: not needed right now, removing
        # if (
        #     "ARM_DOSE" in final.columns
        #     and "ARM_DOSE_UNIT" in final.columns
        #     and "REGIMEN" in final.columns
        # ):
        #     final["ARM_REGIMEN"] = final.apply(
        #         lambda x: (
        #             f"{x['ARM_DOSE']} {x['ARM_DOSE_UNIT']} {x['REGIMEN']}"
        #         ),
        #         axis=1
        #     )
        # TODO: not needed right now, removing

        # Lower case Mean and Median in DV_STAT
        if "DV_STAT" in final.columns:
            final["DV_STAT"] = (
                final["DV_STAT"]
                .astype(str)
                .str.replace(
                    r"(\w+\s+)?Mean(\s+\w+)*",
                    lambda m: m.group().replace("Mean", "mean"),
                    regex=True,
                )
            )
            final["DV_STAT"] = (
                final["DV_STAT"]
                .astype(str)
                .str.replace(
                    r"(\w+\s+)?Median(\s+\w+)*",
                    lambda m: m.group().replace("Median", "median"),
                    regex=True,
                )
            )
            final["DV_STAT"] = (
                final["DV_STAT"]
                .astype(str)
                .str.replace(
                    r"(\w+\s+)?Percentage(\s+\w+)*",
                    lambda m: m.group().replace("Percentage", "percentage"),
                    regex=True,
                )
            )

        if "ROUTE" in final.columns:
            final["ROUTE"] = final["ROUTE"].astype(str).str.replace("Oral", "oral")

        # Drop temporary TRIAL_NUMBER column
        if "TRIAL_NUMBER" in final.columns:
            final = final.drop(columns=["TRIAL_NUMBER"])

        # Rename final_group_column column to GROUP_NAME
        if final_group_column and final_group_column in final.columns:
            final = final.rename(columns={final_group_column: "GROUP_NAME"})

        # Final column ordering
        final_column_order = [
            "FILE_NAME",
            "TITLE",
            "TRIAL_ID",
            "TRIAL_END_YEAR",
            "DOI",
            "BLINDING",
            "NEW_ONSET_T1DM",
            "ENDPOINT_DURATION_MONTHS",
            "SOC_DESCRIPTION",
            "STUDY_DESIGN",
            "PUBMED_ID",
            "JOURNAL_NAME",
            "YEAR",
            "COUNTRY",
            "SPONSOR_TYPE",
            "GROUP_NAME",
            "SOURCE",
            "ARM_TRT",
            "ARM_TRT_CLASS",
            "ARM_DOSE",
            "ARM_DOSE_UNIT",
            "AMT",
            "AMT_UNIT",
            "REGIMEN",
            "N_STUDY",
            "ROUTE",
            "ARM_TIME",
            "ARM_TIME_UNIT",
            "DVID",
            "ENDPOINT",
            "DV_STAT",
            "DV",
            "DV_UNIT",
            "DV_VAR",
            "DV_VAR_STAT",
            "DV_UCI",
            "DV_LCI",
            "EVID",
            "MDV",
            "ADDL",
            "II",
            "II_UNIT",
            "lineName",
            "ARM_ID",
            "N_ARM_TRIAL",
            "CPEP_TYPE",
            "CPEP_TEST",
            "IS_CONTROL",
            "N_RANDOMIZED",
            "N_EVALUABLE",
            "N_DROPOUTS",
            "RANDOMIZATION_RATIO",
            "POPULATION_NOTES",
            "PLOT_COMMENTS",
            "ARM_DUR",
            "ARM_DUR_UNIT",
            "STD_TRT",
            "AGE",
            "AGE_UNIT",
            "AGE_STAT",
            "AGE_VAR",
            "AGE_VAR_STAT",
            "BMI",
            "BMI_STAT",
            "BMI_VAR",
            "BMI_VAR_STAT",
            "CPEP_BASELINE_STD_MEAN",
            "CPEP_BASELINE_STD_MEAN_UNIT",
            "HBA1C_BASELINE_MEAN_PCT",
            "HBA1C_BASELINE_MEAN_PCT_UNIT",
            "HBA1C_VAR",
            "HBA1C_VAR_UNIT",
            "HBA1C_VAR_STAT",
            "SEX_PCT_MALE",
            "SEX_PCT_FEMALE",
            "DISEASE_DURATION_DAYS",
            "NOTES",
        ]

        # Build ordered columns: first columns from the order list that exist,
        # then extras
        ordered_columns = [col for col in final_column_order if col in final.columns]
        extra_columns = [col for col in final.columns if col not in final_column_order]
        final = final[ordered_columns + extra_columns]

        return final, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error when final formatting the merged data",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck all data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


def apply_final_column_order(final):
    """
    Apply the final column ordering to the merged dataframe.
    This should be called at the very end of the merge pipeline.
    """
    error_log = []
    try:
        final_column_order = [
            "FILE_NAME",
            "TITLE",
            "TRIAL_ID",
            "TRIAL_END_YEAR",
            "DOI",
            "BLINDING",
            "NEW_ONSET_T1DM",
            "ENDPOINT_DURATION_MONTHS",
            "SOC_DESCRIPTION",
            "STUDY_DESIGN",
            "PUBMED_ID",
            "JOURNAL_NAME",
            "YEAR",
            "COUNTRY",
            "SPONSOR_TYPE",
            "GROUP_NAME",
            "SOURCE",
            "ARM_TRT",
            "ARM_TRT_CLASS",
            "ARM_DOSE",
            "ARM_DOSE_UNIT",
            "AMT",
            "AMT_UNIT",
            "REGIMEN",
            "N_STUDY",
            "ROUTE",
            "ARM_TIME",
            "ARM_TIME_UNIT",
            "DVID",
            "ENDPOINT",
            "DV_STAT",
            "DV",
            "DV_UNIT",
            "DV_VAR",
            "DV_VAR_STAT",
            "DV_UCI",
            "DV_LCI",
            "EVID",
            "MDV",
            "ADDL",
            "II",
            "II_UNIT",
            "lineName",
            "ARM_ID",
            "N_ARM_TRIAL",
            "CPEP_TYPE",
            "CPEP_TEST",
            "IS_CONTROL",
            "N_RANDOMIZED",
            "N_EVALUABLE",
            "N_DROPOUTS",
            "RANDOMIZATION_RATIO",
            "POPULATION_NOTES",
            "PLOT_COMMENTS",
            "ARM_DUR",
            "ARM_DUR_UNIT",
            "STD_TRT",
            "AGE",
            "AGE_UNIT",
            "AGE_STAT",
            "AGE_VAR",
            "AGE_VAR_STAT",
            "BMI",
            "BMI_STAT",
            "BMI_VAR",
            "BMI_VAR_STAT",
            "CPEP_BASELINE_STD_MEAN",
            "CPEP_BASELINE_STD_MEAN_UNIT",
            "HBA1C_BASELINE_MEAN_PCT",
            "HBA1C_BASELINE_MEAN_PCT_UNIT",
            "HBA1C_VAR",
            "HBA1C_VAR_UNIT",
            "HBA1C_VAR_STAT",
            "SEX_PCT_MALE",
            "SEX_PCT_FEMALE",
            "DISEASE_DURATION_DAYS",
            "NOTES",
        ]

        # Build ordered columns: first columns from the order list that exist,
        # then extras
        ordered_columns = [col for col in final_column_order if col in final.columns]
        extra_columns = [col for col in final.columns if col not in final_column_order]
        final = final[ordered_columns + extra_columns]

        return final, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error applying final column order",
                "error_message": (
                    f"Error: {str(e)}.\n" "Kindly recheck all data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


def add_non_group_dataframes(final, non_group_dataframe):
    error_log = []
    try:
        cols_to_keep = [
            col for col in non_group_dataframe.columns if col not in final.columns
        ]
        if "FILE_NAME" in non_group_dataframe.columns:
            if "FILE_NAME" not in cols_to_keep:
                cols_to_keep += ["FILE_NAME"]
            merged_df = final.merge(
                non_group_dataframe[cols_to_keep], on="FILE_NAME", how="left"
            )
        elif "fileName" in non_group_dataframe.columns:
            if "fileName" not in cols_to_keep:
                cols_to_keep += ["fileName"]
            merged_df = final.merge(
                non_group_dataframe[cols_to_keep], on="fileName", how="left"
            )
        elif "file_name" in non_group_dataframe.columns:
            if "file_name" not in cols_to_keep:
                cols_to_keep += ["file_name"]
            merged_df = final.merge(
                non_group_dataframe[cols_to_keep], on="file_name", how="left"
            )
        else:
            error_log.append(
                {
                    "error_name": "FILE_NAME column not found in data",
                    "error_message": "Kindly add the FILE_NAME column to the data.",
                }
            )
            logger.info(f"traceback: {traceback.format_exc()}")
            return None, error_log

        # Use unique column selection to avoid multiplying duplicate columns
        # When original_cols contains duplicates (e.g., ['DV_VAR', 'DV_VAR']),
        # selecting with merged_df[original_cols] would multiply them exponentially
        original_cols_unique = list(dict.fromkeys(final.columns.tolist()))
        new_cols = [col for col in merged_df.columns if col not in original_cols_unique]

        # Select columns using iloc to preserve column order without duplication issues
        final_col_indices = [
            merged_df.columns.get_loc(c)
            for c in original_cols_unique
            if c in merged_df.columns
        ]

        # Handle case where get_loc returns different types for duplicate columns:
        # - int: unique column
        # - slice: contiguous duplicate columns
        # - boolean/integer array: non-contiguous duplicate columns
        # Take first occurrence in all cases
        def _get_first_col_index(idx):
            if isinstance(idx, int):
                return idx
            elif isinstance(idx, slice):
                return idx.start
            elif isinstance(idx, np.ndarray):
                if idx.dtype == bool:
                    return np.where(idx)[0][0]
                return idx[0]
            # Fallback for other iterables
            return next(iter(idx))

        final_col_indices = [_get_first_col_index(idx) for idx in final_col_indices]
        new_col_indices = [merged_df.columns.get_loc(c) for c in new_cols]
        new_col_indices = [_get_first_col_index(idx) for idx in new_col_indices]

        final = merged_df.iloc[:, final_col_indices + new_col_indices]
        return final, error_log
    except Exception as e:
        error_log.append(
            {
                "error_name": "Error adding data that has no group column.",
                "error_message": (
                    f"Error: {str(e)}.\n"
                    "Kindly recheck those data and rerun the merge."
                ),
            }
        )
        logger.info(f"traceback: {traceback.format_exc()}")
        return None, error_log


def run_whole_merge(tables_df, table_structure_dictionary):
    """
    Executes the complete data merging pipeline for clinical trial data.

    Args:
        tables_df (dict): Dictionary containing DataFrames for observation,
        dosing and covariate data
        table_structure_dictionary (dict): Dictionary containing the table structure
        error_log (list): List of error logs

    Returns:
        pd.DataFrame: Final merged and formatted DataFrame containing all trial data
        list: List of error logs

    The function performs the following steps:
    1. Loads and separates observation data
    2. Merges observation data from multiple sources
    3. Preprocesses dosing and observation tables
    4. Combines observation and dosing data
    5. Standardizes units and creates temporary columns
    6. Performs QC checks and merges covariate data
    7. Applies final formatting
    """
    error_log = []
    # Load and separate observation data
    (
        dataframe_dictionary,
        dose,
        function_error_log,
        table_structure_dictionary,
    ) = load_observation_data(
        tables_df,
        table_structure_dictionary,
    )
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if dataframe_dictionary is None:
            return None, error_log

    # Process observation data
    observation, function_error_log = merge_observation_data(dataframe_dictionary)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if observation is None:
            return None, error_log

    observation, function_error_log = observation_table_preprocessing(observation)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if observation is None:
            return None, error_log

    # Process dosing data
    if dose is not None:
        dose, function_error_log = dosing_table_preprocessing(dose)
        if len(function_error_log) > 0:
            error_log.extend(function_error_log)
            if dose is None:
                return None, error_log

        # Combine and process merged data
        merged_data, function_error_log = combine_observation_and_dosing(
            observation, dose
        )
        if len(function_error_log) > 0:
            error_log.extend(function_error_log)
            if merged_data is None:
                return None, error_log
    else:
        merged_data = observation

    merged_data, function_error_log = unit_standardization(merged_data)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if merged_data is None:
            return None, error_log

    # merged_data, function_error_log = temp_col_creation(merged_data)
    # if len(function_error_log) > 0:
    #     error_log.extend(function_error_log)
    #     if merged_data is None:
    #         return None, error_log

    merged_data, function_error_log = reorder_dataframe_columns(merged_data)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if merged_data is None:
            return None, error_log

    merged_data, function_error_log = dosing_rows_qc(merged_data)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if merged_data is None:
            return None, error_log

    # Final processing steps
    if TableNames.COVARIATE.value in dataframe_dictionary:
        final_data, function_error_log = covariate_merge(
            dataframe_dictionary,
            merged_data,
            table_structure_dictionary[TableNames.COVARIATE.value],
        )
        if len(function_error_log) > 0:
            error_log.extend(function_error_log)
            if final_data is None:
                return None, error_log
    else:
        final_data = merged_data

    final_data, function_error_log = final_formatting(final_data)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if final_data is None:
            return None, error_log

    for table_name, dataframe in tables_df.items():
        if table_name not in [
            TableNames.COVARIATE.value,
            TableNames.OBSERVATION.value,
            TableNames.DOSING.value,
        ]:
            final_data, function_error_log = add_non_group_dataframes(
                final_data, dataframe
            )
            if len(function_error_log) > 0:
                error_log.extend(function_error_log)
                if final_data is None:
                    return None, error_log

    # manually remove lloq column
    if "LLOQ" in final_data.columns:
        final_data = final_data.drop(columns=["LLOQ"])

    # Drop 'fileName' column
    if "fileName" in final_data.columns:
        final_data = final_data.drop(columns=["fileName"])

    # Drop all columns that should be removed from the final output
    # This ensures columns are removed regardless of which table they came from
    final_data = final_data.drop(columns=FINAL_COLUMNS_TO_DROP, errors="ignore")

    # Apply final column ordering at the very end
    final_data, function_error_log = apply_final_column_order(final_data)
    if len(function_error_log) > 0:
        error_log.extend(function_error_log)
        if final_data is None:
            return None, error_log

    return final_data, error_log
