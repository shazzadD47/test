import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.v3.endpoints.merging.logging import logger

# * QC Checks


def check_arm_sums(df):
    """
    Simple check of N_ARM sums vs N_STUDY across all timepoints.
    Only prints when discrepancies are found.

    Parameters:
    -----------
    df : pandas DataFrame
        Must contain FILE_NAME, ARM_TIME, GROUP_NAME, N_ARM, and N_STUDY columns
    """
    error_files = []

    # Loop through each unique file in the dataset
    for file_name in df["FILE_NAME"].unique():
        # Get all data for this specific file
        file_data = df[df["FILE_NAME"] == file_name]

        # Get all unique timepoints and sort them
        timepoints = sorted(file_data["ARM_TIME"].unique())

        # Check each timepoint separately
        for timepoint in timepoints:
            # Get data for this specific timepoint
            time_data = file_data[file_data["ARM_TIME"] == timepoint]

            # Get first occurrence of each GROUP_NAME to avoid double counting
            # This handles cases where same group might appear multiple times
            unique_groups = time_data.groupby("GROUP_NAME").first()

            # Calculate total participants by summing N_ARM across groups
            total_n_arm = unique_groups["N_ARM"].sum()
            # Get the total study participants
            # (N_STUDY should be same for all rows)
            n_study = time_data["N_STUDY"].iloc[0]

            # Check if there's a discrepancy (using small
            # tolerance for float comparison)
            if abs(total_n_arm - n_study) > 0.01:
                # Add file to error list
                error_files.append(file_name)

                # Print detailed error information
                logger.debug(f"\nFile: {file_name}")
                logger.debug(f"Timepoint {timepoint}")

                # Print N_ARM for each group
                logger.debug("\nGroups:")
                for group in unique_groups.index:
                    n_arm = unique_groups.loc[group, "N_ARM"]
                    source = unique_groups.loc[group, "SOURCE"]
                    logger.debug(f"GROUP: {group}: N_ARM {n_arm} : FIGURE: {source}")

                # Print summary statistics
                logger.debug(f"\nTotal N_ARM: {total_n_arm}")
                logger.debug(f"N_STUDY: {n_study}")
                logger.debug(f"Difference: {total_n_arm - n_study}")

                # Show how many times each group appears in the data
                # This helps identify potential duplicate entries
                logger.debug("\nRows per group:")
                group_counts = time_data["GROUP_NAME"].value_counts()

                for group, count in group_counts.items():
                    if count > 1:  # Only show groups that appear multiple times
                        logger.debug(f"  {group}: {count} rows")
                logger.debug("-" * 50)

    return error_files


def check_arm_sums2(df):
    """
    Simple check of N_ARM sums vs N_STUDY across all timepoints.
    Instead of printing errors directly, this function captures all output
    in a variable and returns it along with the list of files with discrepancies.

    Parameters:
    -----------
    df : pandas DataFrame
        Must contain FILE_NAME, ARM_TIME, GROUP_NAME, N_ARM, and N_STUDY columns

    Returns:
    --------
    error_files : list
        List of file names where discrepancies were found.
    output : str
        A single string containing all of the output messages.
    """

    error_files = []
    output_messages = []  # List to capture all output

    # Loop through each unique file in the dataset
    for file_name in df["FILE_NAME"].unique():
        # Get all data for this specific file
        file_data = df[df["FILE_NAME"] == file_name]

        # Filter for rows where EVID is 0 (observations)
        file_data = file_data[file_data["EVID"] == 0]

        # Get all unique timepoints and sort them
        timepoints = sorted(file_data["ARM_TIME"].unique())

        # Check each timepoint separately
        for timepoint in timepoints:
            try:
                # Get data for this specific timepoint
                time_data = file_data[file_data["ARM_TIME"] == timepoint]

                # Get first occurrence of each GROUP_NAME to avoid double counting
                unique_groups = time_data.groupby("GROUP_NAME").first()

                # Calculate total participants by summing N_ARM across groups
                total_n_arm = unique_groups["N_ARM"].sum()
                # Get the total study participants (N_STUDY should be same for all rows)
                n_study = time_data["N_STUDY"].iloc[0]

                # Check if there's a discrepancy
                # (using a small tolerance for float comparison)
                if abs(total_n_arm - n_study) > 0.01:
                    # Add file to error list
                    error_files.append(file_name)

                    # Capture detailed error information in our messages list
                    # output_messages.append(f"\nFile: {file_name}")

                    # Capture N_ARM for each group
                    output_messages.append(f"\nTimepoint {timepoint}")
                    # output_messages.append("Groups:")

                    for group in unique_groups.index:
                        n_arm = unique_groups.loc[group, "N_ARM"]
                        source = unique_groups.loc[group, "SOURCE"]
                        output_messages.append(
                            f"GROUP: {group}: N_ARM {n_arm} : FIGURE: {source}"
                        )

                    # Capture summary statistics
                    output_messages.append(
                        f"Total N_ARM: {total_n_arm} | N_STUDY: {n_study}"
                        f" | Difference: {total_n_arm - n_study}"
                    )

                    # Show how many times each group appears in the data
                    # (for duplicate checking)

                    group_counts = time_data["GROUP_NAME"].value_counts()
                    for group, count in group_counts.items():
                        # Only show groups that appear multiple times
                        if count > 1:
                            output_messages.append("\nRows per group:")
                            output_messages.append(f"  {group}: {count} rows")
                    # output_messages.append("-" * 50)
            except Exception as e:
                output_messages.append(f"Error occured checking time point: {e}")

    # Combine all messages into a single string
    output = "\n".join(output_messages)
    return output


def check_timepoint_equality(df):
    """
    Check if all rows have equal values across columns.
    Returns True if all rows are equal, False otherwise.

    Parameters:
    df (pandas.DataFrame): Input DataFrame to check

    Returns:
    bool: True if all rows have equal values, False otherwise
    """
    try:
        output_messages = []

        # Filter for rows where EVID is 0 (observations)
        df = df[df["EVID"] == 0]

        for filename, file_group in df.groupby("FILE_NAME"):
            for plot, plot_group in file_group.groupby("SOURCE"):
                line_list = []
                for _line, line_group in plot_group.groupby("GROUP_NAME"):
                    line_list.append(line_group["ARM_TIME"].reset_index(drop=True))
                lines_df = pd.concat(line_list, axis=1)
                # row_equality_check = check_row_equality_detailed(lines_df)
                # Check equality across all columns and check if all rows are True

                row_equality_check = (
                    lines_df.eq(lines_df.iloc[:, 0], axis=0).all(axis=1).all()
                )

                if not row_equality_check:
                    output_messages.append("ROW INEQUALITY FOUND")
                    output_messages.append(filename)
                    output_messages.append(plot)
                    try:
                        output_messages.append(
                            lines_df.to_string(index=False, header=True)
                        )
                    except Exception as e:
                        output_messages.append(f"No data to display: {e}")
                    output_messages.append(filename)

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking timepoint equality failed",
            f"Error: {e}",
            "Double check these columns: ARM_TIME, GROUP_NAME",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def analyze_ndose_by_file_and_group(drug_file_df):
    try:
        output_messages = []

        max_n_dose = drug_file_df

        # First group by FILE_NAME, SOURCE, and GROUP_NAME and get the maximum N_DOSE
        group_max_doses = max_n_dose.groupby(["FILE_NAME", "SOURCE", "GROUP_NAME"])[
            "N_DOSE"
        ].max()

        # Now check consistency across SOURCE groups
        for file_name in drug_file_df["FILE_NAME"].unique():
            file_data = group_max_doses[file_name]

            # Group by SOURCE and get all max N_DOSE values
            source_groups = file_data.groupby("SOURCE")

            for source, source_data in source_groups:
                max_doses = source_data.unique()

                if len(max_doses) > 1:
                    output_messages.append(f"File: {file_name}")
                    output_messages.append(f"Source: {source}")
                    output_messages.append(f"Max doses are not matching! {max_doses}")

        final_output = "\n".join(output_messages)

        return final_output
    except Exception as e:
        output_messages = [
            "Checking N_DOSE consistency failed",
            f"Error: {e}",
            "Double check these columns: N_DOSE, SOURCE, GROUP_NAME",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def check_regimen_consistency(drug_file_df):
    try:
        output_messages = []
        for file_name, file_df in drug_file_df.groupby("FILE_NAME"):
            for group_name, group_df in file_df.groupby("GROUP_NAME"):
                arm_dur_unit = group_df["ARM_DUR_UNIT"].iloc[0]
                regimen = group_df["REGIMEN"].iloc[0]
                arm_dur = group_df["ARM_DUR"].iloc[0]
                max_n_dose = group_df["N_DOSE"].max()

                if arm_dur_unit == "weeks":
                    arm_dur = arm_dur * 7
                elif arm_dur_unit == "hours":
                    arm_dur = arm_dur / 24

                if regimen.lower() == "qw":
                    N_dose = (arm_dur / 7) * 1
                elif regimen.lower() == "qd":
                    N_dose = arm_dur * 1
                elif regimen.lower() == "bid":
                    N_dose = arm_dur * 2
                elif regimen.lower() == "tid":
                    N_dose = arm_dur * 3
                elif regimen.lower() == "single dose":
                    N_dose = 1
                else:
                    output_messages.append(f"ERROR Regimen {regimen} is not standard")
                    N_dose = "ERROR"

                if max_n_dose != N_dose:
                    output_messages.append(f"File: {file_name}")
                    output_messages.append(f"Group: {group_name}")
                    output_messages.append(
                        f"ERROR max_n_dose {max_n_dose} doesn't match regime {regimen},"
                    )

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking regimen consistency failed",
            f"Error: {e}",
            "Double check these columns: ARM_DUR_UNIT, REGIMEN, ARM_DUR, N_DOSE",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def check_gender_percentages(drug_file_df):
    """
    Check that male and female percentages sum to 100 for each file.

    Parameters:
    -----------
    df : pandas DataFrame
        Must contain FILE_NAME, MALE_PERCENT_BL, and FEMALE_PERCENT_BL columns
    """
    try:
        output_messages = []

        # For each unique file
        for file_name in drug_file_df["FILE_NAME"].unique():
            file_data = drug_file_df[drug_file_df["FILE_NAME"] == file_name]
            # Take first row since percentages should be same within file
            male_pct = file_data["MALE_PERCENT_BL"].iloc[0]
            female_pct = file_data["FEMALE_PERCENT_BL"].iloc[0]
            total_pct = male_pct + female_pct
            total_pct = float(total_pct)

            # Check if sum equals 100 (using small tolerance for float comparison)
            if abs(total_pct - 100) > 0.01:
                output_messages.append(f"\nFile: {file_name}")
                output_messages.append(f"Male %: {male_pct}")
                output_messages.append(f"Female %: {female_pct}")
                output_messages.append(f"Total %: {total_pct}")
                output_messages.append(f"Difference from 100%: {total_pct - 100}")

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking gender percentages failed",
            f"Error: {e}",
            "Double check these columns: MALE_PERCENT_BL, FEMALE_PERCENT_BL",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def covariate_check(drug_file_df):
    """
    Make sure that any covariates with two numbers has range or IQR as VAR_STAT
    """
    try:
        output_messages = []

        def find_ranges_with_locations(df):
            val_columns = [col for col in df.columns if "_BL" in col]
            # pattern = r'\d+\.?\d*-\d+\.?\d*'
            pattern = r"\d+\.?\d*\s*-\s*\d+\.?\d*"

            # Dictionary to store results with row indices
            detailed_ranges = {}

            for column in val_columns:
                column_findings = []
                for idx, value in df[column].items():
                    if isinstance(value, str):
                        matches = re.findall(pattern, str(value))
                        if matches:
                            for match in matches:
                                column_findings.append(
                                    {
                                        "range": match,
                                        "row_index": idx,
                                        "full_value": value,
                                    }
                                )

                if column_findings:
                    detailed_ranges[column] = column_findings

            return detailed_ranges

        result = drug_file_df
        # Example usage:
        detailed_result = find_ranges_with_locations(drug_file_df)
        for column, findings in detailed_result.items():
            for finding in findings:
                column_name = column + "_STAT"
                row_index = finding["row_index"]
                if (
                    result[column_name].iloc[row_index] != "range"
                    and result[column_name].iloc[row_index] != "IQR"
                ):
                    filename = result["FILE_NAME"].iloc[row_index]
                    output_messages.append(f"\nFile: {filename}")
                    output_messages.append(f"Column: {column}")

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking covariate check failed",
            f"Error: {e}",
            "Double check all columns with _BL in the name",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def check_missing_covariates(drug_file_df):
    """
    Check if any covariates are missing
    """
    output_messages = []
    result = drug_file_df

    def find_missing_cov_with_locations(df):
        val_columns = ["AGE_BL", "BW_BL", "BMI_BL"]

        # Dictionary to store results with row indices
        detailed_ranges = {}

        for column in val_columns:
            column_findings = []
            for idx, value in df[column].items():
                try:
                    if np.isnan(value):
                        column_findings.append(
                            {
                                "paper": df["FILE_NAME"].iloc[idx],
                            }
                        )
                except Exception:
                    if pd.isna(value):
                        column_findings.append(
                            {
                                "paper": df["FILE_NAME"].iloc[idx],
                            }
                        )
            if column_findings:
                detailed_ranges[column] = column_findings

        return detailed_ranges

    missing_cov = find_missing_cov_with_locations(result)
    missing_cov.keys()

    try:
        age_bl_missing_fin = {d["paper"] for d in missing_cov["AGE_BL"]}
        if age_bl_missing_fin:
            output_messages.append(f"Missing AGE_BL in papers: {age_bl_missing_fin}")
    except (KeyError, TypeError):
        pass
    try:
        bw_bl_missing_fin = {d["paper"] for d in missing_cov["BW_BL"]}
        if bw_bl_missing_fin:
            output_messages.append(f"Missing BW_BL in papers: {bw_bl_missing_fin}")
    except (KeyError, TypeError):
        pass
    try:
        bmi_bl_missing_fin = {d["paper"] for d in missing_cov["BMI_BL"]}
        if bmi_bl_missing_fin:
            output_messages.append(f"Missing BMI_BL in papers: {bmi_bl_missing_fin}")
    except (KeyError, TypeError):
        pass

    final_output = "\n".join(output_messages)
    return final_output


def check_if_covariates_biologically_plausible(drug_file_df):
    """
    Check if covariates are biologically plausible
    """
    try:
        output_messages = []
        result = drug_file_df

        def define_biological_ranges():
            """Define dictionary of biologically plausible ranges for covariates"""
            return {
                "AGE_BL": (18, 90),
                "AGE_BL_VAR": (0, 30),
                "BMI_BL": (15, 60),
                "BMI_BL_VAR": (0, 15),
                "BW_BL": (35, 200),
                "BW_BL_VAR": (0, 50),
                "HT_BL": (140, 210),
                "HT_BL_VAR": (0, 30),
                "WAIST_CIRC_BL": (60, 200),
                "WAIST_CIRC_BL_VAR": (0, 40),
                "HBA1C_BL": (4, 11),
                "HBA1C_BL_VAR": (0, 3),
                "T2D_DUR_BL": (0, 30),
                "T2D_DUR_BL_VAR": (0, 15),
                "EGFR_BL": (30, 120),
                "EGFR_BL_VAR": (0, 40),
                "FPG_BL": (3, 20),
                "FPG_BL_VAR": (0, 5),
                "SYSTOLIC_BP_BL": (90, 200),
                "SYSTOLIC_BP_BL_VAR": (0, 30),
                "DIASTOLIC_BP_BL": (50, 120),
                "DIASTOLIC_BP_BL_VAR": (0, 20),
            }

        def check_row_ranges(row):
            """
            Check if values in a row are within biological ranges

            Args:
                row: pandas Series representing one row of data

            Returns:
                list of tuples containing (column name, value,
                expected range) for violations
            """
            ranges = define_biological_ranges()
            violations = []

            for column, (min_val, max_val) in ranges.items():
                if column in row.index:
                    value = row[column]
                    # Check if value is numeric and not null
                    try:
                        if (
                            pd.notnull(value)
                            and isinstance(value, (int, float))
                            and (value < min_val or value > max_val)
                        ):
                            violations.append((column, value, f"{min_val}-{max_val}"))
                    except Exception:
                        violations.append(
                            (
                                f"Failed to check {column} within {min_val}-{max_val}",
                                f"{column}: {value}",
                                f"{min_val}-{max_val}",
                            )
                        )

            # Check percentage sums
            try:
                if all(
                    x in row.index for x in ["MALE_PERCENT_BL", "FEMALE_PERCENT_BL"]
                ):
                    gender_sum = row["MALE_PERCENT_BL"] + row["FEMALE_PERCENT_BL"]
                    if (
                        abs(gender_sum - 100) > 0.1
                    ):  # Allow for small rounding differences
                        violations.append(("Gender percentages", gender_sum, "100"))
            except Exception:
                violations.append(
                    "Could not check if total gender percentage is 100. "
                    "Double check the following columns: "
                    "MALE_PERCENT_BL, FEMALE_PERCENT_BL"
                )
            # Check blood pressure relationship
            try:
                if all(
                    x in row.index for x in ["SYSTOLIC_BP_BL", "DIASTOLIC_BP_BL"]
                ) and (row["SYSTOLIC_BP_BL"] <= row["DIASTOLIC_BP_BL"]):
                    violations.append(
                        (
                            "BP relationship",
                            f"SYSTOLIC_BP_BL: {row['SYSTOLIC_BP_BL']}, "
                            f"DIASTOLIC_BP_BL: {row['DIASTOLIC_BP_BL']}, "
                            "Systolic should be > Diastolic, "
                            "Double check these columns: "
                            "SYSTOLIC_BP_BL, DIASTOLIC_BP_BL",
                        )
                    )
            except Exception:
                violations.append(
                    "Failed to check if SYSTOLIC_BP_BL is > DIASTOLIC_BP_BL, "
                    f"SYSTOLIC_BP_BL: {row['SYSTOLIC_BP_BL']}, "
                    f"DIASTOLIC_BP_BL: {row['DIASTOLIC_BP_BL']}, "
                    "Double check these columns: SYSTOLIC_BP_BL, DIASTOLIC_BP_BL"
                )

            return violations

        for filename in result["FILE_NAME"].unique():
            filedata = result[result["FILE_NAME"] == filename]
            for group in filedata["GROUP_NAME"].unique():
                firstrow = filedata[filedata["GROUP_NAME"] == group].iloc[0]
                violations = check_row_ranges(firstrow)
                if violations:
                    output_messages.append(
                        f"\nViolations in file {filename} for group {group}:"
                    )
                    for violation in violations:
                        if isinstance(violation, tuple):
                            if len(violation) == 3:
                                output_messages.append(
                                    f"{violation[0]}: {violation[1]} "
                                    f"(Expected range: {violation[2]})"
                                )
                            elif len(violation) == 1:
                                output_messages.append(violation[0])
                        else:
                            output_messages.append(violation)

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking if covariates are biologically plausible failed",
            f"Error: {e}",
            "Double check all these columns:",
            "AGE_BL, AGE_BL_VAR, BMI_BL, BMI_BL_VAR, BW_BL, BW_BL_VAR, HT_BL,",
            "HT_BL_VAR, WAIST_CIRC_BL, WAIST_CIRC_BL_VAR, HBA1C_BL, HBA1C_BL_VAR,",
            "T2D_DUR_BL, T2D_DUR_BL_VAR, EGFR_BL, EGFR_BL_VAR, FPG_BL, FPG_BL_VAR,",
            "SYSTOLIC_BP_BL, SYSTOLIC_BP_BL_VAR, DIASTOLIC_BP_BL, DIASTOLIC_BP_BL_VAR",
            "MALE_PERCENT_BL, FEMALE_PERCENT_BL",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# check if each DVID only has one source (one figure)
def check_num_sources(df):
    try:
        output_messages = []

        for file_name in df["FILE_NAME"].unique():
            file_data = df[df["FILE_NAME"] == file_name]
            for DVID in file_data["DVID"].unique():
                mask = file_data["DVID"] == DVID
                source_column = file_data[mask]["SOURCE"]
                if len(source_column.unique()) > 1:

                    output_messages.append(f"\nFile: {file_name}")
                    output_messages.append(f"\nDVID: {DVID}")
                    output_messages.append(f"\nFigures: {source_column.unique()}")
                    output_messages.append("-" * 50)

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking number of sources failed",
            f"Error: {e}",
            "Double check these columns: FILE_NAME, DVID, SOURCE",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def check_DV(df):
    try:
        violations = {}

        for file_name in df["FILE_NAME"].unique():
            file_data = df[df["FILE_NAME"] == file_name]

            # Iterate through rows using itertuples() for better performance
            for row in file_data.itertuples():
                # Create unique key for file/source combination
                unique_key = (file_name, row.SOURCE)

                # Skip if we already found a violation for this file/source
                if unique_key in violations:
                    continue
                # Weight in kg
                if row.DVID == 4 and row.DV_UNIT == "kg":
                    if not 50 <= row.DV <= 140:
                        violations[unique_key] = {
                            "file_name": file_name,
                            "Figure": row.SOURCE,
                            "DVID": row.DVID,
                            "DV_UNIT": row.DV_UNIT,
                            "DV": row.DV,
                            "expected_range": "50-140 kg",
                        }
                elif row.DVID == 4 and row.DV_UNIT == "lb":
                    if not 100 <= row.DV <= 300:
                        violations[unique_key] = {
                            "file_name": file_name,
                            "Figure": row.SOURCE,
                            "DVID": row.DVID,
                            "DV_UNIT": row.DV_UNIT,
                            "DV": row.DV,
                            "expected_range": "100-300 lb",
                        }
                elif row.DVID == 8:
                    if not -30 <= row.DV <= 10:
                        violations[unique_key] = {
                            "file_name": file_name,
                            "Figure": row.SOURCE,
                            "DVID": row.DVID,
                            "DV_UNIT": row.DV_UNIT,
                            "DV": row.DV,
                            "expected_range": "expected range -30-10 ",
                        }
                elif (
                    row.DVID == 3
                    and row.DV_UNIT == "percentage"
                    and not 4 <= row.DV <= 10
                ):
                    violations[unique_key] = {
                        "file_name": file_name,
                        "Figure": row.SOURCE,
                        "DVID": row.DVID,
                        "DV_UNIT": row.DV_UNIT,
                        "DV": row.DV,
                        "expected_range": "expected range 4-10 ",
                    }
                elif (
                    row.DVID == 3
                    and row.DV_UNIT == "mmol/mol"
                    and not 30 <= row.DV <= 75
                ):
                    violations[unique_key] = {
                        "file_name": file_name,
                        "Figure": row.SOURCE,
                        "DVID": row.DVID,
                        "DV_UNIT": row.DV_UNIT,
                        "DV": row.DV,
                        "expected_range": "expected range 30-75",
                    }
                elif row.DVID == 10 and not -10 <= row.DV <= 2:
                    violations[unique_key] = {
                        "file_name": file_name,
                        "Figure": row.SOURCE,
                        "DVID": row.DVID,
                        "DV_UNIT": row.DV_UNIT,
                        "DV": row.DV,
                        "expected_range": "expected range -10-2",
                    }
        return str(violations)
    except Exception as e:
        output_messages = [
            "Checking DV failed",
            f"Error: {e}",
            "Double check these columns: DVID, DV_UNIT, DV",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# Checks that all values of ROUTE are either SC or Oral, as expected
def check_route(df):
    try:
        output_messages = []
        acceptable_routes = {"SC", "oral"}
        mask = ~df["ROUTE"].isin(acceptable_routes) & df["ROUTE"].notna()

        if mask.any():
            invalid_routes = (
                df[mask].groupby(["FILE_NAME", "ROUTE"]).size().reset_index()
            )
            output_messages.append("\nFound invalid routes:")
            for _, row in invalid_routes.iterrows():
                output_messages.append(f"File: {row['FILE_NAME']}")
                output_messages.append(
                    f"Invalid route found: '{row['ROUTE']}' (Expected: SC or oral)\n"
                )

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking route failed",
            f"Error: {e}",
            "Double check these columns: ROUTE",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# Checks that all values of DV_STAT, as expected
def check_dv_stat(df):
    try:
        output_messages = []
        acceptable_dvstat = {
            "mean",
            "median",
            "geometric mean",
            "mean CFB",
            "LS mean",
            "LS mean CFB",
            "percentage",
        }
        mask = ~df["DV_STAT"].isin(acceptable_dvstat) & df["DV_STAT"].notna()

        if mask.any():
            invalid_stat = (
                df[mask]
                .groupby(["FILE_NAME", "DV_STAT", "SOURCE"])
                .size()
                .reset_index()
            )
            output_messages.append("\nFound invalid stat:")
            for _, row in invalid_stat.iterrows():
                output_messages.append(f"File: {row['FILE_NAME']}")
                output_messages.append(f"Figure: {row['SOURCE']}")
                output_messages.append(
                    f"Invalid stat found: '{row['DV_STAT']}' "
                    f"(Expected: mean, mean CFB, median, geometric mean, "
                    f"LS mean, LS mean CFB, percentage)\n"
                )

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking DV_STAT failed",
            f"Error: {e}",
            "Double check these columns: DV_STAT",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# Checks that all values of DV_UNIT are in a list of expected strings
def check_dv_unit(df):
    try:
        output_messages = []
        acceptable_unit = {
            "percentage",
            "kg",
            "nmol/l",
            "mg/dl",
            "ng/ml",
            "mmol/mol",
            "mmol/l",
        }
        mask = ~df["DV_UNIT"].isin(acceptable_unit) & df["DV_STAT"].notna()

        if mask.any():
            invalid_unit = (
                df[mask]
                .groupby(["FILE_NAME", "DV_UNIT", "SOURCE"])
                .size()
                .reset_index()
            )
            output_messages.append("\nFound invalid unit:")
            for _, row in invalid_unit.iterrows():
                output_messages.append(f"File: {row['FILE_NAME']}")
                output_messages.append(f"Figure: {row['SOURCE']}")
                output_messages.append(f"Invalid unit found: '{row['DV_UNIT']}\n")

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking DV_UNIT failed",
            f"Error: {e}",
            "Double check these columns: DV_UNIT",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# Checks that all values of DV_UNIT are in a list of expected strings
def check_dv_var_stat(df):
    try:
        output_messages = []
        acceptable_var = {"SEM", "SE", "95% CI", "IQR", "SD"}
        mask = ~df["DV_VAR_STAT"].isin(acceptable_var) & df["DV_VAR_STAT"].notna()

        if mask.any():
            invalid_var = (
                df[mask]
                .groupby(["FILE_NAME", "DV_VAR_STAT", "SOURCE"])
                .size()
                .reset_index()
            )
            output_messages.append("\nFound invalid var stat:")
            for _, row in invalid_var.iterrows():
                output_messages.append(f"File: {row['FILE_NAME']}")
                output_messages.append(f"Figure: {row['SOURCE']}")
                output_messages.append(
                    f"Invalid VAR STAT found: '{row['DV_VAR_STAT']}\n"
                )

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking DV_VAR_STAT failed",
            f"Error: {e}",
            "Double check these columns: DV_VAR_STAT",
        ]
        final_output = "\n".join(output_messages)
        return final_output


def check_overall_decrease(df):
    """
    Check that DV values decrease from first to last time
    point for each unique combination
    of FILE_NAME, SOURCE, and GROUP_NAME. Only check
    GLP-1 treatment class and DVID=4 or 8
    Args:
        df: pandas DataFrame with columns FILE_NAME, SOURCE,
        GROUP_NAME, ARM_TIME, DV, ARM_TRT_CLASS, DVID
    Returns:
        list: Violations where final value is higher than initial value
    """
    try:
        # Filter for GLP-1 and DVID=4 or 8
        mask = (df["ARM_TRT_CLASS"] == "GLP-1") & (df["DVID"].isin([4, 8]))
        filtered_df = df[mask]
        violations = []
        # Group by the unique identifiers
        for file_name, source, group in (
            filtered_df[["FILE_NAME", "SOURCE", "GROUP_NAME"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            # Get data for this combination
            group_data = filtered_df[
                (filtered_df["FILE_NAME"] == file_name)
                & (filtered_df["SOURCE"] == source)
                & (filtered_df["GROUP_NAME"] == group)
            ].sort_values("ARM_TIME")
            # Get first and last non-null values
            first_row = group_data[group_data["DV"].notna()].iloc[0]
            last_row = group_data[group_data["DV"].notna()].iloc[-1]
            # Only check if times are different
            if (
                first_row["ARM_TIME"] != last_row["ARM_TIME"]
                and last_row["DV"] >= first_row["DV"]
            ):
                violations.append(
                    {
                        "file_name": file_name,
                        "source": source,
                        "group": group,
                        "dvid": first_row["DVID"],  # Added DVID to output
                        "first_time": first_row["ARM_TIME"],
                        "first_value": first_row["DV"],
                        "last_time": last_row["ARM_TIME"],
                        "last_value": last_row["DV"],
                    }
                )
        # Print violations in a readable format
        if violations:
            logger.debug(
                f"\nFound {len(violations)} groups where final value did not decrease:"
            )
            for v in violations:
                logger.debug(f"\nFile: {v['file_name']}")
                logger.debug(f"Source: {v['source']}")
                logger.debug(f"Group: {v['group']}")
                logger.debug(f"DVID: {v['dvid']}")
                logger.debug(
                    f"Initial value: {v['first_value']} at time {v['first_time']}"
                )
                logger.debug(f"Final value: {v['last_value']} at time {v['last_time']}")

        return str(violations)
    except Exception as e:
        output_messages = [
            "Checking overall decrease failed",
            f"Error: {e}",
            "Double check these columns: ARM_TIME, DV",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# Group by the unique identifiers
def check_unique_arm_numbers(df):
    try:
        output_messages = []
        for file_name, group in (
            df[["FILE_NAME", "GROUP_NAME"]].drop_duplicates().itertuples(index=False)
        ):
            # Get data for this combination
            group_data = df[
                (df["FILE_NAME"] == file_name) & (df["GROUP_NAME"] == group)
            ]
            if len(group_data["ARM_NUMBER"].unique()) > 1:
                output_messages.append(f"\nFile: {file_name}")
                output_messages.append(f"More than one ARM_NUMBER for group: {group}")
                for arm_number in group_data["ARM_NUMBER"].unique():
                    error_data = df[
                        (df["FILE_NAME"] == file_name)
                        & (df["GROUP_NAME"] == group)
                        & (df["ARM_NUMBER"] == arm_number)
                    ]
                    source = error_data["SOURCE"].unique()
                    output_messages.append(f"Figure: {source}")

        final_output = "\n".join(output_messages)
        return final_output
    except Exception as e:
        output_messages = [
            "Checking unique arm numbers failed",
            f"Error: {e}",
            "Double check these columns: ARM_NUMBER",
        ]
        final_output = "\n".join(output_messages)
        return final_output


# * Utilities
def get_excel_files(folder_path):
    """
    Get all Excel files in the specified folder (not including subfolders)

    Args:
        folder_path (str or Path): Path to the folder to search

    Returns:
        pd.DataFrame: DataFrame containing paths of Excel files with columns:
            - file_path: Full path to the Excel file
            - file_name: Name of the file without the path
    """

    # Convert to Path object if string
    folder_path = Path(folder_path)

    # Find all Excel files (both .xlsx and .xls)
    excel_files = []
    excel_files.extend(folder_path.glob("*.xlsx"))
    excel_files.extend(folder_path.glob("*.xls"))

    # Convert to list of strings and sort
    excel_files = [str(f) for f in excel_files if f.is_file()]
    excel_files.sort()

    # Create DataFrame
    df = pd.DataFrame(
        {"file_path": excel_files, "file_name": [Path(f).stem for f in excel_files]}
    )

    return df


# * MAIN QC CHECK
def perform_individual_file_qc(drug_file_df):
    """
    Perform QC check on a single Excel file
    """
    try:
        # Read the consolidated file
        # drug_file_df = pd.read_excel(drug_file_df)

        # region QC Checks
        # check row equality
        timepoint_equality_output = check_timepoint_equality(drug_file_df)

        # Run the check and get list of files with errors
        check_arm_sums_output = check_arm_sums2(drug_file_df)

        # Check Max N_DOSE
        check_max_ndose_output = analyze_ndose_by_file_and_group(drug_file_df)

        # Check regimen consistency with dosing table
        check_regimen_consistency_output = check_regimen_consistency(drug_file_df)

        # Check gender percentages
        check_gender_percentages_output = check_gender_percentages(drug_file_df)

        # Check covariate check
        covariate_check_range_output = covariate_check(drug_file_df)

        # Check missing covariates
        check_missing_covariates_output = check_missing_covariates(drug_file_df)

        # Check if covariates are biologically plausible
        check_if_covariates_biologically_plausible_output = (
            check_if_covariates_biologically_plausible(drug_file_df)
        )

        # Check number of sources
        check_num_sources_output = check_num_sources(drug_file_df)

        # Check DV
        check_DV_output = check_DV(drug_file_df)

        # Check route
        check_route_output = check_route(drug_file_df)

        # Check DV_STAT
        check_dv_stat_output = check_dv_stat(drug_file_df)

        # Check DV_UNIT
        check_dv_unit_output = check_dv_unit(drug_file_df)

        # Check DV_VAR_STAT
        check_dv_var_stat_output = check_dv_var_stat(drug_file_df)

        # Check overall decrease
        check_overall_decrease_output = check_overall_decrease(drug_file_df)

        # Check unique arm numbers
        check_unique_arm_numbers_output = check_unique_arm_numbers(drug_file_df)
        # endregion

        error_return = {
            "check_arm_sums_output": check_arm_sums_output,
            "check_max_ndose_output": check_max_ndose_output,
            "check_regimen_consistency_output": check_regimen_consistency_output,
            "check_gender_percentages_output": check_gender_percentages_output,
            "timepoint_equality_output": timepoint_equality_output,
            "covariate_check_range_output": covariate_check_range_output,
            "check_missing_covariates_output": check_missing_covariates_output,
            "check_if_covariates_biologically_plausible_output": (
                check_if_covariates_biologically_plausible_output
            ),
            "check_num_sources_output": check_num_sources_output,
            "check_DV_output": check_DV_output,
            "check_route_output": check_route_output,
            "check_dv_stat_output": check_dv_stat_output,
            "check_dv_unit_output": check_dv_unit_output,
            "check_dv_var_stat_output": check_dv_var_stat_output,
            "check_overall_decrease_output": check_overall_decrease_output,
            "check_unique_arm_numbers_output": check_unique_arm_numbers_output,
        }

        return error_return
    except Exception as e:
        error_message = "Error performing individual file QC."
        error_message += f" Error message: {str(e)}"
        raise Exception(error_message)
