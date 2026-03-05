import pandas as pd

from app.v3.endpoints.merging.constants import TableNames
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.schemas import QCError, merge_error
from app.v3.endpoints.merging.utils import check_if_null

logger = logger.getChild("group_qc")


def find_group_name_column(dataframe: pd.DataFrame) -> str:
    for column in dataframe.columns:
        if column.strip().lower() == "group":
            return column

    for column in dataframe.columns:
        if "group" in column.lower().strip():
            return column
    return "GROUP"


def fix_group_name_classical(group_name: str) -> str:
    if check_if_null(group_name):
        return "NA"

    temp_name = group_name.strip().lower()
    temp_name = "_".join(temp_name.split())
    if "-" in temp_name:
        temp_name = temp_name.replace("-", "_")

    return temp_name


def fix_group_names(
    dataframe_dictionary: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    for key, dataframe in dataframe_dictionary.items():
        if key == TableNames.DOSING.value:
            dosing_group_name = find_group_name_column(dataframe)
            dataframe[dosing_group_name] = dataframe[dosing_group_name].apply(
                fix_group_name_classical
            )
        else:
            other_table_group_name = find_group_name_column(dataframe)
            dataframe[other_table_group_name] = dataframe[other_table_group_name].apply(
                fix_group_name_classical
            )
        dataframe_dictionary[key] = dataframe

    return dataframe_dictionary


def run_group_qc(
    dataframe_dictionary: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[QCError]]:
    dataframe_dictionary = fix_group_names(dataframe_dictionary)

    # if group names for covariate and Dosing
    # do not match with Observation group names,
    # return the error in an error log and halt the
    # qc process.
    (
        observation_group_names,
        covariate_group_names,
        dosing_group_names,
        adverse_event_group_names,
    ) = (
        [],
        [],
        [],
        [],
    )
    if TableNames.OBSERVATION.value in dataframe_dictionary:
        obs_df = dataframe_dictionary[TableNames.OBSERVATION.value]
        group_name_column = find_group_name_column(obs_df)
        if group_name_column != "GROUP":
            obs_df.rename(columns={group_name_column: "GROUP"}, inplace=True)
        observation_group_names = list(obs_df["GROUP"].unique())
    if TableNames.COVARIATE.value in dataframe_dictionary:
        cov_df = dataframe_dictionary[TableNames.COVARIATE.value]
        group_name_column = find_group_name_column(cov_df)
        if group_name_column != "GROUP":
            cov_df.rename(columns={group_name_column: "GROUP"}, inplace=True)
        covariate_group_names = list(cov_df["GROUP"].unique())
    if TableNames.DOSING.value in dataframe_dictionary:
        dosing_df = dataframe_dictionary[TableNames.DOSING.value]
        group_name_column = find_group_name_column(dosing_df)
        if group_name_column != "GROUP":
            dosing_df.rename(columns={group_name_column: "GROUP"}, inplace=True)
        dosing_group_names = list(dosing_df["GROUP"].unique())
    error_log: list[QCError] = []

    if (
        len(observation_group_names) > 0
        and len(covariate_group_names) > 0
        and len(set(observation_group_names) - set(covariate_group_names)) > 0
    ):
        error_log.append(
            merge_error(
                "Covariate group names!=Observation group names.",
                (
                    f"Observation group names: {observation_group_names}\n"
                    f"Covariate group names: {covariate_group_names}\n"
                    f"Difference between observation and covariate group names: "
                    f"{set(observation_group_names) - set(covariate_group_names)}\n"
                    "Ensure all group names same across Observation, "
                    "Covariate and Dosing.\nand then rerun the QC."
                ),
            )
        )

    if (
        len(observation_group_names) > 0
        and len(dosing_group_names) > 0
        and len(set(observation_group_names) - set(dosing_group_names)) > 0
    ):
        error_log.append(
            merge_error(
                "Dosing group names!=Observation group names.",
                (
                    f"Observation group names: {observation_group_names}\n"
                    f"Dosing group names: {dosing_group_names}\n"
                    f"Difference between observation and dosing group names: "
                    f"{set(observation_group_names) - set(dosing_group_names)}\n"
                    "Ensure all group names same across Observation, "
                    "Covariate and Dosing.\nand then rerun the QC."
                ),
            )
        )

    if (
        len(observation_group_names) > 0
        and len(adverse_event_group_names) > 0
        and len(set(observation_group_names) - set(adverse_event_group_names)) > 0
    ):
        error_log.append(
            merge_error(
                "Adverse event group names!=Observation group names.",
                (
                    f"Observation group names: {observation_group_names}\n"
                    f"Adverse event group names: {adverse_event_group_names}\n"
                    f"Difference between observation and adverse event group names: "
                    f"{set(observation_group_names) - set(adverse_event_group_names)}\n"
                    "Ensure all group names same across Observation, "
                    "Covariate and Dosing.\nand then rerun the QC."
                ),
            )
        )

    return dataframe_dictionary, error_log
