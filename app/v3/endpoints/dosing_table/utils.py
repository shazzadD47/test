from typing import get_args, get_origin

import pandas as pd

from app.core.vector_store import VectorStore, VectorStoreRetriever
from app.v3.endpoints.dosing_table.langchain_schemas import TableRow


def get_context_retriever(
    paper_id: str, project_id: str, k: int = 15
) -> VectorStoreRetriever:
    retriever = VectorStore.get_retriever(
        search_kwargs={
            "k": k,
            "filter": {"flag_id": paper_id.strip(), "project_id": project_id.strip()},
        }
    )

    return retriever


def fix_arm_time_starting_from_one(
    df: pd.DataFrame, group_column: str = "GROUP", arm_time_column: str = "ARM_TIME"
):
    per_group_arm_time_values = {}
    for group in df[group_column].unique():
        arm_times = df.loc[df[group_column] == group, arm_time_column].tolist()
        per_group_arm_time_values[group] = arm_times

    for group, group_arm_times in per_group_arm_time_values.items():
        if 1 in group_arm_times:
            group_arm_times = [v - 1 for v in group_arm_times]
            df.loc[df[group_column] == group, arm_time_column] = group_arm_times
    return df


def create_empty_table() -> list[dict]:
    """Create an empty table row with default values based on TableRow schema."""
    empty_row = {}

    for field_name, field_info in TableRow.model_fields.items():
        annotation = field_info.annotation

        # Check if the field is Optional (Union with None)
        origin = get_origin(annotation)
        args = get_args(annotation)

        # If it's a Union type (includes Optional), check if None is one of the types
        if origin is type(None) or (origin and None in args or type(None) in args):
            # For numeric types that can be None, default to None
            if any(arg in (int, float) for arg in args if arg is not type(None)):
                empty_row[field_name] = None
            else:
                empty_row[field_name] = "N/A"
        # For non-optional numeric types
        elif annotation in (int, float):
            empty_row[field_name] = None
        # For string types (including Literal which are string-based)
        else:
            empty_row[field_name] = "N/A"

    return [empty_row]
