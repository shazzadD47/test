import re
from copy import deepcopy
from typing import Any

import json_repair
import numpy as np
import pandas as pd

from app.utils import check_if_null
from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    check_if_numerical_label,
)
from app.v3.endpoints.general_extraction.services.helpers.task_helpers import (
    roll_root_answers,
)


def is_table_empty(table_structure: list[dict[str, Any]]) -> bool:
    """
    Check if all labels in the table structure have empty answers.
    Empty means: answers is null, empty list, or the answers key doesn't exist.
    """
    for label in table_structure:
        answers = label.get("answers")
        if answers is None:
            continue
        if isinstance(answers, list) and len(answers) == 0:
            continue
        if not check_if_null(answers):
            return False
    return True


def create_empty_table_row(
    table_structure: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Create a single-row empty table with N/A for
    string values and null for numeric values.
    """
    empty_table = {}
    for label in table_structure:
        label_name = label["name"]
        if check_if_numerical_label(label):
            empty_table[label_name] = [None]
        else:
            empty_table[label_name] = ["N/A"]
    return empty_table


def create_final_table(
    table_structure: list[dict[str, Any]],
    has_roots: bool,
    inputs: dict[str, Any] | None = None,
):
    # Check if the table is empty (all labels have null/empty answers)
    if is_table_empty(table_structure):
        empty_table = create_empty_table_row(table_structure)
        empty_table = apply_final_table_formatting(empty_table, table_structure)
        return empty_table, empty_table

    final_table = {}
    table_structure_name_hash = {label["name"]: label for label in table_structure}
    if has_roots:
        for label in table_structure:
            label["original_answers"] = deepcopy(label["answers"])

        root_labels = [
            label["name"] for label in table_structure if label["c_type"] == "root"
        ]

        if len(root_labels) == 1:
            root_label = root_labels[0]
            total_rows = len(table_structure_name_hash[root_label]["answers"])

            #  then assign answers to the non-root labels
            for label in table_structure_name_hash:
                label_detail = table_structure_name_hash[label]
                answers = label_detail["answers"]
                if len(answers) < total_rows:
                    null_answers = ["N/A" for _ in range(total_rows - len(answers))]
                    answers = answers + null_answers
                elif len(answers) > total_rows:
                    answers = answers[:total_rows]

                label_detail["answers"] = answers

                final_table[label] = answers
        else:
            for label in table_structure_name_hash:
                label_detail = table_structure_name_hash[label]
                final_table[label] = label_detail["answers"]

    else:
        for label in table_structure_name_hash:
            label_detail = table_structure_name_hash[label]
            final_table[label] = [label_detail["answers"]]

    final_table = convert_array_labels_to_string(final_table)
    final_table = assign_citation_to_numeric_labels(final_table, table_structure)

    if has_roots:
        root_labels = [
            label["name"] for label in table_structure if label["c_type"] == "root"
        ]
        if len(root_labels) > 1:
            rolled_root_answers = roll_root_answers(
                {
                    label: answers
                    for label, answers in final_table.items()
                    if label in root_labels
                }
            )
            for label in root_labels:
                final_table[label] = [row[label] for row in rolled_root_answers]

            # check lenght consistency
            all_label_lengths = [len(answers) for answers in final_table.values()]
            if len(set(all_label_lengths)) > 1:
                max_length = max(all_label_lengths)
                for label, answers in final_table.items():
                    if len(answers) < max_length:
                        final_table[label] = answers + ["N/A"] * (
                            max_length - len(answers)
                        )
                    elif len(answers) > max_length:
                        final_table[label] = answers[:max_length]

    final_table = fill_missing_values_with_standard_na(final_table, table_structure)

    # Replace media file citations with bounding box and page number
    # from figure metadata
    final_table = replace_media_citations_with_bbox(final_table, inputs)

    final_table_with_citations = deepcopy(final_table)
    final_table_with_citations = apply_final_table_formatting(
        final_table_with_citations, table_structure
    )

    final_table = remove_citation_from_labels(final_table)
    final_table = apply_final_table_formatting(final_table, table_structure)

    return final_table, final_table_with_citations


def fill_missing_values_with_standard_na(
    final_table: dict[str, list[Any]],
    table_structure: list[dict[str, Any]],
):
    table_length = 0
    for label in table_structure:
        label_name = label["name"]
        if label_name in final_table:
            answers = final_table[label_name]
            if label["d_type"] == "string":
                answers = [a if not check_if_null(a) else "N/A" for a in answers]
                final_table[label_name] = answers
                if table_length == 0:
                    table_length = len(answers)
            elif check_if_numerical_label(label):
                answers = [a if not check_if_null(a) else None for a in answers]
                final_table[label_name] = answers
                if table_length == 0:
                    table_length = len(answers)

    for label in table_structure:
        label_name = label["name"]
        if label_name not in final_table:
            if label["d_type"] == "string":
                final_table[label_name] = ["N/A"] * table_length
            elif check_if_numerical_label(label):
                final_table[label_name] = [None] * table_length

    return final_table


def build_media_identifier_metadata_map(
    inputs: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """
    Build a mapping from media file identifiers to their metadata (url).

    Args:
        inputs: The workflow inputs containing the figure metadata.

    Returns:
        A dictionary mapping media identifiers to their metadata.
    """
    identifier_map = {}

    if inputs is None or "inputs" not in inputs or inputs["inputs"] is None:
        return identifier_map

    for item in inputs["inputs"]:
        if item.get("type") in ["image", "chart", "table", "equation"]:
            data = item.get("data")
            if data and isinstance(data, list):
                for item_data in data:
                    identifier = item_data.get("identifier")
                    if identifier:
                        metadata = {
                            "url": item_data.get("figure_url"),
                        }
                        identifier_map[identifier] = metadata

    return identifier_map


def replace_media_citations_with_bbox(
    final_table: dict[str, list[Any]],
    inputs: dict[str, Any] | None,
) -> dict[str, list[Any]]:
    """
    Replace citations that have media file identifiers with the image URL
    from the figure metadata.

    For each citation with a <content> tag matching a media file identifier,
    replace <content> with the image URL. Page number is kept unchanged.

    Args:
        final_table: The final table with citations.
        inputs: The workflow inputs containing figure metadata.

    Returns:
        The final table with updated citations.
    """
    if inputs is None:
        return final_table

    identifier_map = build_media_identifier_metadata_map(inputs)

    if not identifier_map:
        return final_table

    # Pattern to match citation tags and extract flag_id, page_no, content
    citation_pattern = re.compile(
        r"<citation>"
        r"<flag_id>([^<]+)</flag_id>"
        r"<page_no>([^<]*)</page_no>"
        r"<content>([^<]*)</content>"
        r"</citation>"
    )

    for label, answers in final_table.items():
        updated_answers = []
        for answer in answers:
            if answer and isinstance(answer, str):
                # Find all citations in the answer
                def replace_citation(match):
                    flag_id = match.group(1)
                    original_page_no = match.group(2)
                    original_content = match.group(3)

                    # Check if content (the identifier) matches a media identifier
                    if original_content in identifier_map:
                        metadata = identifier_map[original_content]
                        new_content = metadata.get("url")

                        # Use URL if available, otherwise keep original content
                        content = (
                            f"<url>{new_content}</url>"
                            if new_content
                            else original_content
                        )

                        return (
                            f"<citation>"
                            f"<flag_id>{flag_id}</flag_id>"
                            f"<page_no>{original_page_no}</page_no>"
                            f"<content>{content}</content>"
                            f"</citation>"
                        )
                    else:
                        # Return original citation unchanged
                        return match.group(0)

                answer = citation_pattern.sub(replace_citation, answer)

            updated_answers.append(answer)
        final_table[label] = updated_answers

    return final_table


def convert_array_labels_to_string(final_table: dict[str, list[Any]]):
    for label in final_table:
        updated_answers = []
        for answer in final_table[label]:
            if isinstance(answer, list):
                updated_answers.append(convert_array_to_string(answer))
            elif isinstance(answer, np.ndarray):
                updated_answers.append(convert_array_to_string(answer.tolist()))
            else:
                updated_answers.append(answer)
        final_table[label] = updated_answers
    return final_table


def convert_array_to_string(array: list[Any]) -> str:
    return ", ".join(str(a) for a in array).strip()


def assign_citation_to_numeric_labels(
    final_table: dict[str, list[Any]],
    table_structure: list[dict[str, Any]],
):
    for label in table_structure:
        label_name = label["name"]
        if label_name in final_table and "citations" in label:
            citations = label["citations"]
            formatted_citations = []
            for each_row_citations in citations:
                each_row_citations_tagged = ""
                if isinstance(each_row_citations, list):
                    for citation in each_row_citations:
                        if isinstance(citation, dict):
                            valid_citation_tag = ""
                            if (
                                "flag_id" in citation
                                and "page_no" in citation
                                and "content" in citation
                            ):
                                valid_citation_tag += (
                                    f"<flag_id>{citation['flag_id']}</flag_id>"
                                )
                                valid_citation_tag += (
                                    f"<page_no>{citation['page_no']}</page_no>"
                                )
                                valid_citation_tag += (
                                    f"<content>{citation['content']}</content>"
                                )
                                each_row_citations_tagged += (
                                    f"<citation>{valid_citation_tag}</citation>"
                                )
                            else:
                                continue

                formatted_citations.append(each_row_citations_tagged)

            if len(formatted_citations) == len(final_table[label_name]):
                final_table[label_name] = [
                    f"{answer} {citation}"
                    for answer, citation in zip(
                        final_table[label_name], formatted_citations
                    )
                ]
            else:
                # if numer of citations is greater, strip to the length of answers
                # otherwise, add empty string to the end of the answers
                if len(formatted_citations) > len(final_table[label_name]):
                    formatted_citations = formatted_citations[
                        : len(final_table[label_name])
                    ]
                else:
                    formatted_citations = formatted_citations + [""] * (
                        len(final_table[label_name]) - len(formatted_citations)
                    )

                final_table[label_name] = [
                    f"{answer} {citation}"
                    for answer, citation in zip(
                        final_table[label_name], formatted_citations
                    )
                ]

    return final_table


def remove_citation_tags(text: str) -> str:
    """
    Removes everything inside <citation></citation> tags
    along with the tags themselves.

    Args:
        text (str): The input text that may contain citation tags.

    Returns:
        str: The text with citation tags and their content removed.
    """
    # Pattern to match <citation>...</citation> tags and their content
    citation_pattern = r"<citation>.*?</citation>"

    # Remove all citation tags and their content
    cleaned_text = re.sub(citation_pattern, "", text, flags=re.DOTALL).strip()

    return cleaned_text


def remove_citation_from_labels(
    final_table: dict[str, list[Any]],
):
    updated_final_table = {}
    for label, answers in final_table.items():
        updated_answers = []
        for answer in answers:
            if answer and isinstance(answer, str):
                answer = remove_citation_tags(answer)
                updated_answers.append(answer)
            else:
                updated_answers.append(answer)
        updated_final_table[label] = updated_answers
    return updated_final_table


def apply_final_table_formatting(
    final_table: dict[str, list[Any]],
    table_structure: list[dict[str, Any]],
):
    final_table = pd.DataFrame(final_table)
    column_order = [
        label["name"]
        for label in table_structure
        if label["name"] in final_table.columns
    ]
    final_table = final_table[column_order]
    final_table = final_table.to_json(orient="records")
    final_table = json_repair.loads(final_table)
    return final_table
