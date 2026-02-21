import json
import re
from itertools import cycle

import json_repair
from celery.utils.log import get_task_logger
from pydantic import BaseModel, Field, create_model
from sqlalchemy import Table, select

from app.constants import d_type_map
from app.core.database.base import engine, get_db_session, metadata
from app.core.database.models import MinerUResponses
from app.utils.utils import check_if_null
from app.v3.endpoints.iterative_autofill.configs import settings as iaf_settings
from app.v3.endpoints.iterative_autofill.constants import (
    MAX_RETRIES,
    SEPARATOR,
)

logger = get_task_logger("iterative_autofill")


def find_suggested_loc_from_query(
    query: str,
):
    rephrased_question, suggested_location = query.split(
        "Suggestion regarding where to find:"
    )
    return rephrased_question, suggested_location


def combine_contexts(
    questions_with_contexts: dict,
    field_names: list[str],
) -> str:
    combined_contexts = []
    count = 1
    for field_name in field_names:
        if field_name in questions_with_contexts:
            question_with_context = f"""
            {count}.
            {field_name}: {questions_with_contexts[field_name]['contexts']}
            """
            question_with_context = re.sub(r"\s+", " ", question_with_context).strip()

            combined_contexts.append(question_with_context)
            count += 1

    return "\n".join(combined_contexts)


def merge_responses(
    prev_response: list[dict] = None,
    new_response: list[dict] = None,
) -> list[dict]:
    final_response = []
    if prev_response is None or len(prev_response) == 0:
        return new_response

    if new_response is None or len(new_response) == 0:
        return prev_response

    if len(prev_response) > len(new_response):
        for prev, new in zip(prev_response, cycle(new_response)):
            updated_label_answers = prev.copy()
            updated_label_answers.update(new)
            final_response.append(updated_label_answers)
    else:
        for prev, new in zip(cycle(prev_response), new_response):
            updated_label_answers = prev.copy()
            updated_label_answers.update(new)
            final_response.append(updated_label_answers)

    return final_response


def get_relation_info(
    relationships: list[dict],
    response: list[dict],
    table_structure_hash: dict[str, dict],
    original_label: bool = True,
) -> str:
    if original_label:
        relation_info = "Following are its parent labels and their values: "
    else:
        relation_info = ""
    if response is not None and len(response) > 0:
        for count, relationship in enumerate(relationships):
            related_label_answers = []
            for label_answer in response:
                for key, value in label_answer.items():
                    if key == relationship["related_label"]:
                        related_label_answers.append(str(value))

            if len(related_label_answers) > 0:
                field_name = relationship["related_label"]
                field_description = table_structure_hash[field_name]["description"]
                relationship_desc = relationship["description"]
                relation_info += f"\n {count+1}. Field: {field_name} \n"
                relation_info += f"Description: {field_description} \n"
                relation_info += f"Answers: {', '.join(related_label_answers)} \n"
                relation_info += f"Related How: {relationship_desc} \n"
                if (
                    table_structure_hash.get(relationship["related_label"], {}).get(
                        "relationships", {}
                    )
                    is not None
                    and len(
                        table_structure_hash.get(relationship["related_label"], {}).get(
                            "relationships", {}
                        )
                    )
                    > 0
                ):
                    related_labels_of_related_label = [
                        relationship["related_label"]
                        for relationship in table_structure_hash[
                            relationship["related_label"]
                        ]["relationships"]
                    ]
                    relation_info += "Related Labels:"
                    relation_info += f" {', '.join(related_labels_of_related_label)} \n"
                    relation_info = (
                        get_relation_info(
                            table_structure_hash[relationship["related_label"]][
                                "relationships"
                            ],
                            response,
                            table_structure_hash,
                            original_label=False,
                        )
                        + relation_info
                    )
                relation_info += "\n"
    return relation_info


def find_next_labels_to_extract(
    all_extracted_labels: list[str],
    prev_extracted_labels: list[str],
    to_generate_labels: list[str],
    table_structure: list[dict],
) -> list[str]:
    """
    Add the labels that are dependent on all previously extracted labels.
    Condition:
    1. All labels that the label is dependent on should be extracted
    """
    related_labels = []
    for field in table_structure:
        if (
            field["name"] not in all_extracted_labels
            and field["name"] not in prev_extracted_labels
            and field["name"] in to_generate_labels
            and "relationships" in field
            and field["relationships"] is not None
            and len(field["relationships"]) > 0
            and all(
                relationship["related_label"] in all_extracted_labels
                and relationship["related_label"] in to_generate_labels
                for relationship in field["relationships"]
            )
        ):
            related_labels.append(field["name"])

    return list(set(related_labels))


def reverse_relationships(
    table_structure: list[dict],
) -> list[dict]:
    modified_table_structure = {
        field["name"]: field.copy() for field in table_structure
    }
    for _k, v in modified_table_structure.items():
        v["relationships"] = None

    for field in table_structure:
        if (
            "relationships" in field
            and field["relationships"] is not None
            and len(field["relationships"]) > 0
        ):
            for relationship in field["relationships"]:
                if relationship["related_label"] is not None:
                    if (
                        relationship["related_label"] in modified_table_structure
                        and modified_table_structure[relationship["related_label"]][
                            "relationships"
                        ]
                        is None
                    ):
                        modified_table_structure[relationship["related_label"]][
                            "relationships"
                        ] = []
                    modified_table_structure[relationship["related_label"]][
                        "relationships"
                    ].append(
                        {
                            "related_label": field["name"],
                            "description": relationship["description"],
                        }
                    )

    # remove duplicate relations
    for label in modified_table_structure:
        if (
            "relationships" in modified_table_structure[label]
            and modified_table_structure[label]["relationships"] is not None
            and len(modified_table_structure[label]["relationships"]) > 0
        ):
            unique_related_label_names = []
            unique_relationships = []
            for relationship in modified_table_structure[label]["relationships"]:
                if relationship["related_label"] not in unique_related_label_names:
                    unique_related_label_names.append(relationship["related_label"])
                    unique_relationships.append(relationship)

            modified_table_structure[label]["relationships"] = unique_relationships

    return list(modified_table_structure.values())


def get_parent_labels(
    table_structure_hash: dict[str, dict],
    labels: list[str],
    already_added_labels: list[str],
    relationship_table: dict[str, list[str]],
    recursion_stage: int = 0,
) -> (list[str], dict[str, list[str]]):
    parent_labels = []
    if recursion_stage > iaf_settings.LOOP_LIMIT:
        for label in labels:
            relationship_table[label] = []
        return parent_labels, relationship_table

    for label in labels:
        if label in relationship_table:
            parent_labels.extend(relationship_table[label])
            parent_labels = list(set(parent_labels))
            if len(parent_labels) == len(table_structure_hash):
                return parent_labels, relationship_table
            continue
        elif (
            table_structure_hash.get(label, {}).get("relationships", []) is None
            or len(table_structure_hash.get(label, {}).get("relationships", [])) == 0
        ):
            relationship_table[label] = []
            continue
        elif (
            table_structure_hash.get(label, {}).get("relationships", []) is not None
            and len(table_structure_hash.get(label, {}).get("relationships", [])) > 0
        ):
            parent_labels_of_relationships = []

            for relationship in table_structure_hash[label]["relationships"]:
                if relationship["related_label"] not in already_added_labels:
                    parent_labels_of_relationships.append(relationship["related_label"])
                    if relationship["related_label"] not in relationship_table:
                        relationship_parent_labels, relationship_table = (
                            get_parent_labels(
                                table_structure_hash,
                                [relationship["related_label"]],
                                parent_labels + [relationship["related_label"]],
                                relationship_table,
                                recursion_stage=recursion_stage + 1,
                            )
                        )
                        parent_labels_of_relationships.extend(
                            relationship_parent_labels
                        )
                    else:
                        parent_labels_of_relationships.extend(
                            relationship_table[relationship["related_label"]]
                        )

            parent_labels_of_relationships = list(set(parent_labels_of_relationships))
            relationship_table[label] = parent_labels_of_relationships
            parent_labels.extend(parent_labels_of_relationships)
            parent_labels = list(set(parent_labels))
            if len(parent_labels) == len(table_structure_hash):
                return parent_labels, relationship_table
    return parent_labels, relationship_table


def check_all_labels_extracted(
    extracted_labels: list[str],
    to_generate_labels: list[str],
) -> bool:
    return len(set(to_generate_labels) - set(extracted_labels)) == 0


def fill_missing_labels(
    final_response: list[dict],
    table_structure: list[dict],
) -> list[dict]:
    updated_final_response = []
    for response in final_response:
        updated_response = response.copy()
        for field in table_structure:
            if field["name"] not in updated_response:
                if field["d_type"] == "string":
                    updated_response[field["name"]] = "N/A"
                else:
                    updated_response[field["name"]] = None
            else:
                if check_if_null(updated_response[field["name"]]):
                    if field["d_type"] == "string":
                        updated_response[field["name"]] = "N/A"
                    else:
                        updated_response[field["name"]] = None
        updated_final_response.append(updated_response)
    return updated_final_response


def prepare_final_response(
    final_response: list[dict],
    table_structure: list[dict],
) -> list[dict]:
    if final_response is None:
        final_response = []
    if isinstance(final_response, dict):
        final_response = [final_response]
    final_response = fill_missing_labels(final_response, table_structure)
    final_response = json_repair.loads(json.dumps({"data": final_response}))
    final_response = final_response["data"]
    labels_schema = {
        field["name"]: (
            d_type_map[field["d_type"]] | None,
            Field(..., description=field["description"]),
        )
        for field in table_structure
    }
    Labels = create_model("Labels", **labels_schema)

    class LabelsSchema(BaseModel):
        data: list[Labels]

    final_response = LabelsSchema(**{"data": final_response}).dict()
    return final_response


def extract_all_paper_texts(flag_id: str):
    retry_count = 0
    from_documents_success = True
    while retry_count < MAX_RETRIES:
        try:
            if from_documents_success:
                documents_table = Table("documents", metadata, autoload_with=engine)
                with get_db_session() as session:
                    query = select(
                        documents_table.c.content, documents_table.c.metadata
                    ).where(documents_table.c.metadata["flag_id"].astext == flag_id)
                    result = session.execute(query).all()

                if result is None or len(result) == 0:
                    from_documents_success = False
                    logger.info(f"Retrieval documents status: {from_documents_success}")
                else:
                    result, total_chunks = format_paper_chunks_from_documents(result)
                    if result is not None:
                        logger.info(
                            f"Retrieval documents status: {from_documents_success}"
                        )
                        return result, total_chunks
                    else:
                        from_documents_success = False
                        logger.info(
                            f"Retrieval documents status: {from_documents_success}"
                        )

            # if not found, get from mineru chunks
            query = select(MinerUResponses).where(
                MinerUResponses.flag_id == flag_id,
                MinerUResponses.response_type == "final",
            )
            with get_db_session() as session:
                chunks = session.execute(query).scalars().all()

            if chunks is None or len(chunks) == 0:
                return None, None
            chunks = chunks[0].response["chunks"]
            chunks, total_chunks = format_paper_chunks(chunks)
            return chunks, total_chunks
        except Exception as e:
            logger.exception(f"Error getting mineru chunks: {e}")
            retry_count += 1
            if retry_count == MAX_RETRIES:
                return None, None
    return None, None


def format_paper_chunks(chunks: list[dict]):
    formatted_chunk_texts = [""] * len(chunks)
    for i, chunk in enumerate(chunks):
        if chunk.get("type") == "table":
            formatted_chunk_texts[i] = {
                "paper_section": chunk.get("metadata", {}).get("heading", ""),
                "chunk_type": chunk.get("type", ""),
                "text": (
                    chunk.get("value", "")
                    + f" Table Data: {chunk.get('metadata',{}).get('table_body',{})}"
                    + f" Caption: {chunk.get('metadata',{}).get('table_caption',{})}"
                    + f" Footnote: {chunk.get('metadata',{}).get('table_footnote',{})}"
                ),
            }
        else:
            formatted_chunk_texts[i] = {
                "paper_section": chunk.get("metadata", {}).get("heading", ""),
                "chunk_type": chunk.get("type", ""),
                "text": chunk.get("value", ""),
            }
    formatted_chunk_texts = {i: formatted_chunk_texts[i] for i in range(len(chunks))}
    # sort by chunk_id in chunks [metadata]['chunk_id']
    formatted_chunk_texts = dict(
        sorted(
            formatted_chunk_texts.items(),
            key=lambda x: chunks[x[0]]["metadata"]["chunk_id"],
        )
    )
    # remove empty chunks
    formatted_chunk_texts = {
        i: chunk
        for i, chunk in formatted_chunk_texts.items()
        if chunk["text"].strip() != ""
        and chunk["paper_section"].strip()
        not in [
            "Reference",
            "References",
            "References and Notes",
            "References and Footnotes",
        ]
        and chunk["text"].strip() != chunk["paper_section"].strip()
    }
    for i in formatted_chunk_texts:
        formatted_chunk_texts[i] = json.dumps(
            formatted_chunk_texts[i],
            ensure_ascii=False,
        )
    formatted_chunk_texts = [
        formatted_chunk_texts[i]
        for i in range(len(formatted_chunk_texts))
        if i in formatted_chunk_texts
    ]
    formatted_chunk_texts = [
        f"{i+1}. {formatted_chunk_texts[i]}" for i in range(len(formatted_chunk_texts))
    ]
    total_chunks = len(formatted_chunk_texts)
    formatted_chunk_texts = SEPARATOR.join(formatted_chunk_texts)
    return formatted_chunk_texts, total_chunks


def format_paper_chunks_from_documents(result):
    formatted_chunk_texts = [""] * len(result)

    # first determine if chunk_id is present in the metdata
    chunk_id_present = False
    for item in result:
        if isinstance(item[1], dict) and "chunk_id" in item[1]:
            chunk_id_present = True
            break
        elif isinstance(item[1], str):
            dict_chunk = json_repair.loads(item[1])
            if isinstance(dict_chunk, dict) and "chunk_id" in dict_chunk:
                chunk_id_present = True
                break
        else:
            continue
    if not chunk_id_present:
        return None, None

    # sort the result by chunk_id in the second element of each item in result
    # chunk id may or may not be present in the result
    # if not present, keep those chunks at the end of the list
    result = sorted(result, key=lambda x: x[1].get("chunk_id", float("inf")))
    for i, item in enumerate(result):
        formatted_chunk_texts[i] = item[0]

    # remove reference and empty chunks
    updated_chunks = []
    for chunk in formatted_chunk_texts:
        dict_chunk = json_repair.loads(chunk)
        if isinstance(dict_chunk, dict):
            text = dict_chunk.get("text", "")
            section_name = dict_chunk.get("paper_section", "")
            if (
                text.strip() == ""
                or section_name.strip()
                in [
                    "Reference",
                    "References",
                    "References and Notes",
                    "References and Footnotes",
                ]
                or text.strip() == section_name.strip()
            ):
                continue
            else:
                updated_chunks.append(chunk)
        else:
            updated_chunks.append(chunk)
    updated_chunks = [f"{i+1}. {updated_chunks[i]}" for i in range(len(updated_chunks))]
    total_chunks = len(updated_chunks)
    updated_chunks = SEPARATOR.join(updated_chunks)
    return updated_chunks, total_chunks


def format_reference_text(text: str):
    if isinstance(text, str):
        first_snippet = text.split()[:5]
        last_snippet = text.split()[-5:]
        text_snippet = " ".join(first_snippet) + " ... " * 2 + " ".join(last_snippet)
        return text_snippet
    return text
