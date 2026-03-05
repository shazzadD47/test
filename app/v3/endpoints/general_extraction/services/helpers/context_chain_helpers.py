import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.utils.files import create_file_input
from app.utils.rag import retrieve_all_contexts
from app.v3.endpoints.general_extraction.configs import settings as ge_settings
from app.v3.endpoints.general_extraction.langchain_schemas import (
    NumericalAnswer,
    RootNumericalAnswer,
)
from app.v3.endpoints.general_extraction.logging import celery_logger as logger
from app.v3.endpoints.general_extraction.prompts.commons import (
    END_OF_INPUTS_PROMPT,
    END_OF_KNOWLEDGE_FILES_PROMPT,
    END_OF_SUPPLEMENTARY_FILES_PROMPT,
    START_OF_INPUTS_PROMPT,
    START_OF_KNOWLEDGE_FILE_PROMPT,
    START_OF_KNOWLEDGE_FILES_PROMPT,
    START_OF_SUPPLEMENTARY_FILE_PROMPT,
    START_OF_SUPPLEMENTARY_FILES_PROMPT,
    SYSTEM_INSTRUCTION_FOR_CONTEXT_GENERATION,
)
from app.v3.endpoints.general_extraction.prompts.no_root import (
    LABEL_CONTEXT_GENERATION_NO_ROOT_PROMPT,
)
from app.v3.endpoints.general_extraction.prompts.numerical import (
    NUMERICAL_CONTEXT_GENERATION_PROMPT,
    NUMERICAL_ROOT_CONTEXT_GENERATION_PROMPT,
    NUMERICAL_START_OF_KNOWLEDGE_FILES_PROMPT,
    NUMERICAL_SYSTEM_INSTRUCTION,
)
from app.v3.endpoints.general_extraction.prompts.with_root import (
    LABEL_CONTEXT_GENERATION_PROMPT,
    LABEL_CONTEXT_GENERATION_ROOT_PROMPT,
)
from app.v3.endpoints.general_extraction.services.helpers.common_helpers import (
    check_if_numerical_label,
    check_if_unit_label,
    check_if_unit_label_has_numerical_non_unit_label,
    create_custom_instructions_prompt,
)
from app.v3.endpoints.general_extraction.services.helpers.information_helpers import (  # noqa: E501
    create_media_inputs,
)
from app.v3.endpoints.general_extraction.services.helpers.input_helpers import (
    format_label_details,
)


def create_chain_inputs(
    project_id: str,
    flag_id: str,
    table_structure: list[dict[str, Any]],
    selected_labels: list[dict[str, Any]] = None,
    inputs: dict[str, Any] = None,
    file_details: dict | None = None,
    has_roots: bool = False,
    numericals_extracted: bool = False,
) -> tuple[
    list[dict[str, Any]],
    dict[int, str],
    bool,
    list[BaseModel],
]:
    custom_instructions = create_custom_instructions_prompt(
        inputs.get("custom_instruction")
    )

    shared_content = _build_shared_content(
        flag_id=flag_id,
        project_id=project_id,
        inputs=inputs,
        file_details=file_details,
    )

    primary_chain_inputs = _create_primary_chain_inputs(
        flag_id=flag_id,
        project_id=project_id,
        inputs=inputs,
        file_details=file_details,
        for_numerical_labels=False,
        shared_content=shared_content,
    )
    if not numericals_extracted:
        primary_numerical_chain_inputs = _create_primary_chain_inputs(
            flag_id=flag_id,
            project_id=project_id,
            inputs=inputs,
            file_details=file_details,
            for_numerical_labels=True,
            shared_content=shared_content,
        )
    else:
        primary_numerical_chain_inputs = []

    return _create_chain_inputs_for_selected_labels(
        table_structure=table_structure,
        selected_labels=selected_labels,
        input_messages=primary_chain_inputs,
        input_numerical_messages=primary_numerical_chain_inputs,
        has_roots=has_roots,
        custom_instructions=custom_instructions,
    )


def _build_shared_content(
    flag_id: str,
    project_id: str,
    inputs: dict[str, Any] = None,
    file_details: dict | None = None,
) -> dict[str, Any]:
    """
    Pre-compute the expensive, shared content that is identical between
    regular and numerical chain inputs:
      - PDF Base64 payload (or text fallback)
      - Supplementary file Base64 payloads
      - Media / image inputs

    Returns a dict with keys:
      "content_placeholder" – the primary PDF/text content dict
      "supplementary_contents" – list of (supp_flag_id, supp_content_dict)
      "media_inputs_regular" – media inputs for regular labels
      "media_inputs_numerical" – media inputs for numerical labels
    """
    # ── Primary PDF content ──────────────────────────────────────────
    if (
        isinstance(file_details, dict)
        and "pdf_path" in file_details
        and file_details["pdf_path"] != "N/A"
    ):
        pdf_path = file_details["pdf_path"]
    else:
        pdf_path = None

    if pdf_path is not None:
        content_placeholder = create_file_input(
            pdf_path, model_name=ge_settings.CONTEXT_GENERATOR_FALLBACK_LLM
        )
    else:
        content_placeholder = {
            "type": "text",
            "text": retrieve_all_contexts(
                flag_id=flag_id, project_id=project_id, file_type="document"
            ),
        }

    # ── Supplementary files ──────────────────────────────────────────
    supplementary_contents: list[tuple[str, dict]] = []
    supplementary_paths = (
        file_details.get("supplementary_paths", [])
        if isinstance(file_details, dict)
        else []
    )
    if supplementary_paths:
        for supp_path in supplementary_paths:
            supp_filename = os.path.basename(supp_path)
            supp_flag_id = os.path.splitext(supp_filename)[0]
            supp_content = create_file_input(
                supp_path, model_name=ge_settings.CONTEXT_GENERATOR_FALLBACK_LLM
            )
            supplementary_contents.append((supp_flag_id, supp_content))

    # ── Media inputs (images / charts / tables) ──────────────────────
    # create_media_inputs mutates item_data in-place (sets image_base64)
    # so after the first call the second call is essentially free.
    media_inputs_regular = create_media_inputs(flag_id, inputs, False)
    media_inputs_numerical = create_media_inputs(flag_id, inputs, True)

    return {
        "content_placeholder": content_placeholder,
        "supplementary_contents": supplementary_contents,
        "media_inputs_regular": media_inputs_regular,
        "media_inputs_numerical": media_inputs_numerical,
    }


def _create_primary_chain_inputs(
    flag_id: str,
    project_id: str,
    inputs: dict[str, Any] = None,
    file_details: dict | None = None,
    for_numerical_labels: bool = False,
    shared_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    today = datetime.now().strftime("%Y-%m-%d %A")
    # create system instruction
    if for_numerical_labels:
        system_instruction = [
            SystemMessage(content=NUMERICAL_SYSTEM_INSTRUCTION.format(date=today))
        ]

    else:
        system_instruction = [
            SystemMessage(
                content=SYSTEM_INSTRUCTION_FOR_CONTEXT_GENERATION.format(date=today)
            )
        ]

    # ── Use pre-built shared content if available ────────────────────
    if shared_content is not None:
        content_placeholder = shared_content["content_placeholder"]
        supplementary_contents = shared_content["supplementary_contents"]
        if for_numerical_labels:
            media_inputs = shared_content["media_inputs_numerical"]
        else:
            media_inputs = shared_content["media_inputs_regular"]
    else:
        # Fallback: compute content from scratch (backwards-compatible)
        if (
            isinstance(file_details, dict)
            and "pdf_path" in file_details
            and file_details["pdf_path"] != "N/A"
        ):
            pdf_path = file_details["pdf_path"]
        else:
            pdf_path = None

        if pdf_path is not None:
            content_placeholder = create_file_input(
                pdf_path, model_name=ge_settings.CONTEXT_GENERATOR_FALLBACK_LLM
            )
        else:
            content_placeholder = {
                "type": "text",
                "text": retrieve_all_contexts(
                    flag_id=flag_id, project_id=project_id, file_type="document"
                ),
            }

        supplementary_contents = []
        supplementary_paths = (
            file_details.get("supplementary_paths", [])
            if isinstance(file_details, dict)
            else []
        )
        if supplementary_paths:
            for supp_path in supplementary_paths:
                supp_filename = os.path.basename(supp_path)
                supp_flag_id = os.path.splitext(supp_filename)[0]
                supp_content = create_file_input(
                    supp_path, model_name=ge_settings.CONTEXT_GENERATOR_FALLBACK_LLM
                )
                supplementary_contents.append((supp_flag_id, supp_content))

        media_inputs = create_media_inputs(flag_id, inputs, for_numerical_labels)

    # ── Build knowledge file inputs ──────────────────────────────────
    if for_numerical_labels:
        knowledge_file_inputs = [
            {"type": "text", "text": NUMERICAL_START_OF_KNOWLEDGE_FILES_PROMPT}
        ]
    else:
        knowledge_file_inputs = [
            {"type": "text", "text": START_OF_KNOWLEDGE_FILES_PROMPT}
        ]
    knowledge_file_inputs += [
        {
            "type": "text",
            "text": START_OF_KNOWLEDGE_FILE_PROMPT.format(index=1, flag_id=flag_id),
        },
        content_placeholder,
    ]

    # Add supplementary files if available
    if supplementary_contents:
        knowledge_file_inputs.append(
            {"type": "text", "text": START_OF_SUPPLEMENTARY_FILES_PROMPT}
        )

        for supp_index, (supp_flag_id, supp_content) in enumerate(
            supplementary_contents, start=1
        ):
            knowledge_file_inputs.append(
                {
                    "type": "text",
                    "text": START_OF_SUPPLEMENTARY_FILE_PROMPT.format(
                        index=supp_index, supplementary_flag_id=supp_flag_id
                    ),
                }
            )
            knowledge_file_inputs.append(supp_content)

        # Close the supplementary files section
        knowledge_file_inputs.append(
            {"type": "text", "text": END_OF_SUPPLEMENTARY_FILES_PROMPT}
        )

    knowledge_file_inputs.append(
        {"type": "text", "text": END_OF_KNOWLEDGE_FILES_PROMPT}
    )

    all_contents = (
        [{"type": "text", "text": START_OF_INPUTS_PROMPT}]
        + knowledge_file_inputs
        + media_inputs
        + [{"type": "text", "text": END_OF_INPUTS_PROMPT}]
    )

    return system_instruction + [HumanMessage(content=all_contents)]


def _select_context_generation_prompt(
    label_data: dict,
    has_roots: bool,
) -> str:
    if label_data["c_type"] == "root" and check_if_numerical_label(label_data):
        return NUMERICAL_ROOT_CONTEXT_GENERATION_PROMPT
    elif check_if_numerical_label(label_data):
        return NUMERICAL_CONTEXT_GENERATION_PROMPT
    else:
        if label_data["c_type"] == "root":
            return LABEL_CONTEXT_GENERATION_ROOT_PROMPT
        else:
            if has_roots:
                return LABEL_CONTEXT_GENERATION_PROMPT
            else:
                return LABEL_CONTEXT_GENERATION_NO_ROOT_PROMPT


def _create_answer_schema(
    total_questions: int,
    has_roots: bool,
    is_root: bool,
    is_number: bool = False,
    is_literal: bool = False,
    literal_options: Literal = None,
):
    if is_root and is_number:

        class AnswerList(BaseModel):
            rows: list[RootNumericalAnswer] = Field(
                ...,
                description="Each row contains a possible value of the label",
            )

        return AnswerList

    elif is_root and is_literal:

        class AnswerList(BaseModel):
            if literal_options:
                rows: list[literal_options] = Field(
                    ...,
                    description="Each row contains a possible value of the label",
                )
            else:
                rows: list[str] = Field(
                    ...,
                    description="Each row contains a possible value of the label",
                )

        return AnswerList

    elif is_root:

        class AnswerList(BaseModel):
            rows: list[str] = Field(
                ...,
                description="Each row contains a possible value of the label",
            )

        return AnswerList

    elif is_number:

        class AnswerList(BaseModel):
            answers: list[NumericalAnswer] = Field(
                ...,
                description=(
                    "List of answers for the label. Lenght of the list of answers"
                    f" must be euqal to {total_questions}."
                    f"Not less than or greaer than {total_questions}."
                    f"Exactly {total_questions} answers are required."
                ),
                min_length=total_questions,
                max_length=total_questions,
            )

        return AnswerList

    elif has_roots:
        description = (
            "List of answers for the label. Lenght of the list of answers"
            f" must be euqal to {total_questions}."
            f"Not less than or greaer than {total_questions}."
            f"Exactly {total_questions} answers are required."
        )
        if is_literal:

            class AnswerList(BaseModel):
                if literal_options:
                    answers: list[literal_options] = Field(
                        ...,
                        description=description,
                        min_length=total_questions,
                        max_length=total_questions,
                    )
                else:
                    answers: list[str] = Field(
                        ...,
                        description=description,
                        min_length=total_questions,
                        max_length=total_questions,
                    )

            return AnswerList
        else:

            class AnswerList(BaseModel):
                answers: list[str] = Field(
                    ...,
                    description=description,
                    min_length=total_questions,
                    max_length=total_questions,
                )

            return AnswerList

    return None


def _create_chain_inputs_for_selected_labels(
    table_structure: list[dict[str, Any]],
    selected_labels: list[dict[str, Any]] = None,
    input_messages: list[BaseMessage] = None,
    input_numerical_messages: list[BaseMessage] = None,
    has_roots: bool = False,
    custom_instructions: str = "",
) -> tuple[list[dict[str, Any]], dict[int, str], list[BaseModel]]:
    all_prompts, all_schemas = [], []
    prompt_to_label_map = {}
    count = 0
    batch_size = ge_settings.QUERY_BATCH_SIZE

    for label in table_structure:
        if label["name"] in selected_labels:
            # check if a label is a unit label and has a corresponding numerical
            # non unit label - if yes, skip it since its answer will be derived
            # from the NumericalAnswer schema's 'unit' field
            if check_if_unit_label(
                label["name"]
            ) and check_if_unit_label_has_numerical_non_unit_label(
                label["name"], table_structure
            ):
                continue

            if label["c_type"] == "root":
                # create label context generation prompt for root labels
                formatted_label_details = format_label_details(
                    label,
                    include_keys=["name", "description"],
                )
                label_prompt = _select_context_generation_prompt(
                    label,
                    has_roots,
                )
                label_prompt = label_prompt.format(
                    label_details=formatted_label_details,
                    special_user_instructions=custom_instructions,
                )
                all_prompts.append(label_prompt)
                prompt_to_label_map[count] = label["name"]

                # create answer schema for root labels
                # Note: total_questions is not used for root labels
                # (no min/max length constraints)
                is_literal = label["d_type"] == "literal"
                literal_options = None
                if is_literal and label.get("literal_options"):
                    literal_options = Literal[*label["literal_options"]]
                schema = _create_answer_schema(
                    total_questions=0,  # Not used for root labels
                    has_roots=has_roots,
                    is_root=label["c_type"] == "root",
                    is_number=check_if_numerical_label(label),
                    is_literal=is_literal,
                    literal_options=literal_options,
                )
                all_schemas.append(schema)
                count += 1

            else:
                if len(label["questions"]) == 0:
                    continue
                if len(label["questions"]) > batch_size:
                    for i in range(0, len(label["questions"]), batch_size):
                        selected_queries = label["questions"][i : i + batch_size]
                        total_questions = len(selected_queries)
                        label_queries = "\n".join(
                            [
                                f"{i + 1}. {query}"
                                for i, query in enumerate(selected_queries)
                            ]
                        )
                        label_data_with_selected_queries = deepcopy(label)
                        label_data_with_selected_queries["questions"] = label_queries
                        if "dependent_on" in label_data_with_selected_queries:
                            label_data_with_selected_queries["dependent_on"] = [
                                dep_info["root_label"]
                                for dep_info in label["dependent_on"]
                            ]
                        formatted_label_details = format_label_details(
                            label_data_with_selected_queries,
                            include_keys=["name", "description"],
                        )
                        label_prompt = _select_context_generation_prompt(
                            label_data_with_selected_queries,
                            has_roots,
                        )

                        label_prompt = label_prompt.format(
                            label_details=formatted_label_details,
                            questions=label_queries,
                            total_questions=total_questions,
                            special_user_instructions=custom_instructions,
                        )
                        all_prompts.append(label_prompt)
                        prompt_to_label_map[count] = label["name"]
                        is_literal = label["d_type"] == "literal"
                        literal_options = None
                        if is_literal and label.get("literal_options"):
                            literal_options = Literal[*label["literal_options"]]
                        schema = _create_answer_schema(
                            total_questions,
                            has_roots,
                            is_root=label["c_type"] == "root",
                            is_number=check_if_numerical_label(label),
                            is_literal=is_literal,
                            literal_options=literal_options,
                        )
                        all_schemas.append(schema)
                        count += 1
                else:
                    total_questions = len(label["questions"])
                    label_queries = "\n".join(
                        [
                            f"{i + 1}. {query}"
                            for i, query in enumerate(label["questions"])
                        ]
                    )
                    label_data = deepcopy(label)
                    if "dependent_on" in label_data:
                        label_data["dependent_on"] = [
                            dep_info["root_label"] for dep_info in label["dependent_on"]
                        ]
                    label_details = format_label_details(
                        label_data,
                        include_keys=["name", "description"],
                    )
                    label_prompt = _select_context_generation_prompt(
                        label_data,
                        has_roots,
                    )
                    label_prompt = label_prompt.format(
                        label_details=label_details,
                        questions=label_queries,
                        total_questions=total_questions,
                        special_user_instructions=custom_instructions,
                    )

                    all_prompts.append(label_prompt)
                    prompt_to_label_map[count] = label["name"]
                    is_literal = label["d_type"] == "literal"
                    literal_options = None
                    if is_literal and label.get("literal_options"):
                        literal_options = Literal[*label["literal_options"]]
                    schema = _create_answer_schema(
                        total_questions,
                        has_roots,
                        is_root=label["c_type"] == "root",
                        is_number=check_if_numerical_label(label),
                        is_literal=is_literal,
                        literal_options=literal_options,
                    )
                    all_schemas.append(schema)
                    count += 1

    table_structure_hash = {label["name"]: label for label in table_structure}

    chain_messages = []
    for index, label in prompt_to_label_map.items():
        prompt = all_prompts[index]
        prompt_message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            )
        ]
        if check_if_numerical_label(table_structure_hash[label]):
            chain_messages.append(input_numerical_messages + prompt_message)
        else:
            chain_messages.append(input_messages + prompt_message)

    error_msg = f"""
    chain_messages length: {len(chain_messages)}
    prompt_to_label_map length: {len(prompt_to_label_map)}
    all_schemas length: {len(all_schemas)}
    """
    if not (len(chain_messages) == len(prompt_to_label_map) == len(all_schemas)):
        logger.error(error_msg)
        raise ValueError(error_msg)
    return chain_messages, prompt_to_label_map, all_schemas
