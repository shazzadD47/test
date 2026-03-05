import time
from copy import deepcopy
from uuid import uuid4

import json_repair
import pandas as pd
from langfuse import observe

from app.constants import NUMERICAL_D_TYPES
from app.core.celery.app import celery_app
from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.utils import check_if_null
from app.v3.endpoints import Status
from app.v3.endpoints.dosing_table.utils import fix_arm_time_starting_from_one
from app.v3.endpoints.dynamic_dosing.constants import DYNAMIC_DOSING_ERROR_MESSAGE
from app.v3.endpoints.dynamic_dosing.logging import celery_logger as logger
from app.v3.endpoints.dynamic_dosing.prompt import build_dynamic_dosing_instruction
from app.v3.endpoints.general_extraction.services.tasks import (
    execute_general_extraction,
)


def _is_array_field(field: dict) -> bool:
    d_type = str(field.get("d_type", "")).lower().strip()
    c_type = str(field.get("c_type", "")).lower().strip()
    return c_type == "array" or d_type.startswith("list[")


def _coerce_scalar(value, d_type: str):
    lower_d_type = str(d_type).lower().strip()
    if lower_d_type.startswith("list[") and lower_d_type.endswith("]"):
        lower_d_type = lower_d_type[5:-1].strip()

    if check_if_null(value):
        if lower_d_type in NUMERICAL_D_TYPES or lower_d_type == "boolean":
            return None
        return "N/A"

    if lower_d_type in NUMERICAL_D_TYPES:
        try:
            numeric_value = float(value)
            if lower_d_type in {"integer", "int"}:
                return int(numeric_value)
            return numeric_value
        except Exception:
            return None

    if lower_d_type == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    return value


def _parse_list_like(value) -> list:
    if check_if_null(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        if "answers" in value and isinstance(value["answers"], list):
            return value["answers"]
        if "rows" in value and isinstance(value["rows"], list):
            return value["rows"]
        return list(value.values())

    if isinstance(value, (int, float, bool)):
        return [value]

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or check_if_null(stripped):
            return []

        try:
            parsed = json_repair.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if "answers" in parsed and isinstance(parsed["answers"], list):
                    return parsed["answers"]
                if "rows" in parsed and isinstance(parsed["rows"], list):
                    return parsed["rows"]
                return list(parsed.values())
            if parsed is not None and parsed != stripped:
                return [parsed]
        except Exception:
            pass

        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]

        return [stripped]

    return [value]


def _create_empty_dynamic_table(table_template: list[dict]) -> list[dict]:
    empty_row = {}
    for field in table_template:
        empty_row[field["name"]] = _coerce_scalar(None, field["d_type"])
    return [empty_row]


def _apply_arm_time_normalization(
    rows: list[dict], ordered_names: list[str]
) -> list[dict]:
    if "GROUP" not in ordered_names or "ARM_TIME" not in ordered_names:
        return rows

    try:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return rows

        frame["ARM_TIME"] = pd.to_numeric(frame["ARM_TIME"], errors="coerce")
        frame = fix_arm_time_starting_from_one(
            frame,
            group_column="GROUP",
            arm_time_column="ARM_TIME",
        )
        frame = frame.where(pd.notna(frame), None)

        return json_repair.loads(frame.to_json(orient="records"))
    except Exception:
        logger.exception("Failed to normalize ARM_TIME values.")
        return rows


def format_dynamic_dosing_output(
    data: list[dict] | None,
    table_template: list[dict],
) -> list[dict]:
    ordered_names = [field["name"] for field in table_template]
    defaults = {
        field["name"]: _coerce_scalar(None, field["d_type"]) for field in table_template
    }

    if not data or not isinstance(data, list):
        return _create_empty_dynamic_table(table_template)

    array_fields = [field["name"] for field in table_template if _is_array_field(field)]
    formatted_rows = []

    for raw_row in data:
        if not isinstance(raw_row, dict):
            continue

        base_row = {field_name: raw_row.get(field_name) for field_name in ordered_names}
        parsed_arrays = {
            field_name: _parse_list_like(base_row.get(field_name))
            for field_name in array_fields
        }

        if not parsed_arrays:
            row = {}
            for field in table_template:
                row[field["name"]] = _coerce_scalar(
                    base_row.get(field["name"]), field["d_type"]
                )
            formatted_rows.append(row)
            continue

        max_len = max([len(values) for values in parsed_arrays.values()] + [1])

        for idx in range(max_len):
            row = {}
            for field in table_template:
                field_name = field["name"]
                if field_name in parsed_arrays:
                    values = parsed_arrays[field_name]
                    current_value = values[idx] if idx < len(values) else None
                else:
                    current_value = base_row.get(field_name)

                row[field_name] = _coerce_scalar(current_value, field["d_type"])
            formatted_rows.append(row)

    if not formatted_rows:
        return _create_empty_dynamic_table(table_template)

    formatted_rows = _apply_arm_time_normalization(formatted_rows, ordered_names)

    ordered_rows = []
    for row in formatted_rows:
        ordered_rows.append(
            {
                field_name: row.get(field_name, defaults[field_name])
                for field_name in ordered_names
            }
        )

    return ordered_rows


@observe()
def extract_dynamic_dosing(
    project_id: str,
    paper_id: str,
    table_template: list[dict],
    langfuse_session_id: str = None,
    metadata: dict = None,
    image_url: str | list[str] | None = None,
) -> dict:
    """
    Extract dynamic dosing rows using the general extraction pipeline.
    """
    start_time = time.time()
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    if metadata is None:
        metadata = {}

    ge_table_template = deepcopy(table_template)
    for label in ge_table_template:
        if label.get("c_type") is None:
            label["c_type"] = "general"

        label["description"] += (
            ". Be very specific. "
            "Provide only the specific answer. "
            "Do not provide any other text or explanation."
        )

    ge_inputs = {
        # General extraction currently expects this identifier as `flag_id`.
        "flag_id": paper_id,
        "project_id": project_id,
        "table_structure": ge_table_template,
        "metadata": metadata,
    }

    if image_url is not None:
        image_urls = [image_url] if isinstance(image_url, str) else image_url
        ge_inputs["inputs"] = [
            {
                "type": "image",
                "data": [{"figure_url": url} for url in image_urls],
            }
        ]
    ge_inputs["custom_instruction"] = build_dynamic_dosing_instruction(
        table_template=table_template,
        has_figure=image_url is not None,
    )

    result = execute_general_extraction(ge_inputs)
    total_time = time.time() - start_time

    if result["metadata"]["status"] == Status.FAILED.value:
        return {
            "data": None,
            "status": Status.FAILED.value,
            "message": "Dynamic dosing extraction failed.",
            "runtime": total_time,
            "metadata": metadata,
        }

    metadata.update(result["metadata"])
    return {
        "data": result["payload"],
        "message": "Dynamic dosing extracted successfully",
        "status": Status.SUCCESS.value,
        "runtime": total_time,
        "metadata": metadata,
    }


@celery_app.task(name="extract_dynamic_dosing_task")
@observe()
@track_all_llm_costs
def extract_dynamic_dosing_task(
    paper_id: str,
    project_id: str,
    image_url: list[str] | str | None,
    table_template: list[dict],
    langfuse_session_id: str = None,
    request_metadata: dict = None,
) -> dict:
    logger.info(
        "Extracting dynamic dosing for paper %s and project %s",
        paper_id,
        project_id,
    )

    metadata = {"ai_metadata": {"cost_metadata": {}}}

    if request_metadata is not None:
        for key, value in request_metadata.items():
            if key not in metadata:
                metadata[key] = value

    event_type = (
        BackendEventEnumType.PRESET_DOSING_TABLE_WITH_FIGURE
        if image_url is not None
        else BackendEventEnumType.PRESET_DOSING_TABLE_NO_FIGURE
    )

    try:
        if langfuse_session_id is None:
            langfuse_session_id = uuid4().hex

        normalized_image_url = None
        if image_url is not None:
            if isinstance(image_url, list):
                valid_urls = [url for url in image_url if not check_if_null(url)]
                normalized_image_url = valid_urls if valid_urls else None
            elif not check_if_null(image_url):
                normalized_image_url = image_url

        if normalized_image_url is None:
            event_type = BackendEventEnumType.PRESET_DOSING_TABLE_NO_FIGURE

        result = extract_dynamic_dosing(
            project_id=project_id,
            paper_id=paper_id,
            table_template=table_template,
            langfuse_session_id=langfuse_session_id,
            metadata=metadata,
            image_url=normalized_image_url,
        )

        final_data = None
        if result["status"] == Status.SUCCESS.value:
            final_data = format_dynamic_dosing_output(result["data"], table_template)

        final_metadata = result.get("metadata", metadata)
        final_metadata["message"] = result["message"]
        final_metadata["status"] = result["status"]
        final_metadata["ai_metadata"]["runtime"] = result["runtime"]

        message = {
            "payload": final_data if final_data else {},
            "metadata": final_metadata,
        }

        send_to_backend(event_type, message)
        return message

    except Exception as e:
        logger.exception("Error occurred when extracting dynamic dosing: %s", e)
        metadata["message"] = DYNAMIC_DOSING_ERROR_MESSAGE
        metadata["status"] = Status.FAILED.value
        metadata["ai_metadata"]["runtime"] = 0

        message = {
            "payload": {},
            "metadata": metadata,
        }
        send_to_backend(event_type, message)
        return message
