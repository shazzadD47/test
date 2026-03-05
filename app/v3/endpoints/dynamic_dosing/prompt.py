from typing import Any

from app.constants import NUMERICAL_D_TYPES


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: Any) -> str:
    return _normalize_text(value).upper()


def _normalize_dtype(value: Any) -> str:
    return _normalize_text(value).lower()


def _base_dtype(dtype: str) -> str:
    normalized = _normalize_dtype(dtype)
    if normalized.startswith("list[") and normalized.endswith("]"):
        return normalized[5:-1].strip()
    return normalized


def _is_numeric_dtype(dtype: str) -> bool:
    base_type = _base_dtype(dtype)
    return base_type in NUMERICAL_D_TYPES or base_type in {"number", "numeric"}


def _is_boolean_dtype(dtype: str) -> bool:
    return _base_dtype(dtype) == "boolean"


def _is_array_dtype(dtype: str, c_type: str) -> bool:
    return _normalize_dtype(c_type) == "array" or _normalize_dtype(dtype).startswith(
        "list["
    )


def _contains_any(text: str, keywords: set[str]) -> bool:
    lowered = _normalize_text(text).lower()
    return any(keyword in lowered for keyword in keywords)


def build_dynamic_dosing_instruction(
    table_template: list[dict[str, Any]],
    has_figure: bool,
) -> str:
    """
    Build a dosing extraction instruction from the dynamic table template.

    Rules are attached only to fields that exist in the requested template.
    """
    normalized_template = table_template or []
    template_names = {
        _normalize_name(field.get("name")) for field in normalized_template
    }
    template_by_name = {
        _normalize_name(field.get("name")): field for field in normalized_template
    }

    field_lines = []
    numeric_fields: list[str] = []
    boolean_fields: list[str] = []
    array_fields: list[str] = []
    literal_fields: list[str] = []

    for idx, field in enumerate(normalized_template, start=1):
        name = _normalize_text(field.get("name"))
        dtype = _normalize_text(field.get("d_type"))
        c_type = _normalize_text(field.get("c_type") or "general")
        description = _normalize_text(field.get("description")) or "No description."
        literal_options = field.get("literal_options")

        field_line = f"{idx}. {name} (d_type={dtype}, c_type={c_type}): {description}"
        if literal_options:
            field_line += (
                f" Allowed values: {', '.join([str(opt) for opt in literal_options])}."
            )
            literal_fields.append(name)
        field_lines.append(field_line)

        if _is_numeric_dtype(dtype):
            numeric_fields.append(name)
        if _is_boolean_dtype(dtype):
            boolean_fields.append(name)
        if _is_array_dtype(dtype, c_type):
            array_fields.append(name)

    field_text_pairs = [
        (
            _normalize_text(field.get("name")),
            _normalize_text(field.get("description")),
        )
        for field in normalized_template
    ]

    has_time_semantics = any(
        _contains_any(
            name, {"time", "week", "day", "duration", "visit", "start", "end"}
        )
        or _contains_any(
            description,
            {"time", "week", "day", "duration", "visit", "start", "end"},
        )
        for name, description in field_text_pairs
    )
    has_dose_semantics = any(
        _contains_any(name, {"dose", "amt", "amount", "mg"})
        or _contains_any(description, {"dose", "amt", "amount", "mg"})
        for name, description in field_text_pairs
    )
    has_interval_semantics = any(
        _contains_any(name, {"interval", "ii", "frequency", "freq"})
        or _contains_any(description, {"interval", "ii", "frequency", "freq"})
        for name, description in field_text_pairs
    )

    global_rules = [
        "Extract dosing data for meta-analysis.",
        "Do not assume a fixed dosing schema. "
        "Some standard columns may be missing and custom columns may be present.",
        "Return only requested fields with exact column names from the template.",
        "GROUP is the single root treatment identifier.",
        "Do not add columns, notes, or explanation text.",
        "If a value is missing, use null for numeric/boolean fields and N/A for string fields.",
        "Do not fabricate unsupported facts. Infer only when strongly supported by evidence.",
        "Preserve escalation chronology step-by-step for each GROUP.",
    ]

    conditional_rules: list[str] = []
    if "ARM_TIME" in template_names:
        conditional_rules.append(
            "If lead-in/run-in exists, re-anchor timeline so earliest ARM_TIME is 0."
        )
    if "AMT" in template_names:
        conditional_rules.append(
            "If placebo is explicitly stated, AMT should be 0 for placebo rows."
        )
    if "ADDL" in template_names:
        conditional_rules.append(
            "When inferable, ADDL should equal total doses minus 1."
        )
    if "ARM_TIME_UNIT" in template_names and "II_UNIT" in template_names:
        conditional_rules.append(
            "ARM_TIME_UNIT and II_UNIT must be internally consistent."
        )
    if "AMT_UNIT" in template_names:
        conditional_rules.append(
            "AMT_UNIT must match the stated dose magnitude and regimen text."
        )
    if has_time_semantics:
        conditional_rules.append(
            "For time-like fields, keep values chronologically ordered within each GROUP."
        )
    if has_dose_semantics:
        conditional_rules.append(
            "For dose-like fields, preserve escalation/de-escalation transitions without dropping steps."
        )
    if has_interval_semantics:
        conditional_rules.append(
            "For interval-like fields, keep units and cadence logically consistent with regimen narrative."
        )

    if array_fields:
        conditional_rules.append(
            f"Array-like fields ({', '.join(array_fields)}) should align by index for each dosing step."
        )
    if numeric_fields:
        conditional_rules.append(
            f"Numeric fields ({', '.join(numeric_fields)}) must be numeric or null."
        )
    if boolean_fields:
        conditional_rules.append(
            f"Boolean fields ({', '.join(boolean_fields)}) must be true, false, or null."
        )
    if literal_fields:
        conditional_rules.append(
            f"Literal fields ({', '.join(literal_fields)}) must use only their allowed values."
        )

    # Guidance for user-provided dynamic fields that may appear over time.
    dynamic_hint_lines = []
    for name, field in template_by_name.items():
        if name in {
            "GROUP",
            "ARM_TIME",
            "AMT",
            "ADDL",
            "ARM_TIME_UNIT",
            "II_UNIT",
            "AMT_UNIT",
        }:
            continue
        dynamic_hint_lines.append(
            f"- {field.get('name')}: use evidence strictly based on its description and d_type."
        )

    mode_rules = (
        [
            "Figure-aware mode:",
            "- Use figure/image evidence as primary for escalation timing and dose transitions.",
            "- Use document context as support; if conflict exists, prefer figure evidence.",
            "- Read axis units/ticks carefully when time columns are present.",
        ]
        if has_figure
        else [
            "Document-only mode:",
            "- Use study design/methods/table text as primary evidence.",
            "- Reconstruct regimen chronology only for requested fields.",
            "- If timeline exists but start is missing, start first step at 0 when ARM_TIME is requested.",
        ]
    )

    sections = [
        "[Requested Template]",
        *field_lines,
        "",
        "[Global Rules]",
        *[f"- {rule}" for rule in global_rules],
        "",
        "[Template-Conditional Rules]",
        *(
            [f"- {rule}" for rule in conditional_rules]
            if conditional_rules
            else ["- No special conditional rules."]
        ),
        "",
        "[Mode Rules]",
        *mode_rules,
    ]

    if dynamic_hint_lines:
        sections.extend(["", "[Dynamic Field Handling]", *dynamic_hint_lines])

    return "\n".join(sections).strip()
