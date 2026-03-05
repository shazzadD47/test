import json
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from app.constants import VALID_C_TYPES, VALID_D_TYPES


class DynamicDosingTableField(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal[VALID_D_TYPES] = Field(  # type: ignore[valid-type]
        ..., description="The data type of the field"
    )
    c_type: Literal[VALID_C_TYPES] | None = Field(  # type: ignore[valid-type]
        None, description="The source type of the field"
    )
    literal_options: list[str] | None = Field(
        None, description="The options for the literal type field"
    )

    @field_validator("literal_options", mode="before")
    @classmethod
    def normalize_literal_options(cls, value):
        if value is None:
            return None

        if isinstance(value, list):
            return [str(item) for item in value]

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "na", "n/a"}:
                return None

            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    pass

            return [item.strip() for item in stripped.split(",") if item.strip()]

        return None


class DynamicDosingPayload(BaseModel):
    project_id: str = Field(..., description="The ID of the project")
    paper_id: str = Field(
        ...,
        description="The ID of the paper",
        validation_alias=AliasChoices("paper_id", "flag_id"),
    )
    image_url: list[str] | str | None = Field(
        None, description="Optional image URL(s) for dosing extraction"
    )
    table_template: list[DynamicDosingTableField] = Field(
        ...,
        description="The table template to extract information",
        validation_alias=AliasChoices(
            "table_template", "table_tamplate", "table_structure"
        ),
    )

    @field_validator("table_template", mode="after")
    @classmethod
    def normalize_group_root(
        cls, table_template: list[DynamicDosingTableField]
    ) -> list[DynamicDosingTableField]:
        group_fields = [
            field for field in table_template if field.name.lower().strip() == "group"
        ]

        if len(group_fields) != 1:
            raise ValueError("table_template must contain exactly one GROUP field")

        group_field = group_fields[0]
        if group_field.d_type.lower().strip() != "string":
            raise ValueError("GROUP field must have d_type='string'")

        for field in table_template:
            field_name = field.name.lower().strip()
            c_type = (field.c_type or "general").lower().strip()

            if field_name == "group":
                field.c_type = "root"
            elif c_type == "root":
                field.c_type = "general"
            elif c_type in {"general", "array", "paper_label"}:
                field.c_type = c_type
            else:
                field.c_type = "general"

        return table_template


class DynamicDosingRequest(BaseModel):
    payload: DynamicDosingPayload
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_request_shape(cls, data: Any):
        """
        Accept both current and legacy request shapes.

        Current:
        {
          "payload": {...},
          "metadata": {...}
        }

        Legacy/flat:
        {
          "project_id": "...",
          "paper_id": "...",  // or flag_id
          "table_structure": [...],
          "metadata": {...}
        }
        """
        if not isinstance(data, dict):
            return data

        if "payload" in data:
            if "metadata" not in data or not isinstance(data.get("metadata"), dict):
                data["metadata"] = {}
            return data

        payload_like_keys = {
            "project_id",
            "paper_id",
            "flag_id",
            "image_url",
            "table_template",
            "table_tamplate",
            "table_structure",
        }
        if not any(key in data for key in payload_like_keys):
            return data

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        payload = {key: value for key, value in data.items() if key != "metadata"}
        return {"payload": payload, "metadata": metadata}
