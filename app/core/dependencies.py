import re
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError, field_validator


class FlagIDValidator(BaseModel):
    flag_id: str

    @field_validator("flag_id")
    def validate_uuid_and_extended_pattern(cls, value: str):
        uuid_pattern = r"^[0-9a-fA-F\-]{36}$"
        extended_pattern = r"^[0-9a-fA-F\-]{36}-supplementary-[0-9a-fA-F]+$"
        if re.match(uuid_pattern, value):
            cls.validate_uuid(value)
        elif re.match(extended_pattern, value):
            uuid_part = value.split("-supplementary-")[0]
            cls.validate_uuid(uuid_part)
        else:
            raise ValueError(
                "flag_id match either '<UUID>' or '<UUID>-supplementary-<identifier>'"
            )
        return value

    @staticmethod
    def validate_uuid(value: str):
        try:
            parsed_uuid = UUID(value)  # Validate UUID
            if parsed_uuid.version not in [1, 4]:
                raise ValueError("The UUID part of flag_id must be of version 1 or 4.")
        except ValueError:
            raise ValueError("The UUID part of flag_id is not valid.")


def validate_flag_id(flag_id: str) -> str:
    try:
        validated_data = FlagIDValidator(flag_id=flag_id)
        return validated_data.flag_id
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Flag ID validation error: {e.errors()}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate flag id: {e}",
        )
