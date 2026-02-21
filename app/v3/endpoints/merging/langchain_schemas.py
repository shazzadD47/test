from pydantic import BaseModel


class UnitNormalizer(BaseModel):
    normalized_units: list[str]


class ValueFixer(BaseModel):
    fixed_values: list[str]


class DVNormalizer(BaseModel):
    normalized_values: list[str]


class RegimenNormalizer(BaseModel):
    normalized_regimens: list[str]
