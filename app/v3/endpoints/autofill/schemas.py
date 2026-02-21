from typing import Literal

from pydantic import BaseModel


class MetaAnalysisTableField(BaseModel):
    name: str
    description: str
    d_type: Literal["string", "float", "integer"]


class MetaAnalysisRootChoices(BaseModel):
    name: str
    description: str
    values: list[str]


class MetaAnalysisAutofillRequest(BaseModel):
    project_id: str
    paper_id: str
    table_structure: list[MetaAnalysisTableField]
    is_root: bool = False
    root_choices: list[MetaAnalysisRootChoices] | None = None
