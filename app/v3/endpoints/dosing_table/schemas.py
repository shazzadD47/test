import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, HttpUrl


class DosingTableRequest(BaseModel):
    project_id: str
    flag_id: str
    image_url: HttpUrl
    metadata: dict


class NoFigureDosingTableRequest(BaseModel):
    project_id: str
    flag_id: str
    metadata: dict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    initial_table: dict
