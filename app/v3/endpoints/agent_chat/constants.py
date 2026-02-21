from enum import StrEnum


class Agents(StrEnum):
    MAIN_AGENT = "main_agent"
    DEEP_AGENT = "deep_agent"
    CODE_GENERATOR = "code_generator"


FILE_TYPES = {
    "ipynb": "code",
    "py": "code",
    "csv": "data",
    "json": "data",
    "txt": "data",
    "md": "data",
}
