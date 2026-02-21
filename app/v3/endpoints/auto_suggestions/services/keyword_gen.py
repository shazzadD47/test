from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.core.auto import AutoChatModel
from app.utils.llms import invoke_llm_with_retry
from app.v3.endpoints.auto_suggestions.configs import settings
from app.v3.endpoints.auto_suggestions.logging import celery_logger as logger
from app.v3.endpoints.auto_suggestions.schemas import FigureSearchCriteria


class KeyWordGenerator:
    def __init__(
        self,
        user_requirement: str,
        keyword_extraction_prompt: str,
        output_schema: BaseModel = FigureSearchCriteria,
    ):
        self.user_requirement: str = user_requirement
        self.keyword_extraction_prompt: str = keyword_extraction_prompt
        self.output_schema = output_schema
        self.keyword_parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=output_schema
        )
        self.keywords_info: dict = self._get_keywords()
        self.keywords_list: list[str] = self._get_keywords_list()
        self.keywords_extraction_info: dict = self._get_keywords_extraction_info()
        logger.info(f"Keywords Extraction Info: {self.keywords_extraction_info}")

    def _get_input_prompt(self) -> PromptTemplate:
        keyword_output_format_instructions: str = (
            self.keyword_parser.get_format_instructions()
        )
        input_prompt: PromptTemplate = PromptTemplate.from_template(
            self.keyword_extraction_prompt
        ).partial(
            user_input=self.user_requirement,
            format_instructions=keyword_output_format_instructions,
        )
        return input_prompt

    def _get_input_message(self) -> list[dict]:
        input_prompt = self._get_input_prompt()
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_prompt.format(),
                    },
                ],
            },
        ]
        return message

    def _get_llm(self) -> Any:
        keyword_llm = AutoChatModel.from_model_name(
            model_name=settings.KEYWORD_GEN_MODEL_NAME,
        )
        return keyword_llm.with_structured_output(self.output_schema)

    def _get_keywords(self) -> dict:
        try:
            keyword_extraction_output = invoke_llm_with_retry(
                self._get_llm(), self._get_input_message()
            )
        except Exception as e:
            logger.error(f"Error getting Keyword for auto suggestion output: {str(e)}")
            raise

        keywords_info: dict = keyword_extraction_output.model_dump()

        return keywords_info

    def _get_keywords_list(self) -> list[str]:
        keywords_list: list[str] = []
        for keyword in self.keywords_info["mandatory_keywords"]:
            keywords_list.append(keyword["keyword"])
        for keyword in self.keywords_info["optional_keywords"]:
            keywords_list.append(keyword["keyword"])
        return keywords_list

    def _get_keywords_extraction_info(self) -> dict:
        mandatory_keywords = [
            kw["keyword"] for kw in self.keywords_info["mandatory_keywords"]
        ]
        optional_keywords = [
            kw["keyword"] for kw in self.keywords_info["optional_keywords"]
        ]
        optional_keywords_info = self.keywords_info["optional_keywords"]

        return {
            "mandatory_keywords": mandatory_keywords,
            "optional_keywords": optional_keywords,
            "optional_keywords_info": optional_keywords_info,
            "keywords_list": self.keywords_list,
            "keywords_info": self.keywords_info,
        }
