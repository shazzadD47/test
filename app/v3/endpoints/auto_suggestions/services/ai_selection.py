from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.core.auto import AutoChatModel
from app.utils.llms import invoke_llm_with_retry
from app.v3.endpoints.auto_suggestions.configs import settings
from app.v3.endpoints.auto_suggestions.constants import (
    AISelectionInputTypes,
    DatabaseFigureTypes,
)
from app.v3.endpoints.auto_suggestions.logging import celery_logger as logger
from app.v3.endpoints.auto_suggestions.schemas import (
    AutoFigureConnectionAnnotation,
    ImageSelection,
    ImageSelectionOutput,
)


class AIAutoSuggestionGenerator:
    def __init__(
        self,
        all_data: list[AutoFigureConnectionAnnotation],
        keywords_extraction_info: dict,
        user_requirement: str,
        selection_type: list[AISelectionInputTypes],
        figure_auto_suggestion_prompt: str,
        output_schema: BaseModel = ImageSelectionOutput,
    ):
        self.all_data: list[AutoFigureConnectionAnnotation] = all_data
        self.keywords_extraction_info: dict = keywords_extraction_info
        self.user_requirement: str = user_requirement
        self.selection_type: list[str] = selection_type
        self.figure_auto_suggestion_prompt: str = figure_auto_suggestion_prompt
        self.output_schema: BaseModel = output_schema
        self.auto_selection_output_parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=output_schema
        )
        figure_details: dict = self._gen_figures_details()
        self.figures_info: list[dict] = figure_details["figures_info"]
        self.figure_id_map: list[dict] = figure_details["figure_id_map"]

    def _gen_figures_details(self) -> dict:
        figure_id_map = []
        figures_info = []
        for idx, image_data in enumerate(self.all_data):
            figure_type: str = image_data["type"]
            if figure_type not in self.selection_type:
                continue
            figure_id: str = image_data["id"]
            mapped_figure_number: int = f"{idx+1}"
            figure_caption: str = image_data["caption"]
            figure_footnote: str = image_data["footnote"]
            figure_description: str = image_data["description"]

            figure_id_map.append({idx + 1: figure_id})

            if figure_type == DatabaseFigureTypes.PLOT:
                figures_info.append(
                    {
                        "image_description_generated": figure_description,
                        "image_number": mapped_figure_number,
                    }
                )
            elif figure_type == DatabaseFigureTypes.TABLE:
                figures_info.append(
                    {
                        "image_description_generated": figure_description,
                        "image_number": mapped_figure_number,
                        **(
                            {"caption": figure_caption}
                            if figure_caption is not None
                            else {}
                        ),
                        **(
                            {"footnote": figure_footnote}
                            if figure_footnote is not None
                            else {}
                        ),
                    }
                )

        return {
            "figures_info": figures_info,
            "figure_id_map": figure_id_map,
        }

    def _get_input_prompt(self) -> PromptTemplate:
        auto_figure_suggestion_output_instruction: str = (
            self.auto_selection_output_parser.get_format_instructions()
        )
        input_prompt: PromptTemplate = PromptTemplate.from_template(
            self.figure_auto_suggestion_prompt
        ).partial(
            format_instructions=auto_figure_suggestion_output_instruction,
            user_requirement=self.user_requirement,
            input_json=self.figures_info,
            keywords=self.keywords_extraction_info["keywords_list"],
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
        auto_suggestion_llm = AutoChatModel.from_model_name(
            model_name=settings.SUGGESTION_GEN_MODEL_NAME,
        )
        return auto_suggestion_llm.with_structured_output(self.output_schema)

    def _replace_figure_id_with_number(self, llm_output: list[ImageSelection]):
        for output in llm_output:
            for mapping in self.figure_id_map:
                for k, v in mapping.items():
                    if str(k) == str(output["image_number"]):
                        output["image_number"] = v
        return llm_output

    def get_llm_suggestion_output(self) -> list[ImageSelection]:
        try:
            llm_output_result = invoke_llm_with_retry(
                self._get_llm(), self._get_input_message()
            )
        except Exception as e:
            logger.error(f"Error getting LLM suggestion output: {str(e)}")
            raise
        llm_output_result = llm_output_result.model_dump()["selected_images"]
        llm_output_result = self._replace_figure_id_with_number(llm_output_result)

        return llm_output_result
