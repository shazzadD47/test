import base64

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.core.auto import AutoChatModel
from app.utils.image import get_image_from_url
from app.utils.llms import invoke_llm_with_retry
from app.v3.endpoints.plot_digitizer.configs import settings as pd_settings
from app.v3.endpoints.plot_digitizer.prompts import (
    GEMINI_AXIS_VALUES_EXTRACTION_PROMPT,
)
from app.v3.endpoints.plot_digitizer.schemas import GeminiAxisValuesOutput


class AxisValueExtractor:
    def __init__(
        self,
        image_url: str,
        instruction_prompt: str = GEMINI_AXIS_VALUES_EXTRACTION_PROMPT,
        output_schema: BaseModel = GeminiAxisValuesOutput,
    ):
        self.image_url = image_url
        self.instruction_prompt = instruction_prompt
        self.output_schema = output_schema

        self.parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=output_schema
        )

    def _get_base64_image(self) -> tuple[str, str]:
        image_bytes, media_type = get_image_from_url(
            url=self.image_url, return_media_type=True
        )
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_image, media_type

    def _get_input_prompt(self) -> PromptTemplate:
        output_format_instructions = self.parser.get_format_instructions()

        # Same strategy as LLMLegendMapGen
        input_prompt: PromptTemplate = PromptTemplate.from_template(
            self.instruction_prompt
        ).partial(
            output_format_instructions=output_format_instructions,
        )
        return input_prompt

    # ---------------------------------------------------------
    def _get_input_message(self) -> list[dict]:
        base64_image, media_type = self._get_base64_image()
        input_prompt = self._get_input_prompt()

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": input_prompt.format(),
                    },
                ],
            },
        ]
        return message

    # ---------------------------------------------------------
    def _get_llm(self) -> AutoChatModel:
        llm = AutoChatModel.from_model_name(
            model_name=pd_settings.AXIS_MODEL_NAME,
        )
        return llm.with_structured_output(self.output_schema)

    # ---------------------------------------------------------
    def get_axis_values(self):
        llm = self._get_llm()
        message = self._get_input_message()
        response = invoke_llm_with_retry(llm, message)
        return response.model_dump()
