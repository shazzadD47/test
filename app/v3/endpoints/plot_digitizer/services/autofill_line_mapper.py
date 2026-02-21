import base64
import unicodedata
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.core.auto import AutoChatModel
from app.utils.image import get_image_from_url
from app.utils.llms import invoke_llm_with_retry
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.helpers import PlotDigitizerHelper
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.prompts import LEGEND_MAPPING_PROMPT
from app.v3.endpoints.plot_digitizer.schemas import (
    Florence2Output,
    LLMLegendMapGenInput,
    LLMLegendMapGenOutputList,
)


class LLMLegendMapGen:
    def __init__(
        self,
        ai_model_legends: list[str],
        autofill_legends: list[str],
        image_url: str,
        legend_mapping_prompt: str = LEGEND_MAPPING_PROMPT,
        output_schema: BaseModel = LLMLegendMapGenOutputList,
        input_schema: BaseModel = LLMLegendMapGenInput,
    ):
        self.ai_model_legends = ai_model_legends
        self.output_schema = output_schema
        self.autofill_legends = autofill_legends
        self.image_url = image_url
        self.parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=output_schema
        )
        self.legend_mapping_prompt = legend_mapping_prompt

    def _get_base_64_image(self) -> tuple[str, str]:
        image_bytes, media_type = get_image_from_url(
            url=self.image_url, return_media_type=True
        )
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_image, media_type

    def _get_input_prompt(self) -> PromptTemplate:
        output_format_instructions: str = self.parser.get_format_instructions()
        input_prompt: PromptTemplate = PromptTemplate.from_template(
            self.legend_mapping_prompt
        ).partial(
            ai_model_legends=self.ai_model_legends,
            autofill_legends=self.autofill_legends,
            output_format_instructions=output_format_instructions,
        )
        return input_prompt

    def _get_input_message(self) -> list[dict]:
        base64_image, media_type = self._get_base_64_image()
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

    def _get_llm(self) -> AutoChatModel:
        llm = AutoChatModel.from_model_name(
            model_name=settings.LEGEND_MAPPIND_MODEL_NAME,
        )
        return llm.with_structured_output(self.output_schema)

    def get_mapped_legends(self) -> LLMLegendMapGenOutputList:
        legend_mapping_output = invoke_llm_with_retry(
            self._get_llm(), self._get_input_message()
        )
        return legend_mapping_output.model_dump()


class AutofillMapper:
    def __init__(
        self, autofill_response: dict, ai_pd_output: list[dict] | None, image_url: str
    ):
        self.image_url = image_url
        self.autofill_response = autofill_response
        self.plot_type = self.autofill_response["data"]["plot_axis_data"]["chart_type"]
        self.is_autofill_line_empty = False
        self.autofill_legends = self._get_autofill_legends()
        logger.info(f"Autofill extracted line names are: {self.autofill_legends}")
        self.ai_pd_output = ai_pd_output
        if self.ai_pd_output:
            self.ai_detected_line_name = self._get_ai_detected_line_name()
        else:
            self.ai_detected_line_name = []
        logger.info(f"AI model detected line names are {self.ai_detected_line_name}")

    def format_autofill_response(self, return_output=False):
        self.autofill_response["data"]["detected_line_points"] = []
        list_line = self.autofill_response["data"]["lines"]
        if list_line:
            for data in list_line:
                data["points"] = []
        if return_output:
            return self.autofill_response

    def _get_autofill_legends(self) -> list[str]:
        self.format_autofill_response()
        autofill_legends = []
        list_line = self.autofill_response["data"]["lines"]
        if list_line:
            for line in list_line:
                autofill_legends.append(line["labels"]["line_name"])
        elif self.autofill_response["data"]["legends"]:
            self.is_autofill_line_empty = True
            autofill_legends = self.autofill_response["data"]["legends"]
        return autofill_legends

    def _get_ai_detected_line_name(self) -> list[str]:
        # get ai detected line name
        ai_detected_line_name = []
        for ai_output in self.ai_pd_output:
            ai_detected_line_name.append(ai_output["line_name"])
        return ai_detected_line_name

    def _gen_sentence_llm_legend_map(self) -> list[dict]:
        # get llm based legend map
        llm_legend_map_gen = LLMLegendMapGen(
            ai_model_legends=self.ai_detected_line_name,
            autofill_legends=self.autofill_legends,
            image_url=self.image_url,
            legend_mapping_prompt=LEGEND_MAPPING_PROMPT,
            output_schema=LLMLegendMapGenOutputList,
            input_schema=LLMLegendMapGenInput,
        )

        mapped_legends = llm_legend_map_gen.get_mapped_legends()
        return mapped_legends

    @staticmethod
    def normalize_text(input_text: str) -> str:
        normalized_text = unicodedata.normalize("NFKC", input_text)
        normalized_text = normalized_text.rstrip().lower()
        return normalized_text

    def _get_mapped_lines(self) -> list[dict]:
        llm_mapped_legends = self._gen_sentence_llm_legend_map()
        logger.info(f"LLM mapped legends: {llm_mapped_legends['matched_list']}")
        mapped_points = []
        for mapped_line_name in llm_mapped_legends["matched_list"]:
            ai_detected_line_name = mapped_line_name["ai_model_legend"]
            autofill_legend = mapped_line_name["autofill_legend"]
            for ai_output in self.ai_pd_output:
                if self.normalize_text(ai_detected_line_name) == self.normalize_text(
                    ai_output["line_name"]
                ):
                    logger.info(
                        f"ai_output: {ai_output} mapped with autofill_legend: {autofill_legend}"  # noqa: E501
                    )
                    mapped_points.append(
                        {
                            "line_name": autofill_legend,
                            "data_points": ai_output["data_points"],
                        }
                    )
                    break
        return mapped_points

    def _convert_to_reponse_point_format(
        self, point_list: list[tuple[float, float]]
    ) -> list[dict]:
        response_point_list = []
        for point in point_list:
            point_dict = PlotDigitizerHelper.get_point_dict(point, self.plot_type)
            response_point_list.append(point_dict)
        return response_point_list

    def _format_magic_key_output(self):
        for output in self.ai_pd_output:
            data_points = output["data_points"]
            reponse_format_point = self._convert_to_reponse_point_format(data_points)
            self.autofill_response["data"]["detected_line_points"].append(
                reponse_format_point
            )

    def get_mapped_autofill_response(self) -> Any:
        logger.info("Mapping autofill legendss to AI model output legends")
        if not self.ai_pd_output:
            return self.autofill_response

        mapped_points = self._get_mapped_lines()
        list_line = self.autofill_response["data"]["lines"]
        self._format_magic_key_output()

        if self.is_autofill_line_empty:
            line_info = []
            for line_points in mapped_points:
                temp_point = self._convert_to_reponse_point_format(
                    line_points["data_points"]
                )
                line_info.append(
                    {
                        "labels": {"line_name": line_points["line_name"]},
                        "points": temp_point,
                    }
                )
            self.autofill_response["data"]["lines"] = line_info

        else:
            for line in list_line:
                autofill_line_name = line["labels"]["line_name"]
                for line_mapped in mapped_points:
                    if self.normalize_text(autofill_line_name) == self.normalize_text(
                        line_mapped["line_name"]
                    ):
                        line["points"] = self._convert_to_reponse_point_format(
                            line_mapped["data_points"]
                        )
                        break

        return self.autofill_response


class AutofillAIMapperKaplan(AutofillMapper):
    def __init__(
        self,
        autofill_response: dict,
        ai_pd_output: list[Florence2Output],
        image_url: str,
    ):
        super().__init__(autofill_response, ai_pd_output, image_url)

    def _convert_to_reponse_point_format(
        self, point_list: list[tuple[float, float]]
    ) -> list[dict]:
        response_point_list = []
        for point in point_list:
            response_point_list.append(
                {
                    "x": round(point[0]),
                    "y": round(point[1]),
                    "topBarPixelDistance": 0,
                    "bottomBarPixelDistance": 0,
                    "deviationPixelDistance": 0,
                }
            )
        return response_point_list
