import re

import cv2
import numpy as np
from PIL import Image

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import (
    Florence2DataInfo,
    Florence2Output,
)
from app.v3.endpoints.plot_digitizer.utils import get_request_florence_2


class KaplainMeierDigitizer:
    def __init__(self, img_url: str):
        self.img_url = img_url
        self.image_main = self.load_main_image()

    def load_main_image(self) -> np.ndarray:
        try:
            image_byte = get_image_from_url(self.img_url)
        except Exception as e:
            logger.exception(f"Error when tried to get image from url: {e}.")
        image_main = np.frombuffer(image_byte, np.uint8)
        image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
        image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
        return image_main

    def _extract_data_from_gen_string(
        self, input_string: str, image_width: int, image_height: int
    ) -> list[dict]:
        """
        Extract data from the generated string.
        """
        pattern = r"(.*?)((?:<loc_\w+>)+)"
        matches = re.findall(pattern, input_string)
        output = []
        for match in matches:
            line_name = match[0]
            points = match[1]
            pattern_point = r"<loc_(.*?)><loc_(.*?)>"
            points_match = re.findall(pattern_point, points)
            for x, y in points_match:
                out_temp = {
                    "line_name": line_name,
                    "x": int(int(x) * image_width / 1000),
                    "y": int(int(y) * image_height / 1000),
                }
                output.append(out_temp)
        return output

    def _extract_florence_2_data_points(self, florence_2_out: str) -> Florence2DataInfo:
        """
        Extract data-points from the florence2 output raw output into sturctured format.
        """
        precess_image = Image.fromarray(self.image_main.copy())

        image_height = precess_image.height
        image_width = precess_image.width

        suffix_out = self._extract_data_from_gen_string(
            florence_2_out["generated_text"], image_width, image_height
        )

        extracted_line_name = []
        data_points_info = {"data_points": []}
        for data in suffix_out:
            line_name = data["line_name"].replace("</s>", "").replace("<s>", "")
            if line_name not in extracted_line_name:
                extracted_line_name.append(line_name)
                tmp_line_data = {line_name: [(data["x"], data["y"])]}
                data_points_info["data_points"].append(tmp_line_data)
            else:
                for line in data_points_info["data_points"]:
                    check_line_name = list(line.keys())[0]
                    if check_line_name == line_name:
                        line[line_name].append((data["x"], data["y"]))
                        break

        return data_points_info

    def _extract_data_points_structured(
        self, data_points_info: Florence2DataInfo
    ) -> list[Florence2Output]:
        data_point_structured = []
        for line in data_points_info["data_points"]:
            point_list = list(line.values())[0]
            line_name = list(line.keys())[0]
            data_point_structured.append(
                {"line_name": line_name, "data_points": point_list}
            )
        return data_point_structured

    def get_ai_digitized_points(self) -> list[Florence2Output]:
        kaplain_meier_out = get_request_florence_2(
            get_url=settings.KAPLAIN_MEIER_API, params={"img_url": self.img_url}
        )
        data_points_info = self._extract_florence_2_data_points(kaplain_meier_out)
        ai_digitized_points = self._extract_data_points_structured(data_points_info)
        return ai_digitized_points
