from typing import Any

import cv2
import numpy as np

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.constants import ChartType
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.florence_2 import (
    florence2_call_manual_mapping,
)


class SpiderPlotDigitizer:
    def __init__(self, img_url: str, autofill_response: Any):
        self.img_url = img_url
        self.main_image = self._get_main_image()
        self.autofill_response = autofill_response
        self.format_autofill_response()

    def format_autofill_response(self):
        self.autofill_response["data"]["detected_line_points"] = []
        list_line = self.autofill_response["data"]["lines"]
        for data in list_line:
            data["points"] = []

    def _get_main_image(self) -> np.ndarray:
        image_byte = get_image_from_url(self.img_url)
        image_main = np.frombuffer(image_byte, np.uint8)
        image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
        image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
        return image_main

    def digitize_plot(self) -> Any:
        try:
            self.autofil_response = florence2_call_manual_mapping(
                self.main_image,
                self.img_url,
                self.autofill_response,
                chart_type=ChartType.SPIDER_PLOT,
            )
        except Exception as e:
            logger.exception(
                f"Failed to digitizer spider plot: {e}. Error message: {e}"
            )
        return self.autofill_response
