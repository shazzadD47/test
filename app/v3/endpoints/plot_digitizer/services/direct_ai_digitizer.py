import numpy as np

from app.v3.endpoints import Status
from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.constants import (
    MAX_RETRIES,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.schemas import Florence2Output
from app.v3.endpoints.plot_digitizer.services.florence_2 import (
    extract_data_points_rescale,
    extract_florence_2_data,
)
from app.v3.endpoints.plot_digitizer.utils import (
    get_request_florence_2,
)


class DirectAIDigitizer:
    def __init__(self, main_image: np.ndarray, image_url: str, chart_type: str):
        self.image_url = image_url
        self.main_image = main_image
        self.chart_type = chart_type

    def _call_digitizer_ai_model(self) -> dict:
        call_ai_model = 0
        while True:
            try:
                florence_2_out: str = get_request_florence_2(
                    get_url=settings.FLORENCE_2_API, params={"img_url": self.image_url}
                )
                return {
                    "ai_output": florence_2_out,
                    "status": Status.SUCCESS.value,
                }
            except Exception as e:
                call_ai_model += 1
                if call_ai_model > MAX_RETRIES:
                    logger.info(f"Florence-2 call error: {e}")
                    return {
                        "ai_output": None,
                        "status": Status.FAILED.value,
                    }

    def get_digitized_plot(self) -> list[Florence2Output] | None:
        logger.info(f"Calling AI digitizer model for image: {self.image_url}")
        ai_output = self._call_digitizer_ai_model()
        if ai_output["status"] == Status.FAILED.value or not ai_output["ai_output"]:
            logger.info(
                f"AI digitizer call error. Output status:{ai_output['status']}. Output:{ai_output['ai_output']}"  # noqa E501
            )
            return None

        ai_model_output = ai_output["ai_output"]
        data_points_info = extract_florence_2_data(
            self.main_image.copy(), ai_model_output, self.chart_type
        )
        data_point_structured: list[Florence2Output] = extract_data_points_rescale(
            image_main=self.main_image,
            data_points_info=data_points_info,
            chart_type=self.chart_type,
        )
        logger.info("AI digitizer model ran successfully.")
        return data_point_structured
