import time
import uuid
from typing import Any

import cv2
import numpy as np

from app.utils.image import get_image_from_url
from app.v3.endpoints.plot_digitizer.constants import (
    DIGITIZATION_ERROR_MESSAGE,
)
from app.v3.endpoints.plot_digitizer.helpers import PlotDigitizerHelper
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.autofill_line_mapper import (
    AutofillMapper,
)
from app.v3.endpoints.plot_digitizer.services.direct_ai_digitizer import (
    DirectAIDigitizer,
)
from app.v3.endpoints.plot_digitizer.services.florence_2 import (
    florence2_call_manual_mapping,
    get_florence_2_only,
)


def florence_2_manual_digitizer(
    image_main: np.ndarray,
    img_url: str,
    autofil_response: Any,
    logger: Any,
    error_message: str = "",
) -> Any:
    start_time = time.time()
    try:
        autofil_response = florence2_call_manual_mapping(
            image_main, img_url, autofil_response
        )
        logger.info(
            f"Florence2 digitizer finished in: {time.time() - start_time:0.2f} sec"
        )
    except Exception as e:
        error_message = DIGITIZATION_ERROR_MESSAGE
        error_message += "Error occured when executing Florence2 manual mapping."
        logger.exception(
            f"Failed to get data points from Florence2: {e}. Error message: {error_message}"  # noqa E501
        )

    return autofil_response


def ai_digitizer(img_url: str, autofil_response: Any, chart_type: str) -> Any:
    try:
        logger.info(f"Starting AI plot digitizer. Input image url is: {img_url}")
        error_message = DIGITIZATION_ERROR_MESSAGE
        try:
            image_byte = get_image_from_url(img_url)
        except Exception as e:
            error_message += "Error occured when getting image from url."
            logger.exception(
                f"Error when tried to get image from url: {e}. Error message: {error_message}"  # noqa E501
            )
        image_main = np.frombuffer(image_byte, np.uint8)
        image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
        image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)

        autofil_response["data"]["detected_line_points"] = []
        list_line = autofil_response["data"]["lines"]
        for data in list_line:
            data["points"] = []

        has_legend = autofil_response["data"]["has_legend"]
        is_single_line = len(list_line) == 1

        if is_single_line:
            logger.info("Found single line plot image")
            start_time = time.time()
            try:
                data_point_structured = get_florence_2_only(
                    image_main, img_url, chart_type
                )

                for point_data in data_point_structured:
                    point_list = point_data["data_points"]
                    temp_point = []
                    for point in point_list:
                        point_dict = PlotDigitizerHelper.get_point_dict(
                            point, chart_type
                        )
                        temp_point.append(point_dict)

                    autofil_response["data"]["detected_line_points"].append(temp_point)

                point_list = max(
                    data_point_structured,
                    key=lambda list_points: len(list_points["data_points"]),
                )
                temp_point = []
                for point in point_list["data_points"]:
                    point_dict = PlotDigitizerHelper.get_point_dict(point, chart_type)
                    temp_point.append(point_dict)
                for data in list_line:
                    data["points"] = temp_point
                logger.info(
                    f"Florence2 digitizer finished in: {time.time() - start_time:0.2f} sec"  # noqa E501
                )
            except Exception as e:
                error_message += "Error occured when getting data points from Florence2 for single line plot."  # noqa E501
                logger.exception(
                    f"Failed to get data points from Florence2: {e}. Error message: {error_message}"  # noqa E501
                )

        elif not has_legend:
            logger.info(
                "Plot image has no legend. Assigning data points for manual mapping"
            )
            autofil_response = florence_2_manual_digitizer(
                image_main, img_url, autofil_response, logger
            )

        else:
            start_time = time.time()

            logger.info("Running direct AI digitizer")
            ai_pd_output = DirectAIDigitizer(
                image_main, img_url, chart_type
            ).get_digitized_plot()
            autofill_mapper = AutofillMapper(autofil_response, ai_pd_output, img_url)
            if not ai_pd_output:
                autofil_response = autofill_mapper.format_autofill_response(
                    return_output=True
                )
                logger.info("Florence2 digitizer failed")
            else:
                autofil_response = autofill_mapper.get_mapped_autofill_response()
                logger.info(
                    f"Florence2 digitizer finished in: {time.time() - start_time:0.2f} sec"  # noqa E501
                )

        return autofil_response

    except Exception as e:
        if DIGITIZATION_ERROR_MESSAGE in str(e):
            logger.exception(f"Error occured when digitizing plot: {e}")
        else:
            logger.exception(
                f"Error occured when digitizing plot: {e}. Error message: {DIGITIZATION_ERROR_MESSAGE}"  # noqa E501
            )


def associate_lines_spider_plot(autofill_response: Any) -> Any:
    if not autofill_response["data"].get("detected_line_points"):
        logger.info("No detected lines in Spider Plot. Returning original response.")
        return autofill_response

    autofill_response["data"]["lines"] = []
    for line_idx, line_points in enumerate(
        autofill_response["data"].get("detected_line_points")
    ):
        temp_line_data = {
            "labels": {"line_name": f"line_{line_idx}"},
            "points": line_points,
            "id": str(uuid.uuid4()),
        }
        autofill_response["data"]["lines"].append(temp_line_data)

    return autofill_response
