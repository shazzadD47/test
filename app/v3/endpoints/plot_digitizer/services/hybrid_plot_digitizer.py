import time

import cv2
import numpy as np

from app.utils.image import async_get_image_from_url
from app.v3.endpoints.plot_digitizer.constants import ChartType
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.color_filtering import (
    find_dominant_color,
)
from app.v3.endpoints.plot_digitizer.services.end_to_end_autofill_digitizer import (  # noqa: E501
    end_to_end_autofill_digitizer,
)
from app.v3.endpoints.plot_digitizer.services.florence_2 import (
    chart_dete_call,
    get_florence_2_only,
    get_florence_2_points,
    run_florence2_randomly,
)
from app.v3.endpoints.plot_digitizer.services.sentence_xfomer_mapping import (  # noqa: E501
    get_sentence_xfomer_mapping,
)
from app.v3.endpoints.plot_digitizer.utils import (
    extract_plot_area,
    filter_points,
)


async def line_former_digitizer(img_url: str, autofil_response):
    start_time = time.time()
    response = await end_to_end_autofill_digitizer(
        img_url=img_url, autofil_response=autofil_response
    )
    logger.info(
        f"LineFormer digitization finished in: {time.time() - start_time:0.2f} sec"
    )

    return response


async def hybrid_plot_digitizer(img_url: str, autofil_response):
    logger.info(f"Starting hybrid plot digitizer. Input image url is: {img_url}")
    image_byte = await async_get_image_from_url(img_url)
    image_main = np.frombuffer(image_byte, np.uint8)
    image_main = cv2.imdecode(image_main, cv2.IMREAD_COLOR)
    image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)

    list_line = autofil_response["data"]["lines"]
    line_count = len(list_line)
    for data in list_line:
        data["points"] = []
    has_legend = autofil_response["data"]["has_legend"]
    is_single_line = len(list_line) <= 1
    chart_type = autofil_response["data"]["plot_axis_data"]["chart_type"]

    if is_single_line:
        logger.info("Found single line plot image")
        start_time = time.time()
        data_point_structured = await get_florence_2_only(image_main, img_url)
        point_list = max(
            data_point_structured,
            key=lambda list_points: len(list_points["data_points"]),
        )
        temp_point = []
        for point in point_list["data_points"]:
            temp_point.append(
                {
                    "x": int(point[0]),
                    "y": int(point[1]),
                    "topBarPixelDistance": 0,
                    "bottomBarPixelDistance": 0,
                    "deviationPixelDistance": 0,
                }
            )
        for data in list_line:
            data["points"] = temp_point
        response = autofil_response
        logger.info(
            f"Florence2 digitizer finished in: {time.time() - start_time:0.2f} sec"
        )

    elif not has_legend:
        logger.info("Plot image has no legend. Assigning data points randomly")
        start_time = time.time()
        response = await run_florence2_randomly(image_main, img_url, autofil_response)
        logger.info(
            f"Florence2 digitizer finished in: {time.time() - start_time:0.2f} sec"
        )

    else:
        chart_dete_fail, char_dete_response, chart_dete_out = await chart_dete_call(
            image_main, line_count
        )
        check_image = extract_plot_area(
            chart_dete_out=chart_dete_out, input_image=image_main.copy()
        )
        _, is_bw_image = find_dominant_color(check_image)
        if is_bw_image:
            logger.info(
                "Black and white image detected. Running line former based digitizer."
            )
            response = await line_former_digitizer(
                img_url=img_url, autofil_response=autofil_response
            )

        else:
            start_time = time.time()
            if chart_dete_fail:
                if chart_type == ChartType.SCATTER_PLOT:
                    logger.info(f"Chart type is: {chart_type}. Call Florence-2")
                    start_time = time.time()
                    response = await run_florence2_randomly(
                        image_main, img_url, autofil_response
                    )
                    logger.info(
                        f"Florence2 finished in: {time.time() - start_time:0.2f} sec"
                    )

                else:
                    logger.info("Calling LineFormer")
                    response = await line_former_digitizer(
                        img_url=img_url, autofil_response=autofil_response
                    )
            else:
                logger.info("Color image detected. Running Florence-2 based digitizer.")
                mapped_legend_patch, not_enough_line = await get_florence_2_points(
                    image_main, img_url, char_dete_response, line_count, chart_type
                )
                if not_enough_line:
                    logger.info("Not enough line from Florence2. Call LineFormer")
                    response = await line_former_digitizer(
                        img_url=img_url, autofil_response=autofil_response
                    )
                else:
                    response = await get_sentence_xfomer_mapping(
                        mapped_legend_patch, autofil_response
                    )
                    run_time = time.time() - start_time
                    logger.info(f"Florence2 digitizer finished in: {run_time:0.2f} sec")

    logger.info("Filtering overlapped points.")
    response = filter_points(response)
    return response
