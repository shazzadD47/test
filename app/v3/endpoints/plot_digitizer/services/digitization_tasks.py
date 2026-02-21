import time
import traceback

from app.v3.endpoints.plot_digitizer.constants import (
    CATEGORICAL_SKIP_PLOTS,
    DIGITIZATION_ERROR_MESSAGE,
    ERROR_BAR_SUPPORTED_PLOTS,
    ChartType,
)
from app.v3.endpoints.plot_digitizer.helpers import (
    PlotDigitizerHelper,
    image_resolution_check,
    rescale_plot_digitizer,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.ai_digitizer import (
    ai_digitizer,
    associate_lines_spider_plot,
)
from app.v3.endpoints.plot_digitizer.services.autofill_line_mapper import (
    AutofillAIMapperKaplan,
)
from app.v3.endpoints.plot_digitizer.services.categorical_axis_detection import (  # noqa: E501
    CategoricalAxisDetection,
)
from app.v3.endpoints.plot_digitizer.services.error_bar_detection import (
    add_error_bar_autofill_response,
)
from app.v3.endpoints.plot_digitizer.services.kaplain_meier_digitizer import (  # noqa: E501
    KaplainMeierDigitizer,
)
from app.v3.endpoints.plot_digitizer.services.spider_plot_digitizer import (
    SpiderPlotDigitizer,
)
from utils.file_ops import delete_bucket_object


def digitize_plot(
    figure_url: str,
    autofill_response: dict,
    chart_type: str,
    chart_supported: bool,
    x_is_categorical: bool,
):
    if not chart_supported:
        return autofill_response

    logger.info("Started AI plot digitizer.....")
    original_figure_url = figure_url
    is_resized_image = False
    start_time = time.time()

    if chart_type in {ChartType.KAPLAN_MEIER_CURVE}:
        logger.info(f"Plot type is: {chart_type}. Starting Kaplain Meier digitizer.")
        try:
            helper = PlotDigitizerHelper(image_url=original_figure_url)
            filtered_output = helper.get_filtered_plot_area()
        except Exception as e:
            logger.exception(f"Error occured when getting plot area filtered: {e}")
            filtered_output = {
                "is_plot_area_filtered": False,
                "image_url": original_figure_url,
                "object_key": None,
            }
        logger.info(f"Filtered image url: {filtered_output['image_url']}")
        kaplain_meier_digitizer = KaplainMeierDigitizer(
            img_url=filtered_output["image_url"]
        )
        digitized_output = kaplain_meier_digitizer.get_ai_digitized_points()
        line_mapper = AutofillAIMapperKaplan(
            autofill_response=autofill_response,
            ai_pd_output=digitized_output,
            image_url=filtered_output["image_url"],
        )
        response = line_mapper.get_mapped_autofill_response()

        if filtered_output["is_plot_area_filtered"]:
            try:
                delete_bucket_object(filtered_output["object_key"])
            except Exception as e:
                logger.exception(f"Error occured when deleting resized image: {e}")

    else:
        if chart_type == ChartType.SPIDER_PLOT:
            logger.info("Running digitizer for spider plot")
            try:
                spider_plot_digitizer = SpiderPlotDigitizer(
                    original_figure_url, autofill_response
                )
                response = spider_plot_digitizer.digitize_plot()

                logger.info("Associating lines for Spider plot...")
                response = associate_lines_spider_plot(response)
            except Exception as e:
                logger.exception(f"Error occured when digitizing spider plot: {e}")
                error_message = DIGITIZATION_ERROR_MESSAGE
                error_message += "Error occured when digitizing spider plot."
                raise Exception(error_message)

        else:
            try:
                resolution_check_output = image_resolution_check(figure_url)
            except Exception as e:
                logger.exception(f"Error when tried to check image resolution: {e}")
                trackeback = traceback.format_exc()
                logger.exception(f"Traceback: {trackeback}")
                error_message = DIGITIZATION_ERROR_MESSAGE
                error_message += "Error occured when checking image resolution."
                raise Exception(error_message)

            is_resized_image = resolution_check_output["is_resized_image"]
            if is_resized_image:
                figure_url = resolution_check_output["image_url"]

            try:
                response = ai_digitizer(
                    img_url=figure_url,
                    autofil_response=autofill_response,
                    chart_type=chart_type,
                )

            except Exception as e:
                logger.exception(f"Error occured when digitizing normal line plot: {e}")
                error_message = DIGITIZATION_ERROR_MESSAGE
                error_message += "Error occured when digitizing normal line plot."
                raise Exception(error_message)

            if is_resized_image:
                try:
                    delete_bucket_object(resolution_check_output["object_key"])
                except Exception as e:
                    logger.exception(f"Error occured when deleting resized image: {e}")
                try:
                    response = rescale_plot_digitizer(
                        resolution_check_output=resolution_check_output,
                        autofill_response=autofill_response,
                    )
                    if chart_type in ERROR_BAR_SUPPORTED_PLOTS:
                        response = add_error_bar_autofill_response(
                            autofill_response=autofill_response,
                            image_url=original_figure_url,
                        )
                except Exception as e:
                    logger.exception(
                        f"Error occured when rescaling plot digitizer: {e}"
                    )
                    error_message = DIGITIZATION_ERROR_MESSAGE
                    error_message += "Error occured when rescaling plot digitizer."
                    raise Exception(error_message)

    if chart_type != ChartType.BAR_PLOT:
        response = CategoricalAxisDetection.format_categorical_point(response)
    if x_is_categorical and chart_type not in CATEGORICAL_SKIP_PLOTS:
        categorical_axis_detector = CategoricalAxisDetection(
            response=response, image_url=original_figure_url
        )
        response = categorical_axis_detector.map_categorical_labels_to_points()

    run_time = time.time() - start_time
    logger.info(f"AI plot digitizer finished in {run_time:0.2f} seconds")
    response["runtime"] = {"digitization_runtime": run_time}

    return response
