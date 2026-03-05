import time
from uuid import uuid4

from langfuse import observe

from app.utils import check_if_null
from app.utils.image import convert_image_to_base64, get_image_from_url
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.get_title_summery.utils.utils import secure_file_path
from app.v3.endpoints.plot_digitizer.helpers import (
    check_bounding_box_inside,
    check_if_dummy_legend,
    get_plot_legends,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.utils import (
    combine_images_vertically_with_padding,
)


@observe()
def extract_legends_from_figure(
    image: str,
    original_image: str,
    media_type: str,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    bounding_box: dict = None,
    paths: dict = None,
    chart_type: str = "N/A",
    langfuse_session_id: str = None,
) -> tuple[list[str], bool, bool, dict, str]:
    setup_langfuse_handler(langfuse_session_id)

    start_time = time.time()
    legend_found = False
    legend_found_from_legends_patch = False
    found_legend_in_original_image = False
    extracted_legends = []
    legends_from_first_cropped_image = []
    image_bytes = None
    original_image_str = convert_image_to_base64(original_image)

    # First, try to extract legends from legend_urls if present
    if legend_urls and bounding_box_legends is not None:
        logger.info("Extracting legends from legend_urls first")
        for idx, legend_url in enumerate(legend_urls):
            # combine original image with user cropped image
            image, media_type = get_image_from_url(legend_url, return_media_type=True)
            user_cropped_image_path = secure_file_path(
                base_dir="./", filename=f"{uuid4()}.png"
            )
            if "user_cropped_image_paths" not in paths:
                paths["user_cropped_image_paths"] = []
            paths["user_cropped_image_paths"].append(user_cropped_image_path)
            with open(user_cropped_image_path, "wb") as fo:
                fo.write(image)
            bounding_box_legend = bounding_box_legends[idx]

            is_legend_image_inside_plot = check_bounding_box_inside(
                bounding_box, bounding_box_legend
            )
            if not is_legend_image_inside_plot:
                image = combine_images_vertically_with_padding(
                    image1_path=paths["original_image_path"],
                    image2_path=user_cropped_image_path,
                    bounding_box_legend=bounding_box_legend,
                    bounding_box=bounding_box,
                )
                image_bytes = bytes(image)
                image = convert_image_to_base64(image)
            else:
                logger.info("Lengend image inside plot, skipping legend concatenation")
                image_bytes = bytes(original_image)
                image = original_image_str
                found_legend_in_original_image = True

            legend_result = get_plot_legends(
                image=image,
                media_type=media_type,
                chart_type=chart_type,
                langfuse_session_id=langfuse_session_id,
            )
            logger.info(
                f"Legends from cropped legend figures: {legend_result['legends']}"
            )
            extracted_legends_from_figure = legend_result["legends"]
            # Store first result as fallback
            if idx == 0:
                legends_from_first_cropped_image = extracted_legends_from_figure

            if not (
                len(extracted_legends_from_figure) == 0
                or (
                    all(
                        check_if_null(legend)
                        for legend in extracted_legends_from_figure
                    )
                )
                or check_if_dummy_legend(extracted_legends_from_figure)
            ):
                legend_found = True
                extracted_legends = extracted_legends_from_figure
                logger.info("Legends found from cropped legend figures")
                legend_found_from_legends_patch = True
                img_with_legends_path = secure_file_path(
                    base_dir="./", filename=f"{uuid4()}.png"
                )
                paths["image_with_legends_path"] = img_with_legends_path
                with open(paths["image_with_legends_path"], "wb") as fo:
                    fo.write(image_bytes)
                logger.info(f"Image+legend saved {paths['image_with_legends_path']}")

            if legend_found:
                break

    # If no legends found yet (either no legend_urls or legend_urls failed),
    # fall back to extracting from original image
    if not legend_found:
        logger.info("Falling back to extracting legends from original image")
        legend_result = get_plot_legends(
            image=original_image_str,
            media_type=media_type,
            chart_type=chart_type,
            langfuse_session_id=langfuse_session_id,
        )
        extracted_legends_from_figure = legend_result["legends"]
        # Only set as fallback if we don't already have one from legend_urls
        if not legends_from_first_cropped_image:
            legends_from_first_cropped_image = extracted_legends_from_figure
        logger.info(f"Legends from original image: {extracted_legends_from_figure}")

        if not (
            len(extracted_legends_from_figure) == 0
            or (all(check_if_null(legend) for legend in extracted_legends_from_figure))
            or check_if_dummy_legend(extracted_legends_from_figure)
        ):
            legend_found = True
            extracted_legends = extracted_legends_from_figure
            image = original_image_str
            logger.info("Legends found from original image")

    if not legend_found and "legend_paths" in paths:
        for legend_path in paths["legend_paths"]:
            image = combine_images_vertically_with_padding(
                image1_path=paths["original_image_path"],
                image2_path=legend_path,
                bounding_box_legend=None,
                bounding_box=None,
            )
            logger.info("Extracting legends from combined image")
            image_base64 = convert_image_to_base64(image)
            legend_result = get_plot_legends(
                image=image_base64,
                media_type=media_type,
                chart_type=chart_type,
                langfuse_session_id=langfuse_session_id,
            )
            extracted_legends_from_figure = legend_result["legends"]

            logger.info(
                f"Legend from db legend patches: {extracted_legends_from_figure}"
            )
            if not (
                len(extracted_legends_from_figure) == 0
                or (
                    all(
                        check_if_null(legend)
                        for legend in extracted_legends_from_figure
                    )
                )
                or check_if_dummy_legend(extracted_legends_from_figure)
            ):
                legend_found = True
                legend_found_from_legends_patch = True
                extracted_legends = extracted_legends_from_figure
                logger.info("Legends found from found db legend patches")
                paths["image_with_legends_path"] = secure_file_path(
                    base_dir="./", filename=f"{uuid4()}.png"
                )
                with open(paths["image_with_legends_path"], "wb") as fo:
                    fo.write(image)
                logger.info(f"Image+legend saved {paths['image_with_legends_path']}")
                image = image_base64

            if legend_found:
                break

    # legend not found from db or user cropped image,
    # then use legends from first user cropped image
    if not legend_found:
        logger.info("Legends not found in other cropped figs or db")
        logger.info(
            f"Legends set to 1st cropped fig: {legends_from_first_cropped_image}"
        )
        extracted_legends = legends_from_first_cropped_image
        image = convert_image_to_base64(original_image)

    # if no legends found, set to line_1 for dummy legend
    if len(extracted_legends) == 0 or (
        all(check_if_null(legend) for legend in extracted_legends)
    ):
        extracted_legends = ["line_1"]

    logger.info(f"Legends: {extracted_legends}")
    logger.info(f"Legends retrieved in {time.time() - start_time} seconds.")

    return (
        extracted_legends,
        legend_found,
        legend_found_from_legends_patch,
        found_legend_in_original_image,
        paths,
        image,
    )
