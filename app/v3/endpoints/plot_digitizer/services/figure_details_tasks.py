import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langfuse import observe
from pydantic import Field

from app.configs import settings
from app.utils.image import convert_image_to_base64, get_image_from_url
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.get_title_summery.utils.utils import secure_file_path
from app.v3.endpoints.plot_digitizer.constants import (
    AUTOFILL_ERROR_MESSAGE,
    MAX_RETRIES,
    SEPARATOR,
    SPIDER_PLOT_DEF,
    ChartType,
)
from app.v3.endpoints.plot_digitizer.helpers import (
    create_ref_instructions,
    get_descriptions_from_schema,
    get_plot_structure_details,
    get_questions_from_schema,
    remove_saved_files,
    upload_files_with_retries,
)
from app.v3.endpoints.plot_digitizer.langchain_schemas import (
    AxisDetails,
    PlotDetails,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.legend_extraction import (
    extract_legends_from_figure,
)
from app.v3.endpoints.plot_digitizer.services.matching_figure import (
    find_matching_figure,
)
from app.v3.endpoints.plot_digitizer.services.rag_tasks import (
    rephrase_question,
)
from app.v3.endpoints.plot_digitizer.utils import (
    combine_images_vertically_with_padding,
)


@observe()
def extract_plot_details(
    figure_url: list[str] | str,
    paper_id: str,
    page_no: int = None,
    bounding_box: dict = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    chart_type: str = None,
    line_names_to_extract: list[dict] = None,
    table_structure: list[dict] = None,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id)

    # if extracted data has legends in them, then skip extracting
    # legends.
    extract_legends = not (
        line_names_to_extract
        and isinstance(line_names_to_extract, list)
        and len(line_names_to_extract) > 0
    )

    extract_axis_details = not chart_type

    start_time = time.time()

    original_image_path, original_image, media_type = _download_and_save_image(
        figure_url
    )
    image = convert_image_to_base64(original_image)

    if extract_axis_details:
        # first find the matching figure in database
        try:
            figure_details, paths = find_matching_figure(
                paper_id=paper_id,
                page_no=page_no,
                bounding_box=bounding_box,
            )
        except Exception as e:
            error_message = f"Error when tried to find matching figure. Error: {e}"
            logger.info(error_message)
            figure_details, paths = {}, {}

        paths["original_image_path"] = original_image_path

        axis_result = _extract_plot_structure_details(
            image=image,
            media_type=media_type,
            figure_details=figure_details,
            table_structure=table_structure,
            langfuse_session_id=langfuse_session_id,
        )
    else:
        axis_result = {
            "plot_axis_data": {
                "chart_type": chart_type,
            }
        }
        paths = {
            "original_image_path": original_image_path,
        }

    plot_chart_type = "N/A"
    if extract_legends:
        plot_chart_type = axis_result["plot_axis_data"]["chart_type"]

        if plot_chart_type == ChartType.SPIDER_PLOT:
            extracted_legends = []
            legend_found, legend_found_from_legends_patch = False, False
            modified_image_url, object_key = "N/A", "N/A"
        else:
            (
                extracted_legends,
                legend_found,
                legend_found_from_legends_patch,
                paths,
                image,
                modified_image_url,
                object_key,
            ) = _extract_legends_from_plot(
                image=image,
                original_image=original_image,
                media_type=media_type,
                bounding_box=bounding_box,
                legend_urls=legend_urls,
                bounding_box_legends=bounding_box_legends,
                paths=paths,
                chart_type=plot_chart_type,
                langfuse_session_id=langfuse_session_id,
            )
    else:
        extracted_legends = [
            line_name["line_name"] for line_name in line_names_to_extract
        ]
        if legend_urls and bounding_box_legends:
            image, paths, modified_image_url, object_key = (
                _combine_image_with_legend_image(
                    image=image,
                    bounding_box=bounding_box,
                    legend_url=legend_urls[0],
                    bounding_box_legend=bounding_box_legends[0],
                    paths=paths,
                )
            )
        else:
            modified_image_url, object_key = "N/A", "N/A"

        legend_found, legend_found_from_legends_patch = True, True

    logger.info(f"Plot details extracted in {time.time() - start_time} seconds.")

    return {
        "axis_result": axis_result,
        "legends": extracted_legends,
        "legend_found": legend_found,
        "legend_found_from_legends_patch": legend_found_from_legends_patch,
        "paths": paths,
        "image": image,
        "modified_image_url": modified_image_url,
        "object_key": object_key,
    }


def _download_and_save_image(url: str) -> tuple[str, Path, str]:
    # download the passed image in input and convert it to base64
    start_time = time.time()
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            image, media_type = get_image_from_url(url, return_media_type=True)
            break
        except Exception as e:
            retry_count += 1
            error_message = f"An error occurred while downloading image. Error: {e}"
            if retry_count == MAX_RETRIES:
                error_message = AUTOFILL_ERROR_MESSAGE + error_message
                raise Exception(error_message)
            continue

    original_image = bytes(image)
    save_path = secure_file_path(base_dir="./", filename=f"{uuid4()}.png")
    with open(save_path, "wb") as fo:
        fo.write(original_image)
    logger.info(f"Image retrieved in {time.time() - start_time} seconds.")

    return save_path, original_image, media_type


@observe()
def _extract_plot_structure_details(
    image: bytes,
    media_type: str,
    figure_details: dict = None,
    table_structure: list[dict] = None,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id)

    # extract figure data (like axis) from figure
    start_time = time.time()

    # edit ref fields so that it only chooses among
    # the table structure categories
    column_names_with_n_a_x = [v["name"] for v in table_structure] + ["x"]
    column_names_with_n_a_y = [v["name"] for v in table_structure] + ["y"]

    # Create a dynamic PlotDetails class with the new fields
    class AxisDetailsWithRef(AxisDetails):
        x_ref: Literal[*column_names_with_n_a_x] = Field(
            ...,
            description="The label of the x axis after mapping to a category.",
        )
        y_ref: Literal[*column_names_with_n_a_y] = Field(
            ...,
            description="The label of the y axis after mapping to a category.",
        )

    # Use the dynamic class instead of modifying the original
    class PlotDetailsWithRef(PlotDetails):
        plot_axis_data: AxisDetailsWithRef = Field(
            ...,
            description="The details of the plot axis",
        )

    # first generate questions from schema
    questions_from_axis_details = get_questions_from_schema(
        PlotDetailsWithRef,
    )
    questions_from_axis_details = rephrase_question(
        questions=questions_from_axis_details["questions"],
        keys=questions_from_axis_details["keys"],
    )

    questions_from_axis_details = [
        v for query in questions_from_axis_details for _, v in query.items()
    ]
    questions_from_axis_details = [
        f"{count+1}. {question}"
        for count, question in enumerate(questions_from_axis_details)
    ]

    # add instructions for selecting axis reference and to
    # detect spider plots to the list of questions
    x_ref_instructions = create_ref_instructions(table_structure, "x axis")
    y_ref_instructions = create_ref_instructions(table_structure, "y axis")

    # find after whcih question to add the instructions
    descriptions = get_descriptions_from_schema(PlotDetailsWithRef)
    ref_keys = ["x_ref", "y_ref"]
    for index, key in enumerate(descriptions):
        if key in ref_keys:
            if key == "x_ref":
                questions_from_axis_details[index] += f"\n{x_ref_instructions}"
            else:
                questions_from_axis_details[index] += f"\n{y_ref_instructions}"

    chart_type_key = "chart_type"
    for index, key in enumerate(descriptions):
        if key == chart_type_key:
            questions_from_axis_details[index] += f"\n{SPIDER_PLOT_DEF}"
            break

    questions_from_axis_details = SEPARATOR.join(questions_from_axis_details)

    logger.info(
        f"""Questions from schema axis details generated in
        {time.time() - start_time} seconds."""
    )

    # then get figure details from image
    start_time = time.time()
    if figure_details and isinstance(figure_details, dict):
        figure_details_for_prompt = [
            f"{key}: {value}" for key, value in figure_details.items()
        ]
        figure_details_for_prompt = SEPARATOR.join(figure_details_for_prompt)
    else:
        figure_details_for_prompt = ""

    axis_result = get_plot_structure_details(
        image=image,
        media_type=media_type,
        questions=questions_from_axis_details,
        figure_data=figure_details_for_prompt,
        schema=PlotDetailsWithRef,
        langfuse_session_id=langfuse_session_id,
    )
    logger.info(f"Axis details retrieved in {time.time() - start_time} seconds.")

    return axis_result


@observe()
def _extract_legends_from_plot(
    image: bytes,
    original_image: bytes,
    media_type: str,
    bounding_box: dict = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    paths: dict = None,
    langfuse_session_id: str = None,
    chart_type: str = "N/A",
):
    # extract legends from image
    (
        extracted_legends,
        legend_found,
        legend_found_from_legends_patch,
        found_legend_in_original_image,
        paths,
        image,
    ) = extract_legends_from_figure(
        image=image,
        original_image=original_image,
        media_type=media_type,
        legend_urls=legend_urls,
        bounding_box_legends=bounding_box_legends,
        bounding_box=bounding_box,
        paths=paths,
        chart_type=chart_type,
        langfuse_session_id=langfuse_session_id,
    )

    if legend_found_from_legends_patch and not found_legend_in_original_image:
        upload_path = f"plot-autofill-temp-bucket-{uuid4()}"
        upload_files_with_retries(
            paths["image_with_legends_path"],
            upload_path,
        )
        object_key = (
            f"{upload_path}" f"/{os.path.basename(paths['image_with_legends_path'])}"
        )

        modified_image_url = f"{settings.S3_SPACES_PUBLIC_BASE_URL}" f"/{object_key}"
    else:
        modified_image_url, object_key = "N/A", "N/A"

    remove_saved_files(paths)

    return (
        extracted_legends,
        legend_found,
        legend_found_from_legends_patch,
        paths,
        image,
        modified_image_url,
        object_key,
    )


def _combine_image_with_legend_image(
    image: bytes,
    bounding_box: dict,
    legend_url: str,
    bounding_box_legend: dict,
    paths: dict = None,
):
    legend_image = get_image_from_url(legend_url)
    user_cropped_image_path = secure_file_path(base_dir="./", filename=f"{uuid4()}.png")
    if "user_cropped_image_paths" not in paths:
        paths["user_cropped_image_paths"] = []
    paths["user_cropped_image_paths"].append(user_cropped_image_path)
    with open(user_cropped_image_path, "wb") as fo:
        fo.write(legend_image)
    image = combine_images_vertically_with_padding(
        image1_path=paths["original_image_path"],
        image2_path=user_cropped_image_path,
        bounding_box_legend=bounding_box_legend,
        bounding_box=bounding_box,
    )
    image_bytes = bytes(image)
    image = convert_image_to_base64(image)
    img_with_legends_path = secure_file_path(base_dir="./", filename=f"{uuid4()}.png")
    paths["image_with_legends_path"] = img_with_legends_path
    with open(paths["image_with_legends_path"], "wb") as fo:
        fo.write(image_bytes)

    upload_path = f"plot-autofill-temp-bucket-{uuid4()}"
    upload_files_with_retries(
        paths["image_with_legends_path"],
        upload_path,
    )
    object_key = (
        f"{upload_path}" f"/{os.path.basename(paths['image_with_legends_path'])}"
    )

    modified_image_url = f"{settings.S3_SPACES_PUBLIC_BASE_URL}" f"/{object_key}"

    remove_saved_files(paths)

    return image, paths, modified_image_url, object_key
