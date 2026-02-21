import time
import traceback
from uuid import uuid4

from celery import shared_task
from langfuse import observe

from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.core.usage.calculate_cost import calculate_total_cost
from app.core.utils.decorators.cost_tracker import track_all_llm_costs
from app.core.utils.decorators.helpers import combine_cost_metadatas_of_models
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints import Status
from app.v3.endpoints.general_extraction.services.tasks import (
    execute_general_extraction,
)
from app.v3.endpoints.plot_digitizer.configs import settings as pd_settings
from app.v3.endpoints.plot_digitizer.constants import (
    SUPPORTED_PLOTS_FOR_DIGITIZATION,
    ChartType,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.services.digitization_tasks import (
    digitize_plot,
)
from app.v3.endpoints.plot_digitizer.services.figure_details_tasks import (
    extract_plot_details,
)
from app.v3.endpoints.plot_digitizer.utils import (
    create_input_for_ge,
)
from utils.file_ops import delete_bucket_object


@shared_task(
    name="plot autofill dynamic metadata extraction task",
    bind=True,
    max_retries=0,
    default_retry_delay=10,
)
def dynamic_metadata_extraction_service(
    self,
    figure_url: list[str] | str,
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    page_no: int = None,
    bounding_box: dict = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    run_autofill: bool = True,
    run_digitization: bool = True,
    line_names_to_extract: list[dict] = None,
    generate_labels: list[str] = None,
    chart_type: str = None,
    metadata: dict = None,
    langfuse_session_id: str = None,
):
    # Initialize metadata if None
    if not metadata:
        metadata = {}

    retry_count = 0
    success = False

    # Track original flags and successful responses for partial retries
    original_run_autofill = run_autofill
    original_run_digitization = run_digitization
    successful_autofill_response = None
    successful_digitization_response = None

    while retry_count < pd_settings.MAX_RETRIES:
        try:
            output = extract_dynamic_metadata(
                figure_url=figure_url,
                paper_id=paper_id,
                project_id=project_id,
                table_structure=table_structure,
                page_no=page_no,
                bounding_box=bounding_box,
                legend_urls=legend_urls,
                bounding_box_legends=bounding_box_legends,
                run_autofill=run_autofill,
                run_digitization=run_digitization,
                line_names_to_extract=line_names_to_extract,
                generate_labels=generate_labels,
                chart_type=chart_type,
                metadata=metadata,
                langfuse_session_id=langfuse_session_id,
            )
            result = output.get("result", output)
            metadata = output.get("metadata", {})

            # Check autofill status
            if run_autofill:
                autofill_status = (
                    result.get("autofill_response", {})
                    .get("metadata", {})
                    .get("status", Status.FAILED.value)
                )
                if autofill_status == Status.SUCCESS.value:
                    successful_autofill_response = result.get("autofill_response")
            else:
                # Use previously successful response or mark as not applicable
                autofill_status = (
                    Status.SUCCESS.value
                    if successful_autofill_response
                    else Status.FAILED.value
                )

            # Check digitization status
            if run_digitization:
                digitization_status = (
                    result.get("digitization_response", {})
                    .get("metadata", {})
                    .get("status", Status.FAILED.value)
                )
                if digitization_status == Status.SUCCESS.value:
                    successful_digitization_response = result.get(
                        "digitization_response"
                    )
            else:
                # Use previously successful response or mark as not applicable
                digitization_status = (
                    Status.SUCCESS.value
                    if successful_digitization_response
                    else Status.FAILED.value
                )

            # Determine overall success based on original flags
            if original_run_autofill and original_run_digitization:
                autofill_succeeded = (
                    autofill_status == Status.SUCCESS.value
                    or successful_autofill_response is not None
                )
                digitization_succeeded = (
                    digitization_status == Status.SUCCESS.value
                    or successful_digitization_response is not None
                )
                success = autofill_succeeded and digitization_succeeded
            elif original_run_autofill:
                success = (
                    autofill_status == Status.SUCCESS.value
                    or successful_autofill_response is not None
                )
            elif original_run_digitization:
                success = (
                    digitization_status == Status.SUCCESS.value
                    or successful_digitization_response is not None
                )
            else:
                success = True

            if success:
                # Merge successful responses from current and previous attempts
                final_output = _merge_successful_responses(
                    output,
                    successful_autofill_response,
                    successful_digitization_response,
                    original_run_autofill,
                    original_run_digitization,
                )
                return final_output, success
            else:
                retry_count += 1
                retry_delay = pd_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
                time.sleep(retry_delay)
                if "metadata" in output and isinstance(output["metadata"], dict):
                    metadata.update(output["metadata"])
                if retry_count >= pd_settings.MAX_RETRIES:
                    # Merge any successful responses before returning
                    final_output = _merge_successful_responses(
                        output,
                        successful_autofill_response,
                        successful_digitization_response,
                        original_run_autofill,
                        original_run_digitization,
                    )
                    # Only send events for failed parts (successful parts already sent)
                    if original_run_autofill and successful_autofill_response is None:
                        send_to_backend(
                            BackendEventEnumType.PRESET_AUTOFILL_PLOT, final_output
                        )
                    if (
                        original_run_digitization
                        and successful_digitization_response is None
                    ):
                        send_to_backend(
                            BackendEventEnumType.PRESET_AUTOFILL_DIGITIZATION,
                            final_output,
                        )
                    return final_output, success

                # Determine which parts to retry - only retry failed parts
                autofill_succeeded = successful_autofill_response is not None
                digitization_succeeded = successful_digitization_response is not None

                if original_run_autofill and original_run_digitization:
                    # Only retry the failed part(s)
                    run_autofill = not autofill_succeeded
                    run_digitization = not digitization_succeeded
                # If only one was originally requested, keep retrying it
                continue
        except Exception as e:
            logger.info(e)
            logger.info(traceback.format_exc())
            retry_count += 1
            if retry_count >= pd_settings.MAX_RETRIES:
                error_response = {
                    "payload": {},
                    "metadata": {"status": Status.FAILED.value},
                }
                # Merge any successful responses from previous attempts
                final_output = _merge_successful_responses(
                    error_response,
                    successful_autofill_response,
                    successful_digitization_response,
                    original_run_autofill,
                    original_run_digitization,
                )
                # Only send events for failed parts (successful parts already sent)
                if original_run_autofill and successful_autofill_response is None:
                    send_to_backend(
                        BackendEventEnumType.PRESET_AUTOFILL_PLOT, final_output
                    )
                if (
                    original_run_digitization
                    and successful_digitization_response is None
                ):
                    send_to_backend(
                        BackendEventEnumType.PRESET_AUTOFILL_DIGITIZATION, final_output
                    )
                return final_output, success
            # Sleep before retry
            retry_delay = pd_settings.DEFAULT_RETRY_DELAY * (2**retry_count)
            time.sleep(retry_delay)

    # Fallback return if all retries are exhausted
    return {"payload": {}, "metadata": {"status": Status.FAILED.value}}, success


def _merge_successful_responses(
    current_output: dict,
    successful_autofill_response: dict | None,
    successful_digitization_response: dict | None,
    original_run_autofill: bool,
    original_run_digitization: bool,
) -> dict:
    """Merge successful responses from previous attempts with current output.

    This ensures that valid data is retained across retries by preferring
    previously successful responses over current failed or missing responses.
    """
    result = current_output.get("result", current_output)

    # Helper to check if a response is successful
    def _is_successful_response(response: dict | None) -> bool:
        if not response:
            return False
        return response.get("metadata", {}).get("status") == Status.SUCCESS.value

    # If we have a successful autofill from a previous attempt, use it
    # unless the current response is also successful
    current_autofill = result.get("autofill_response")
    if (
        successful_autofill_response is not None
        and original_run_autofill
        and not _is_successful_response(current_autofill)
    ):
        if "result" in current_output:
            current_output["result"]["autofill_response"] = successful_autofill_response
        else:
            current_output["autofill_response"] = successful_autofill_response

    # If we have a successful digitization from a previous attempt, use it
    # unless the current response is also successful
    current_digitization = result.get("digitization_response")
    if (
        successful_digitization_response is not None
        and original_run_digitization
        and not _is_successful_response(current_digitization)
    ):
        if "result" in current_output:
            current_output["result"][
                "digitization_response"
            ] = successful_digitization_response
        else:
            current_output["digitization_response"] = successful_digitization_response

    return current_output


@observe()
def extract_dynamic_metadata(
    figure_url: list[str] | str,
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    page_no: int = None,
    bounding_box: dict = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    run_autofill: bool = True,
    run_digitization: bool = True,
    line_names_to_extract: list[dict] = None,
    generate_labels: list[str] = None,
    chart_type: str = None,
    metadata: dict = None,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id, name="extract_dynamic_metadata")

    # Initialize metadata if None
    if not metadata:
        metadata = {}
    if chart_type and (run_autofill and not run_digitization):
        pass
    else:
        chart_type = None

    plot_details_with_metadata = extract_details(
        figure_url=figure_url,
        paper_id=paper_id,
        page_no=page_no,
        bounding_box=bounding_box,
        legend_urls=legend_urls,
        bounding_box_legends=bounding_box_legends,
        line_names_to_extract=line_names_to_extract,
        chart_type=chart_type,
        table_structure=table_structure,
        langfuse_session_id=langfuse_session_id,
    )

    if "result" in plot_details_with_metadata:
        plot_details = plot_details_with_metadata.get("result")
    else:
        plot_details = plot_details_with_metadata

    axis_result = plot_details["axis_result"]
    extracted_legends = plot_details["legends"]
    axis_result["legends"] = plot_details["legends"]
    legend_found = plot_details["legend_found"]
    figure_url = (
        plot_details["modified_image_url"]
        if plot_details["modified_image_url"] != "N/A"
        else figure_url
    )
    object_key = plot_details["object_key"]
    chart_type = axis_result["plot_axis_data"]["chart_type"]

    if line_names_to_extract:
        line_names_to_extract = {
            line["line_name"]: line["line_id"] for line in line_names_to_extract
        }
        extracted_legends = list(line_names_to_extract.keys())
    else:
        line_names_to_extract = {legend: f"{uuid4()}" for legend in extracted_legends}

    # if the chart type is spider-plot, then run digitization and find
    # number of lines in the plot. Then run autofill for a single line
    # and assign that to all the lines.
    chart_type = axis_result["plot_axis_data"]["chart_type"]
    if chart_type == ChartType.SPIDER_PLOT:
        run_digitization = True

    # run digitization
    digitization_legends = None
    if run_digitization:
        try:
            digitization_output = extract_digitization_data(
                figure_url,
                extracted_legends,
                axis_result,
                legend_found,
                line_names_to_extract,
                metadata,
                langfuse_session_id,
            )
            if "result" in digitization_output:
                digitization_response = digitization_output["result"][0]
                digitization_legends = digitization_output["result"][1]
                digitization_cost_metadata = (
                    digitization_output.get("metadata", {})
                    .get("ai_metadata", {})
                    .get("cost_metadata", {})
                )
                digitization_response = _assign_cost_metadata(
                    digitization_response, digitization_cost_metadata
                )
            else:
                digitization_response, digitization_legends = digitization_output
                digitization_cost_metadata = (
                    digitization_response.get("metadata", {})
                    .get("ai_metadata", {})
                    .get("cost_metadata", {})
                )

            if run_autofill:
                send_to_backend(
                    BackendEventEnumType.PRESET_AUTOFILL_DIGITIZATION,
                    digitization_response,
                )
            else:
                if "metadata" in plot_details_with_metadata:
                    plot_details_cost_metadata = (
                        plot_details_with_metadata["metadata"]
                        .get("ai_metadata", {})
                        .get("cost_metadata", {})
                    )
                    if len(plot_details_cost_metadata) > 0:
                        if len(digitization_cost_metadata) > 0:
                            pd_cost_details = plot_details_cost_metadata.get(
                                "llm_cost_details", {}
                            )
                            digitization_cost_details = digitization_cost_metadata.get(
                                "llm_cost_details", {}
                            )
                            digitization_cost_metadata = (
                                combine_cost_metadatas_of_models(
                                    [
                                        pd_cost_details,
                                        digitization_cost_details,
                                    ]
                                )
                            )
                            total_cost = calculate_total_cost(
                                digitization_cost_metadata
                            )
                            digitization_cost_metadata = {
                                k: v.model_dump()
                                for k, v in digitization_cost_metadata.items()
                            }
                            digitization_cost_metadata_with_total_cost = {
                                "total_cost": total_cost,
                                "llm_cost_details": digitization_cost_metadata,
                            }
                            digitization_response = _assign_cost_metadata(
                                digitization_response,
                                digitization_cost_metadata_with_total_cost,
                            )
                        else:
                            digitization_response = _assign_cost_metadata(
                                digitization_response, plot_details_cost_metadata
                            )
                send_to_backend(
                    BackendEventEnumType.PRESET_AUTOFILL_DIGITIZATION,
                    digitization_response,
                )
                if object_key != "N/A":
                    try:
                        delete_bucket_object(object_key)
                    except Exception as e:
                        logger.info(f"Error deleting, error: {e}")

                return {"digitization_response": digitization_response}

        except Exception as e:
            error_message = f"Error occured when extracting digitization plot: {e}"
            logger.exception(error_message)
            logger.exception(traceback.format_exc())
            digitization_response = {
                "payload": {},
                "metadata": {
                    "status": Status.FAILED.value,
                    "message": error_message,
                },
            }
            if not run_autofill:
                return {"digitization_response": digitization_response}

    # run autofill
    if run_autofill:
        try:
            if chart_type == ChartType.SPIDER_PLOT and digitization_legends is not None:
                extracted_legends = digitization_legends

            autofill_response = extract_autofill_data(
                paper_id=paper_id,
                project_id=project_id,
                table_structure=table_structure,
                generate_labels=generate_labels,
                figure_url=figure_url,
                extracted_legends=extracted_legends,
                axis_result=axis_result,
                line_names_to_extract=line_names_to_extract,
                metadata=metadata,
                langfuse_session_id=langfuse_session_id,
            )

            if object_key != "N/A":
                try:
                    delete_bucket_object(object_key)
                except Exception as e:
                    logger.info(f"Error deleting, error: {e}")

            if "metadata" in plot_details_with_metadata:
                plot_details_cost_metadata = (
                    plot_details_with_metadata["metadata"]
                    .get("ai_metadata", {})
                    .get("cost_metadata", {})
                )
                if len(plot_details_cost_metadata) > 0:
                    autofill_response_cost_metadata = (
                        autofill_response["metadata"]
                        .get("ai_metadata", {})
                        .get("cost_metadata", {})
                    )
                    if len(autofill_response_cost_metadata) > 0:
                        pd_cost_details = plot_details_cost_metadata.get(
                            "llm_cost_details", {}
                        )
                        autofill_cost_details = autofill_response_cost_metadata.get(
                            "llm_cost_details", {}
                        )
                        combined_cost_metadata = combine_cost_metadatas_of_models(
                            [
                                pd_cost_details,
                                autofill_cost_details,
                            ]
                        )
                        total_cost = calculate_total_cost(combined_cost_metadata)
                        cost_metadata = {
                            k: v.model_dump() for k, v in combined_cost_metadata.items()
                        }
                        cost_metadata_with_total_cost = {
                            "total_cost": total_cost,
                            "llm_cost_details": cost_metadata,
                        }
                        autofill_response["metadata"]["ai_metadata"][
                            "cost_metadata"
                        ] = cost_metadata_with_total_cost
                    else:
                        autofill_response["metadata"]["ai_metadata"][
                            "cost_metadata"
                        ] = plot_details_cost_metadata

            send_to_backend(
                BackendEventEnumType.PRESET_AUTOFILL_PLOT,
                autofill_response,
            )
            if run_digitization:
                return {
                    "digitization_response": digitization_response,
                    "autofill_response": autofill_response,
                }
            return {"autofill_response": autofill_response}
        except Exception as e:
            error_message = f"Error occured when extracting autofill plot: {e}"
            logger.exception(error_message)
            logger.exception(traceback.format_exc())
            autofill_response = {
                "payload": {},
                "metadata": {
                    "status": Status.FAILED.value,
                    "message": error_message,
                },
            }
            if run_digitization:
                return {
                    "digitization_response": digitization_response,
                    "autofill_response": autofill_response,
                }
            return {"autofill_response": autofill_response}

    return {
        "autofill_response": {},
        "digitization_response": {},
    }


def _assign_cost_metadata(response: dict, cost_metadata: dict) -> dict:
    if (
        "metadata" in response
        and isinstance(response["metadata"], dict)
        and "ai_metadata" in response["metadata"]
        and isinstance(response["metadata"]["ai_metadata"], dict)
    ):
        response["metadata"]["ai_metadata"]["cost_metadata"] = cost_metadata
    elif "metadata" in response and isinstance(response["metadata"], dict):
        response["metadata"]["ai_metadata"] = {"cost_metadata": cost_metadata}
    else:
        response["metadata"] = {"ai_metadata": {"cost_metadata": cost_metadata}}
    return response


@observe()
@track_all_llm_costs
def extract_details(
    figure_url: str,
    paper_id: str,
    page_no: int = None,
    bounding_box: dict = None,
    legend_urls: list[str] = None,
    bounding_box_legends: list[dict] = None,
    line_names_to_extract: list[dict] = None,
    chart_type: str = None,
    table_structure: list[dict] = None,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id)
    start_time = time.time()
    plot_details = extract_plot_details(
        figure_url=figure_url,
        paper_id=paper_id,
        page_no=page_no,
        bounding_box=bounding_box,
        legend_urls=legend_urls,
        bounding_box_legends=bounding_box_legends,
        chart_type=chart_type,
        line_names_to_extract=line_names_to_extract,
        table_structure=table_structure,
        langfuse_session_id=langfuse_session_id,
    )
    run_time = time.time() - start_time
    logger.info(f"Plot details extraction runtime: {run_time:0.2f} seconds")
    return plot_details


@observe()
@track_all_llm_costs
def extract_digitization_data(
    figure_url: str,
    extracted_legends: list[str],
    axis_result: dict,
    legend_found: bool,
    line_names_to_extract: list[dict],
    metadata: dict,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id)
    start_time = time.time()
    autofill_response = {
        "lines": [{"labels": {"line_name": legend}} for legend in extracted_legends]
    }
    merged_result = {
        "data": {
            **axis_result,
            **autofill_response,
            "has_legend": legend_found,
        }
    }
    chart_type = axis_result["plot_axis_data"]["chart_type"]
    chart_supported = chart_type in SUPPORTED_PLOTS_FOR_DIGITIZATION
    digitization_response = digitize_plot(
        figure_url,
        merged_result,
        axis_result["plot_axis_data"]["chart_type"],
        chart_supported,
        axis_result["plot_axis_data"]["x_is_categorical"],
    )

    if chart_type == ChartType.SPIDER_PLOT:
        extracted_legends = [
            line["labels"]["line_name"]
            for line in digitization_response["data"]["lines"]
        ]
        digitization_response["data"]["legends"] = extracted_legends

    if len(line_names_to_extract) > 0:
        updated_lines = []
        for line in digitization_response["data"]["lines"]:
            line_name = line["labels"]["line_name"]
            if line_name in line_names_to_extract:
                line["id"] = line_names_to_extract[line_name]
                updated_lines.append(line)

        digitization_response["data"]["lines"] = updated_lines
    else:
        for line in digitization_response["data"]["lines"]:
            line_names_to_extract[line["labels"]["line_name"]] = line["id"]

    run_time = time.time() - start_time
    logger.info(f"Digitization runtime: {run_time:0.2f} seconds")

    metadata["message"] = "Plot digitization finished successfully"
    metadata["status"] = Status.SUCCESS.value
    digitization_response = {"payload": digitization_response, "metadata": metadata}
    return digitization_response, extracted_legends


@observe()
@track_all_llm_costs
def extract_autofill_data(
    paper_id: str,
    project_id: str,
    table_structure: list[dict],
    generate_labels: list[str],
    figure_url: str,
    extracted_legends: list[str],
    axis_result: dict,
    line_names_to_extract: list[dict],
    metadata: dict,
    langfuse_session_id: str = None,
):
    setup_langfuse_handler(langfuse_session_id)
    start_time = time.time()
    chart_type = axis_result["plot_axis_data"]["chart_type"]
    if chart_type == ChartType.SPIDER_PLOT:
        original_extracted_legends = extracted_legends.copy()
        extracted_legends = ["line_1"]
    else:
        original_extracted_legends = extracted_legends.copy()

    ge_input = create_input_for_ge(
        paper_id,
        project_id,
        table_structure,
        figure_url,
        extracted_legends,
        generate_labels,
        chart_type,
    )
    logger.info(f"GE input: {ge_input}")
    ge_response = execute_general_extraction(ge_input)

    # Check if GE response was successful
    if not (
        "metadata" in ge_response
        and "status" in ge_response["metadata"]
        and ge_response["metadata"]["status"] == Status.SUCCESS.value
    ):
        error_message = ge_response.get("metadata", {}).get(
            "message", "General extraction failed"
        )
        logger.error(f"General extraction failed: {error_message}")
        metadata["message"] = error_message
        metadata["status"] = Status.FAILED.value
        return {
            "payload": {},
            "metadata": metadata,
        }

    autofill_response = convert_ge_output_to_paf_format(
        ge_response["payload"],
        chart_type,
        original_extracted_legends,
        line_names_to_extract,
    )
    if (
        "ai_metadata" in ge_response["metadata"]
        and "result_with_citations" in ge_response["metadata"]["ai_metadata"]
    ):
        autofill_response_with_citations = convert_ge_output_to_paf_format(
            ge_response["metadata"]["ai_metadata"]["result_with_citations"],
            chart_type,
            original_extracted_legends,
            line_names_to_extract,
        )
        if "ai_metadata" in metadata:
            metadata["ai_metadata"][
                "result_with_citations"
            ] = autofill_response_with_citations
        else:
            metadata["ai_metadata"] = {
                "result_with_citations": autofill_response_with_citations
            }

    metadata["message"] = "Plot Autofill finished successfully."
    metadata["status"] = Status.SUCCESS.value
    autofill_response = {
        "payload": {"data": autofill_response},
        "metadata": metadata,
    }
    run_time = time.time() - start_time
    logger.info(f"Autofill runtime: {run_time:0.2f} seconds")

    return autofill_response


def convert_ge_output_to_paf_format(
    ge_response: list[dict],
    chart_type: str,
    original_extracted_legends: list[str],
    line_names_to_extract: list[str],
):
    autofill_response = {
        "lines": [{"labels": single_line_label} for single_line_label in ge_response]
    }

    if chart_type == ChartType.SPIDER_PLOT:
        autofill_label_response = autofill_response["lines"][0]["labels"]
        updated_lines = []
        for legend in original_extracted_legends:
            line_response_for_legend = autofill_label_response.copy()
            line_response_for_legend["line_name"] = legend
            line_info = {
                "labels": line_response_for_legend,
                "id": line_names_to_extract[legend],
            }
            updated_lines.append(line_info)
        autofill_response["lines"] = updated_lines
    else:
        updated_lines = []
        for line in autofill_response["lines"]:
            line_name = line["labels"]["line_name"]
            if line_name in line_names_to_extract:
                line["id"] = line_names_to_extract[line_name]
                updated_lines.append(line)

        autofill_response["lines"] = updated_lines

    return autofill_response
