from fastapi import APIRouter, Query

from app.v3.endpoints.plot_digitizer.constants import AxisMinMax
from app.v3.endpoints.plot_digitizer.logging import logger
from app.v3.endpoints.plot_digitizer.schemas import (
    AxesMinMaxOutput,
    DynamicPlotDigitizerRequest,
)
from app.v3.endpoints.plot_digitizer.services import (
    end_to_end_autofill_digitizer,
)
from app.v3.endpoints.plot_digitizer.services.axes_min_max_detection import (  # noqa: E501
    chart_dete_min_max,
)
from app.v3.endpoints.plot_digitizer.services.axis_value_extraction import (
    AxisValueExtractor,
)

router = APIRouter(tags=["plot_digitization"])


@router.get("/plot-digitizer/detect-axes-origin-min-max")
async def axes_min_max(
    figure_url: str = Query(..., example=AxisMinMax.example_url)
) -> AxesMinMaxOutput | None:
    min_max_out = None
    try:
        logger.debug("Running axes detection using ChartDete")
        min_max_out = await chart_dete_min_max(figure_url)
    except Exception as e:
        logger.debug(f"ChartDete failed for axes detection: {e}.")

    return min_max_out


@router.post("/plot-digitizer/metadata/dynamic-celery")
async def get_dynamic_plot_metadata(
    data: DynamicPlotDigitizerRequest,
):
    logger.info(f"Received payload: {data.model_dump_json()}")
    if data.metadata:
        generate_labels = data.metadata.get("generate_labels")
        chart_type = data.metadata.get("chart_type")
    else:
        generate_labels = None
        chart_type = None
    return await end_to_end_autofill_digitizer(
        data.payload.figure_url,
        data.payload.paper_id,
        data.payload.project_id or "",
        data.payload.table_structure,
        data.payload.page_number,
        data.payload.bounding_box,
        data.payload.legend_urls,
        data.payload.bounding_box_legends,
        data.payload.run_autofill,
        data.payload.run_digitization,
        data.payload.line_names_to_extract,
        generate_labels,
        chart_type,
        data.metadata,
    )


@router.get("/plot-digitizer/detect-axes-origin-min-max-values")
async def merged_axes_min_max_values(
    figure_url: str = Query(..., example=AxisMinMax.example_url)
) -> dict:

    try:

        pixel_out = await chart_dete_min_max(figure_url)

        if pixel_out is None:
            return {"error": "chart_dete_min_max() returned None"}

        pixel_points = pixel_out.get("points", [])

        # Convert list → dict for easy lookup
        pixel_map = {p["label"]: p for p in pixel_points}

        # 2️⃣ Get numeric min/max values
        extractor = AxisValueExtractor(figure_url)
        numeric = extractor.get_axis_values()
        mapping = {
            "xmin": numeric["x_axis"]["min_val"],
            "xmax": numeric["x_axis"]["max_val"],
            "ymin": numeric["y_axis"]["min_val"],
            "ymax": numeric["y_axis"]["max_val"],
        }

        # 3️⃣ Merge pixel coordinates with numeric values
        merged_points = []
        for label, value in mapping.items():
            if label in pixel_map:
                p = pixel_map[label]
                merged_points.append(
                    {
                        "label": p["label"],
                        "x": p["x"],
                        "y": p["y"],
                        "value": float(value),
                    }
                )

        return {"points": merged_points}

    except Exception as e:
        logger.error(f"Failed to generate merged output: {e}")
        return {"error": str(e)}
