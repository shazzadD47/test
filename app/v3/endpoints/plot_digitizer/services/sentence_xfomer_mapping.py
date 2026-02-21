from app.v3.endpoints.plot_digitizer.configs import settings
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.utils import (
    normalize_text,
    post_request_sentence_xformer,
)


def get_sentence_xfomer_mapping(mapped_legend_patch, autofil_response):
    ocr_legends = []
    for lines in mapped_legend_patch:
        ocr_legends.append(lines["lable_text"])
    logger.info(f"Extracted OCR legends: {ocr_legends}")

    autofill_legends = []
    list_line = autofil_response["data"]["lines"]
    for line in list_line:
        autofill_legends.append(line["labels"]["line_name"])

    for line in list_line:
        line["points"] = []

    sentence_xformer_payload = {
        "ocr_legends": ocr_legends,
        "autofill_legends": autofill_legends,
    }
    setence_xformer_out = post_request_sentence_xformer(
        post_url=settings.SENTENCE_TRANSFORMER_API, payload=sentence_xformer_payload
    )

    for mapped_line in setence_xformer_out["matched_list"]:
        ocr_legend, autofill_legend = list(mapped_line.items())[0]

        mapped_points = None
        for mapped_legend in mapped_legend_patch:
            if normalize_text(mapped_legend["lable_text"]) in normalize_text(
                ocr_legend
            ):
                mapped_points = mapped_legend["data_points"]
                break
        if mapped_points is not None:
            for line in list_line:
                ground_line_name = normalize_text(line["labels"]["line_name"])
                if ground_line_name in normalize_text(autofill_legend):
                    temp_point = []
                    for point in mapped_points:
                        temp_point.append(
                            {
                                "x": round(point[0]),
                                "y": round(point[1]),
                                "topBarPixelDistance": round(point[3]),
                                "bottomBarPixelDistance": round(point[4]),
                                "deviationPixelDistance": round(point[2]),
                            }
                        )
                    line["points"] = temp_point

    return autofil_response
