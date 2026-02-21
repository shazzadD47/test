from typing import Any

from app.v3.endpoints.general_extraction.prompts.commons import (
    END_OF_MEDIA_FILES_PROMPT,
    START_OF_MEDIA_FILE_PROMPT,
    START_OF_MEDIA_FILES_PROMPT,
)
from app.v3.endpoints.general_extraction.prompts.numerical import (
    NUMERICAL_START_OF_MEDIA_FILES_PROMPT,
)
from app.v3.endpoints.general_extraction.services.helpers.analysis_helpers import (
    return_image_with_legends,
)


def create_media_inputs(
    flag_id: str,
    inputs: dict,
    for_numerical_labels: bool = False,
) -> list[dict[str, Any]]:
    contents = []
    count = 0

    if "inputs" not in inputs or inputs["inputs"] is None or len(inputs["inputs"]) == 0:
        return contents

    for item in inputs["inputs"]:
        if item.get("type") in ["image", "chart", "table", "equation"]:
            for item_data in item.get("data"):
                if item_data.get("image_base64") is None:
                    image, media_type = return_image_with_legends(item_data)
                    item_data["image_base64"] = image
                    item_data["media_type"] = media_type
                else:
                    image = item_data.get("image_base64")
                    media_type = item_data.get("media_type")
                identifier = item_data.get("identifier")

                count += 1

                contents.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                START_OF_MEDIA_FILE_PROMPT.format(
                                    identifier=identifier, index=count
                                )
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{media_type};base64,{image}"),
                                "detail": "high",
                            },
                        },
                    ]
                )
        else:
            count += 1
            identifier = item.get("name", f"text_input_{count}")
            if item.get("data") is not None:
                contents.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                START_OF_MEDIA_FILE_PROMPT.format(
                                    identifier=identifier, index=count
                                )
                            ),
                        },
                        {"type": "text", "text": item.get("data")},
                    ]
                )

    if count > 0:
        contents = (
            [
                {
                    "type": "text",
                    "text": (
                        NUMERICAL_START_OF_MEDIA_FILES_PROMPT.format(flag_id=flag_id)
                        if for_numerical_labels
                        else START_OF_MEDIA_FILES_PROMPT.format(flag_id=flag_id)
                    ),
                }
            ]
            + contents
            + [{"type": "text", "text": END_OF_MEDIA_FILES_PROMPT}]
        )

    return contents
