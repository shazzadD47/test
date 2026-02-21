import base64

import cv2
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.prompts import (
    PROMPT_AREA_COMPARE,
    PROMPT_LEGEND_DETECTION,
)
from app.v3.endpoints.get_title_summery.schemas import AreaCompare, DetectLegends


def legend_map(img_paths: list, client: ChatAnthropic):
    content_list = []
    for img_path in img_paths:
        _, buffer = cv2.imencode(".png", img_path)
        img_path = base64.b64encode(buffer).decode("utf-8")

        json_obj = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_path,
            },
        }
        content_list.append(json_obj)

    text_json = {"type": "text", "text": PROMPT_LEGEND_DETECTION}
    content_list.append(text_json)
    structured_client = client.with_structured_output(DetectLegends)
    human_message = HumanMessage(content=content_list)
    try:
        message = structured_client.invoke([human_message])
        logger.debug(f"new message is {message}")
    except Exception as e:
        logger.exception(f"processing failed for  {str(e)}")

    return message


def area_compare(img_paths, client: ChatAnthropic):
    content_list = []
    for img_path in img_paths:
        _, buffer = cv2.imencode(".png", img_path)
        img_path = base64.b64encode(buffer).decode("utf-8")

        json_obj = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_path,
            },
        }
        content_list.append(json_obj)
    text_json = {"type": "text", "text": PROMPT_AREA_COMPARE}
    content_list.append(text_json)
    structured_client = client.with_structured_output(AreaCompare)
    human_message = HumanMessage(content=content_list)

    message = structured_client.invoke([human_message])
    logger.debug(f"new message area compare is {message}")

    return message
