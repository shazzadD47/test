import base64
import json

import cv2
from anthropic import AsyncAnthropic
from anthropic import InternalServerError as AnthropicInternalServerError

from app.configs import settings
from app.exceptions.http import AnthropicServerFailed
from app.v3.endpoints.plot_digitizer.exceptions import (
    ClaudeImageProcessingFailed,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger
from app.v3.endpoints.plot_digitizer.prompts import LEGEND_MAPPING_PROMPTS


def define_json(img_paths, legend_or_markers, names):
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
    text_json = {
        "type": "text",
        "text": LEGEND_MAPPING_PROMPTS.format(
            legend_or_markers=legend_or_markers, names=names
        ),
    }
    content_list.append(text_json)
    return content_list


client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


async def extract_labels_individual_images_with_color_added(
    image_path, legend_list, legend_or_markers="legend"
):
    names = ""
    name_list = legend_list
    for i in range(len(name_list)):
        if i < len(name_list) - 1:
            names = names + name_list[i] + ","
        else:
            names = names + name_list[i]
    content_list = define_json(image_path, legend_or_markers, names)
    try:
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
        )
    except AnthropicInternalServerError:
        logger.exception("An error occurred at Anthropic's end.")
        raise AnthropicServerFailed()
    except Exception:
        logger.exception(
            "An error occurred while extracting labels from individual images."
        )
        raise ClaudeImageProcessingFailed()
    json_data = json.loads(message.content[0].text)
    legends = {key: value[legend_or_markers] for key, value in json_data.items()}
    return legends
