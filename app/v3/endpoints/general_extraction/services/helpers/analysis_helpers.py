import os

from app.utils.image import (
    combine_images_vertically_with_padding,
    convert_image_to_base64,
    get_image_from_url,
)
from app.utils.utils import generate_unique_id


def return_image_with_legends(image_data: dict) -> tuple[str, str]:
    image, media_type = get_image_from_url(
        image_data["figure_url"], return_media_type=True
    )

    has_legends = (
        "legends" in image_data
        and image_data["legends"] is not None
        and len(image_data["legends"]) > 0
    )

    if has_legends:
        orig_image_path = generate_unique_id() + ".png"
        with open(orig_image_path, "wb") as fo:
            fo.write(image)

        bounding_box = image_data["bounding_box"]

        for legend_info in image_data["legends"]:
            legend_image_path = generate_unique_id() + ".png"
            legend_image, _ = get_image_from_url(
                legend_info["figure_url"], return_media_type=True
            )
            with open(legend_image_path, "wb") as fo:
                fo.write(legend_image)

            image_with_legend = combine_images_vertically_with_padding(
                image1_path=orig_image_path,
                image2_path=legend_image_path,
                bounding_box_legend=legend_info["bounding_box"],
                bounding_box=bounding_box,
            )
            image = image_with_legend
            with open(orig_image_path, "wb") as fo:
                fo.write(image)
            if os.path.exists(legend_image_path):
                os.remove(legend_image_path)

        image = bytes(image)
        image = convert_image_to_base64(image)

        if os.path.exists(orig_image_path):
            os.remove(orig_image_path)

        return image, media_type
    else:
        image = convert_image_to_base64(image)
        return image, media_type
