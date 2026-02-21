import json
import os

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile

load_dotenv()
equation_extract = APIRouter(tags=["Equation_extract"])


@equation_extract.post("/process-image-equation")
async def process_image(file: UploadFile = File(...)):
    """
    Process an image containing an equation and extract its LaTeX representation using
    the Mathpix API.

    Args:
        file (UploadFile): The image file containing the equation.

    Returns:
        dict: Dictionary containing the extracted LaTeX representation of the equation.

    Raises:
        HTTPException: If there is an error communicating with the Mathpix API.

    """
    try:
        image_bytes = await file.read()

        response = requests.post(
            "https://api.mathpix.com/v3/latex",
            timeout=60,
            files={"file": image_bytes},
            data={
                "options_json": json.dumps(
                    {
                        "formats": [
                            "text",
                            "latex_simplified",
                            "latex_styled",
                            "mathml",
                            "asciimath",
                            "latex_list",
                        ]
                    }
                )
            },
            headers={
                "app id": os.getenv("MATHPIX_APP_ID"),
                "app_key": os.getenv("MATHPIX_APP_KEY"),
            },
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as err:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with Mathpix API: {err}"
        )
