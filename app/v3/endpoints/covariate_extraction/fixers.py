import json
from typing import Any, Self

import json_repair
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.v3.endpoints.covariate_extraction.constants import (
    MANDATORY_COV_COLUMNS,
    NUMBER_COLUMNS,
    STRING_COLUMNS,
)


class COVTableOutputFixer(Runnable):

    required_keys: dict[str, Any] = {
        "Trial_ARM": "N/A",
        "COV": "N/A",
        "group_name": "N/A",
    }

    for column in MANDATORY_COV_COLUMNS:
        if column not in required_keys:
            for col_type in NUMBER_COLUMNS:
                if column.lower().endswith(col_type):
                    required_keys[column] = None
            for col_type in STRING_COLUMNS:
                if column.lower().endswith(col_type):
                    required_keys[column] = "N/A"

    def extract_data(
        self, input_data: dict[str, Any] | list[Any]
    ) -> list[dict[str, Any]]:
        """
        Extracts and normalizes 'data' from nested or alternate key names.
        """
        # If input_data is a list, wrap it in a dict under 'data'
        if isinstance(input_data, list):
            return input_data

        if isinstance(input_data, str):
            input_data = json_repair.loads(input_data)

        # Check if any key in the input_data has list content to consider it as 'data'
        for _, value in input_data.items():
            if isinstance(value, list):
                return value

        for _, value in input_data.items():
            if isinstance(value, dict):
                return self.extract_data(value)

        # Default to an empty list if 'data' or list-like structure is not found
        return []

    def normalize_data_items(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Ensures each item in 'data' contains all required keys with default values.
        """
        return [{**self.required_keys, **item} for item in data]

    def invoke(
        self: Self,
        input: BaseModel,
        *args,
    ) -> dict[str, Any]:
        input_data = input.model_dump()

        # Step 1: Extract the 'data' key (or equivalent) if nested or mislabeled
        data = self.extract_data(input_data)

        # Step 2: Normalize each item in 'data' to ensure all required keys
        fixed_data = self.normalize_data_items(data)

        # Step 3: Return the output in the desired format
        return json.dumps(
            {
                "data": fixed_data,
            }
        )


class COVTypeOutputFixer(Runnable):
    def __init__(self, standard_covariates: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_covariates = standard_covariates["covariate_name"].tolist()
        self.standard_covariates = list(set(self.standard_covariates))

    def fill_missing_covariates(self, key: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Fills missing covariates in the data with the standard covariates.
        """
        for covariate in self.standard_covariates:
            if covariate not in data:
                data[covariate] = "N/A"
        return {key: data}

    def fix_data(self, input_data: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
        """
        Extracts and normalizes 'mapping' from nested or alternate key names.
        """
        if isinstance(input_data, str):
            input_data = json_repair.loads(input_data)

        # Check if any key in the input_data has list content
        # to consider it as 'mapping'
        for key in input_data:
            if key == "mapping" and isinstance(input_data[key], dict):
                return self.fill_missing_covariates(key, input_data[key])
            else:
                return self.fix_data(input_data[key])

        return {}

    def invoke(
        self: Self,
        input: dict,
        *args,
    ) -> dict[str, Any]:
        fixed_data = self.fix_data(input)

        return json.dumps(fixed_data)
