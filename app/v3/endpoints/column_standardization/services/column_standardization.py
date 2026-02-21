from langchain_core.runnables import RunnablePassthrough

from app.core.auto import AutoChatModel
from app.utils.llms import batch_invoke_chain_with_retry
from app.v3.endpoints.column_standardization.configs import settings as cs_settings
from app.v3.endpoints.column_standardization.logging import celery_logger as logger
from app.v3.endpoints.column_standardization.schemas import ColumnStandardizationOutput
from app.v3.endpoints.column_standardization.utils import (
    create_column_standardization_prompt,
)


class ColumnStandardizer:
    def __init__(
        self,
        column_name: str,
        column_description: str | None = None,
        user_instruction: str | None = None,
    ):
        """
        Initialize ColumnStandardizer with column-specific context.

        Args:
            column_name: Name of the column being standardized
            column_description: Optional description of the column's content
            user_instruction: Optional user-specific standardization instructions
        """
        self.column_name = column_name
        self.column_description = column_description
        self.user_instruction = user_instruction

        self.llm = AutoChatModel.from_model_name(
            cs_settings.MODEL_NAME,
            temperature=cs_settings.TEMPERATURE,
        )

        self.prompt_template = create_column_standardization_prompt(
            column_name=column_name,
            column_description=column_description,
            user_instruction=user_instruction,
        )

        self.structured_llm = self.llm.with_structured_output(
            ColumnStandardizationOutput, method="function_calling"
        )

        self.standardization_chain = (
            {"values_str": RunnablePassthrough()}
            | self.prompt_template
            | self.structured_llm
        )

        logger.info(
            f"Initialized ColumnStandardizer for column '{column_name}' "
            f"with function calling structured output"
        )

    def _create_values_string(self, column_values: dict[str, list[str]]) -> str:
        """Create the values string for the prompt."""
        values_str = ""
        for col, values in column_values.items():
            for value in values:
                values_str += f"- {value} ({col})\n"
        return values_str.strip()

    def _prepare_batch_inputs(
        self, all_values_for_llm: list[tuple[str, str]]
    ) -> list[str]:
        """Prepare batch inputs for LCEL batch processing."""
        batch_inputs = []

        for i in range(0, len(all_values_for_llm), cs_settings.BATCH_SIZE):
            batch = all_values_for_llm[i : i + cs_settings.BATCH_SIZE]
            batch_column_values = {}
            for col, value in batch:
                batch_column_values.setdefault(col, []).append(value)

            values_str = self._create_values_string(batch_column_values)
            batch_inputs.append(values_str)

        return batch_inputs

    def _standardize_values_batch(
        self, values_for_llm_list: list[tuple[str, str]]
    ) -> dict[str, str]:
        """Standardize a batch of column values using LLM."""
        if not values_for_llm_list:
            return {}

        batch_inputs = self._prepare_batch_inputs(values_for_llm_list)
        total_batches = len(batch_inputs)

        logger.info(
            f"Processing {len(values_for_llm_list)} values "
            f"in {total_batches} batch(es) using function calling structured output"
        )

        new_standardizations = {}

        try:
            batch_results = batch_invoke_chain_with_retry(
                chain=self.standardization_chain,
                input=batch_inputs,
                config={"max_concurrency": cs_settings.MAX_CONCURRENCY},
                max_retries=cs_settings.MAX_RETRIES,
                reraise=True,
                min_wait_seconds=2,
                max_wait_seconds=10,
                wait_exponential_multiplier=0.75,
                retry_if_exception_types=(Exception,),
            )

            for batch_idx, batch_result in enumerate(batch_results):
                if isinstance(batch_result, ColumnStandardizationOutput):
                    result_dict = batch_result.to_dict()

                    if not result_dict:
                        logger.warning(
                            f"Batch {batch_idx + 1}/{total_batches} returned \
                                empty mappings. "
                            f"This may indicate the LLM didn't \
                                understand the prompt."
                        )
                        if batch_idx < len(batch_inputs):
                            logger.debug(
                                f"Failed batch input: \
                                {batch_inputs[batch_idx][:500]}..."
                            )
                    else:
                        new_standardizations.update(result_dict)
                        logger.debug(
                            f"Batch {batch_idx + 1}/{total_batches}: "
                            f"received {len(result_dict)} mappings"
                        )
                else:
                    logger.warning(
                        f"Batch {batch_idx + 1}/{total_batches}: "
                        f"Unexpected result type: {type(batch_result)}"
                    )

            if new_standardizations:
                logger.info(
                    f"Successfully standardized {len(new_standardizations)} values "
                    f"via function calling structured output"
                )
            else:
                logger.error(
                    "LLM returned empty standardization results for all batches. "
                    "This indicates the model failed to process the request properly."
                )
                logger.warning("Falling back to identity mapping (no changes)")
                for col, value in values_for_llm_list:
                    value_key = f"{value} ({col})"
                    new_standardizations[value_key] = value

        except Exception as e:
            logger.exception(f"Batch processing failed after retries: {e}")
            logger.warning("Exception occurred, falling back to identity mapping")
            new_standardizations = {}
            for col, value in values_for_llm_list:
                value_key = f"{value} ({col})"
                new_standardizations[value_key] = value

        return new_standardizations

    def standardize_column_values(
        self, column_values: dict[str, set[str]]
    ) -> dict[str, dict[str, str]]:
        """
        Standardize column values without caching.

        Args:
            column_values: Dict mapping column name to set of unique values

        Returns:
            Dict mapping column name to dict of original->standardized value mappings
        """
        total_values = sum(len(values) for values in column_values.values())
        logger.info(
            f"Starting standardization of {total_values} unique values "
            f"across {len(column_values)} column(s)"
        )

        values_for_llm = []
        for col, values in column_values.items():
            for value in values:
                if value and str(value).strip():
                    values_for_llm.append((col, str(value)))

        if not values_for_llm:
            logger.info("No valid values to standardize")
            return {col: {} for col in column_values}

        logger.info(f"Processing {len(values_for_llm)} non-empty values with LLM")

        new_standardizations = self._standardize_values_batch(values_for_llm)

        standardization_mapping = {col: {} for col in column_values}

        for col, value in values_for_llm:
            value_key = f"{value} ({col})"
            if value_key in new_standardizations:
                standardized_value = new_standardizations[value_key]
                standardization_mapping[col][value] = standardized_value
            else:
                logger.warning(
                    f"Value '{value}' for column '{col}' not found in "
                    f"standardization results, keeping original"
                )
                standardization_mapping[col][value] = value

        logger.info(
            f"Completed standardization: {len(new_standardizations)} \
                values standardized"
        )

        return standardization_mapping
