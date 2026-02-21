from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.core.auto import AutoChatModel
from app.utils.cache import (
    batch_save_to_cache,
    load_from_cache,
)
from app.utils.llms import batch_invoke_chain_with_retry
from app.v3.endpoints.unit_standardization.configs import settings as uc_settings
from app.v3.endpoints.unit_standardization.logging import celery_logger as logger
from app.v3.endpoints.unit_standardization.prompts import (
    UNIT_STANDARDIZATION_PROMPT,
)
from app.v3.endpoints.unit_standardization.schemas import UnitStandardizationOutput


class UnitStandardizer:
    def __init__(self):
        # Initialize LLM
        self.llm = AutoChatModel.from_model_name(
            uc_settings.MODEL_NAME,
            temperature=uc_settings.TEMPERATURE,
        )

        # Create the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["units_str"],
            template=UNIT_STANDARDIZATION_PROMPT,
        )

        # Create structured output LLM
        self.structured_llm = self.llm.with_structured_output(
            UnitStandardizationOutput, method="function_calling"
        )

        # Create the LCEL chain
        self.standardization_chain = (
            {"units_str": RunnablePassthrough()}
            | self.prompt_template
            | self.structured_llm
        )

        logger.info(
            "Initialized UnitStandardizer with function calling structured output"
        )

    def _get_cache_key(self, unit: str, column: str) -> str:
        """Generate cache key for unit-column pair."""
        return f"{uc_settings.CACHE_PREFIX}{column}:{unit}"

    def _create_units_string(self, column_units: dict[str, list[str]]) -> str:
        """Create the units string for the prompt."""
        units_str = ""
        for col, units in column_units.items():
            for unit in units:
                units_str += f"- {unit} ({col})\n"
        return units_str.strip()

    def _prepare_batch_inputs(
        self, all_units_for_llm: list[tuple[str, str]]
    ) -> list[str]:
        """Prepare batch inputs for LCEL batch processing."""
        batch_inputs = []

        for i in range(0, len(all_units_for_llm), uc_settings.BATCH_SIZE):
            batch = all_units_for_llm[i : i + uc_settings.BATCH_SIZE]
            batch_column_units = {}
            for col, unit in batch:
                batch_column_units.setdefault(col, []).append(unit)

            units_str = self._create_units_string(batch_column_units)
            batch_inputs.append(units_str)

        return batch_inputs

    def _standardize_units_batch(
        self, units_for_llm_list: list[tuple[str, str]]
    ) -> dict[str, str]:
        if not units_for_llm_list:
            return {}

        # Prepare batch inputs
        batch_inputs = self._prepare_batch_inputs(units_for_llm_list)
        total_batches = len(batch_inputs)

        logger.info(
            f"Processing {len(units_for_llm_list)} units "
            f"in {total_batches} batch(es) using function calling structured output"
        )

        new_standardizations = {}

        try:
            # Use batch_invoke_chain_with_retry from llms.py
            batch_results = batch_invoke_chain_with_retry(
                chain=self.standardization_chain,
                input=batch_inputs,
                config={"max_concurrency": uc_settings.MAX_CONCURRENCY},
                max_retries=uc_settings.MAX_RETRIES,
                reraise=True,
                min_wait_seconds=2,
                max_wait_seconds=10,
                wait_exponential_multiplier=0.75,
                retry_if_exception_types=(Exception,),
            )

            # Process results
            for batch_result in batch_results:
                if isinstance(batch_result, UnitStandardizationOutput):
                    result_dict = batch_result.to_dict()
                    new_standardizations.update(result_dict)
                else:
                    logger.warning(f"Unexpected result type: {type(batch_result)}")

            if new_standardizations:
                logger.info(
                    f"Successfully standardized {len(new_standardizations)} units "
                    f"via function calling structured output"
                )
            else:
                logger.warning(
                    "Function calling structured output batch returned empty result"
                )

        except Exception as e:
            logger.exception(f"Batch processing failed after retries: {e}")

        return new_standardizations

    def standardize_units(
        self, column_units: dict[str, set[str]]
    ) -> dict[str, dict[str, str]]:
        """Synchronous unit standardization with
        global caching and LCEL batch processing."""
        total_units = sum(len(units) for units in column_units.values())
        logger.info(
            f"Starting standardization of {total_units} "
            f"units across {len(column_units)} columns"
        )

        standardization_mapping = {col: {} for col in column_units}
        units_for_llm = []

        # Step 1: Check cache
        cache_hits = 0
        for col, units in column_units.items():
            for unit in units:
                cache_key = self._get_cache_key(unit, col)
                cached_value = load_from_cache(cache_key)
                if cached_value is not None:
                    standardization_mapping[col][unit] = cached_value
                    cache_hits += 1
                else:
                    units_for_llm.append((col, unit))

        logger.info(f"Cache hits: {cache_hits}/{total_units}")
        logger.info(f"Units requiring LLM processing: {len(units_for_llm)}")

        # Step 2: Process remaining units with LCEL batch
        if units_for_llm:
            new_standardizations = self._standardize_units_batch(units_for_llm)

            # Step 3: Update mapping + cache
            cache_updates = {}
            for col, unit in units_for_llm:
                unit_key = f"{unit} ({col})"
                if unit_key in new_standardizations:
                    standardized_unit = new_standardizations[unit_key]
                    standardization_mapping[col][unit] = standardized_unit
                    cache_updates[self._get_cache_key(unit, col)] = standardized_unit

            if cache_updates:
                batch_save_to_cache(
                    cache_updates, expiration=uc_settings.CACHE_EXPIRATION
                )

            logger.info(
                f"Added {len(new_standardizations)} new standardizations to cache"
            )

        return standardization_mapping
