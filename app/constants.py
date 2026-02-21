import logging

from tenacity import (
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)

from app.logging import logger

d_type_map = {
    "integer": int | float,
    "int": int | float,
    "float": float,
    "double": float,
    "decimal": float,
    "string": str,
    "number": int | float,
    "numeric": int | float,
    "list": list,
    "array": list,
    "dict": dict,
    "boolean": bool,
    "list[string]": list[str],
    "list[float]": list[float],
    "list[integer]": list[int | float],
    "csv": str,
    "literal": str,  # Fallback to str;
    # code needing specific literals should construct Literal[...] dynamically
    # from the literal_options list
}

# ---------------------------------------------------------------------------
# Centralised type enumerations for d_type / c_type across
# general_extraction, plot_digitizer and covariate_extraction.
# All three pipelines MUST reference these tuples so that schema
# validation and runtime checks stay in sync.
# ---------------------------------------------------------------------------

# Every valid value that can appear in a table-field's ``d_type``.
VALID_D_TYPES: tuple[str, ...] = tuple(d_type_map.keys())

# Subset of d_types that represent numeric values.
NUMERICAL_D_TYPES: tuple[str, ...] = (
    "float",
    "integer",
    "int",
    "number",
    "numeric",
    "double",
    "decimal",
)

# Every valid value that can appear in a table-field's ``c_type``.
VALID_C_TYPES: tuple[str, ...] = (
    "general",
    "root",
    "General",
    "Root",
    "array",
    "paper_label",
)

tenacity_kwargs = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(
        multiplier=0.75,
        min=2,
        max=10,
    ),
    "reraise": True,
    "before_sleep": before_sleep_log(logger, logging.WARNING),
}
