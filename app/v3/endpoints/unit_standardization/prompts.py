from langchain_core.prompts import PromptTemplate

UNIT_STANDARDIZATION_RULES = """
    Rules:
    1. Use full words for base units (e.g., "gram" not "g", "liter"
       not "l")
    2. Use singular forms (e.g., "gram" not "grams"), EXCEPT:
    -  Time units (e.g., "day", "hour", "week", "month", "year",
       "minute", "second") must always be in **plural form** with
       full spelling (e.g., "days", "hours", "weeks", "months",
       "years", "minutes", "seconds").
    3. Be case-sensitive where it matters (MB ≠ Mb)
    4. For derived units (e.g., concentration, velocity, force,
       pressure), use "/" for "per" (e.g., "milligram/liter" for
       concentration, "meter/second" for velocity, "newton/meter^2"
       for pressure)
    5. Keep the same scientific meaning
    6. If a unit is already in standard form, keep it unchanged
    7. For unknown or ambiguous units, return them unchanged
    8. Handle unicode characters properly (μ, −, ⁻, etc.)
    9. For area, volume, or any other powers, use metric symbol
       notation with the caret (^) for superscripts (e.g., meter^2
       instead of m² or square meter, meter^3 instead of m³ or
       cubic meter).
    10. If a symbol is used as a unit (e.g., "%", "$", "€", "°C"),
        convert it to its full word form (e.g., "percent",
        "US dollar", "euro", "degree Celsius").
    11. If the unit is missing or "none", "N/A", "n/a", "NA", or
        any case variation, convert it to "N/A".
    12. Units should be provided in a non-verbose way.
        For example, if the unit is "milligram",
        then provide "milligram" and not "The unit for xyz is milligram".
    13. If multiple units are listed with numeric prefixes
        (e.g., "1. %", "2. mmol/mol"), standardize each unit and return
        them as a comma-separated list (e.g., "percent, millimole/mole").
"""

UNIT_STANDARDIZATION_PROMPT = """You are a scientific unit standardization expert. Your task is to convert the following units to their standard, full forms.

{unit_standardization_rules}

Units to standardize:
{units_str}

Please respond with a JSON object containing a "unit_mappings" field with key-value pairs of original units and their standardized forms:
{{
    "unit_mappings": {{
        "original_unit_1 (column_name)": "standardized_unit_1",
        "original_unit_2 (column_name)": "standardized_unit_2"
    }}
}}

Example:
{{
    "unit_mappings": {{
        "mg/l (concentration_unit)": "milligram/liter",
        "Days (time_unit)": "days",
        "Day (time_unit)": "days",
        "µg/mL (concentration_unit)": "microgram/milliliter",
        "ngml−1 (concentration_unit)": "nanogram/milliliter",
        "ngml⁻¹ (concentration_unit)": "nanogram/milliliter",
        "mm³ (volume_unit)": "millimeter^3",
        "MB (data_unit)": "megabyte",
        "Mb (data_unit)": "megabit",
        "mg/m2 (concentration_unit)": "milligram/meter^2",
        "milliliter/minute/1.73 meter² (any_unit)": "milliliter/minute/1.73 meter^2",
        "g/m3 (concentration_unit)": "gram/meter^3",
        "Day (ARM_TIME_UNIT)": "days",
        "Hour (X_Unit)": "hours",
        "min (time_unit)": "minutes",
        "sec (time_unit)": "seconds",
        "% (ratio_unit)": "percent",
        "$ (currency_unit)": "US dollar",
        "€ (currency_unit)": "euro",
        "°C (temperature_unit)": "degree Celsius",
        "None (any_unit)": "N/A",
        "N/A (any_unit)": "N/A"
        "1. mg/dL\n2. mmol/L", (any_unit)": "milligram/deciliter, millimole/liter"
    }}
}}
"""

prompt = PromptTemplate.from_template(UNIT_STANDARDIZATION_PROMPT)
prompt = prompt.partial(unit_standardization_rules=UNIT_STANDARDIZATION_RULES)
assignments = prompt.partial_variables
assignments.update({k: f"{{{k}}}" for k in prompt.input_variables})
formatted_prompt = prompt.template
for k, v in assignments.items():
    formatted_prompt = formatted_prompt.replace(f"{{{k}}}", v)
UNIT_STANDARDIZATION_PROMPT = formatted_prompt
