UNIT_NORMALIZATION_PROMPT = """
    You are an expert in unit normalization.
    You are given a list of unit values which you need to normalize.

    Input Unit Values: {values}

    Normalization Rules:
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

    Output Format:
    Strictly follow the output format. Do not return any other text or comments.
"""


STANDARDIZE_VALUES_PROMPT = """
    You are an expert in fixing errors.
    Input Values: {values}
    Fixing Rules:
    1. Fix any spelling errors, inaccuracies, inconsistencies,
    and capitalization errors.
    2. Remove any unnecessary punctuation.
    3. Replace full values with their correct abbreviations
    if you are absolutely certain of the correctness of the abbreviation.
    Do not convert existing abbreviations to whole values.
    4. Undo the superscript/subscript. For example, kg/m² to kg/m2.
    5. Be consistent with your output.

    Return the fixed values in a list. It MUST be
    same order and length as the input values.
    So, the first item of the output must be the fixed value
    of the first item of the input values.
    Output Format:
    Strictly follow the output format.
    Do not return any other text or comments.
    """

STANDARDIZE_REGIMEN_PROMPT = """
    You are an expert in normalizing drug regimen values.
    You are given a list of drug regimen values which you need to normalize.

    Input Values: {values}

    Regimen abbreviations:
    {REGIMEN_ABBREVIATIONS}

    Similar Regimen Mapping: (format: abbreviation: description)
    {SIMILAR_REGIMEN_MAPPING}

    Normalization Rules:
    1. If the value is not in the regimen abbreviations,
    try to map it to a similar regimen
    based on the similar regimen mapping. If failed, try
    to map it to an abbreviation
    using the description of the regimen abbreviations.
    If failed, return the value as is.
    2. If the value is in the regimen abbreviations,
    return the value as it is.
    3. If the value is in the similar regimen mapping,
    return the mapped regimen.
    4. Try to return an abbreviation if you are
    absolutely certain of the correctness of the abbreviation.
    Return the fixed values in a list. It MUST be
    same order and length as the input values.
    So, the first item of the output must be the fixed value
    of the first item of the input values.
    5. Be consistent with your output.

    Output Format:
    Strictly follow the output format.
    Do not return any other text or comments.
    """

STANDARDIZE_DV_VALUES_PROMPT = """
    **Role:** You are an expert in normalizing values.

    **Goal:** Normalize an input list of values (which can include units
    or statistical terms) to a predefined list of "constant" values,
    following specific mapping rules. The output must be a list of
    normalized values maintaining the original order.

    ## Constant Values (Normalization Targets)
    This list defines the *only* acceptable output values after
    normalization, unless a direct map cannot be found.
    constant_values = {constant_values}

    ## Input Values (To Be Normalized)
    input_values = {input_values}

    ## Normalization Rules

    1.  **Exact Match:** If an input value is *exactly* present in the
        `constant_values` list (case-sensitive), return the value as
        it is.
        * *Example:* Input **'km'** maps to **'km'**.

    2.  **Synonym/Abbreviation Mapping:** If an input value is *not* an
        exact match, try to map it to a value in the `constant_values`
        list if it is a common synonym, abbreviation, or different
        case/format of a constant value. Define these common mappings
        *explicitly* to remove ambiguity:
        * 'km' maps to **'kilometer'**
        * 'nanomole/L' maps to **'nmol/l'** (handling case and slight
        unit variation)
        * 'receiver operating characteristic' maps to **'ROC'**
        (abbreviation)
        * 'coefficient of variation' maps to **'CV'** (abbreviation)
        * 'standard deviation' maps to **'SD'** (abbreviation)
        * 'gram' maps to **'kg'** (This is a *unit conversion* mapping,
        simplifying 'gram' to the base 'kg' constant for weight).
        *Note: The system should not perform numerical unit
        conversion, only unit label mapping.*

    3. Question your self before mapping. "Is a the same as b?" - if
    answer is "yes" with 100% percent confidence, then and only then
    map to the value in constant list.

    3.  **No Match (Negative Example):** If the input value cannot be
        mapped by Rule 1 or Rule 2, return the input value as it is.
        * *Example:* Input **'some random term'** maps to
        **'some random term'**.

    4.  **Output Format and Order (Strict Requirement):** Return the
        fixed values in a list. It **MUST** be the same order and
        length as the input values. The first item of the output must
        be the fixed value of the first item of the input values, and
        so on.

    ## Example Inputs
    ['milimiter mole per litre', 'receiver operating characteristic',
    'coefficient of variation', 'nanomole/L', 'standard deviation',
    'gram', 'some random term']

    ## Expected Output for the Example Inputs
    ['mmol/l', 'ROC', 'CV', 'nmol/l', 'SD', 'kg', 'some random term']

    ## Output Format
    Strictly follow the output format. Do not return any other text or
    comments.

    ['<normalized_value_1>', '<normalized_value_2>', ...,
    '<normalized_value_n>']
"""
