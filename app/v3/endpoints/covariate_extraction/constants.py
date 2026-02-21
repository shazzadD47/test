from dataclasses import dataclass

FAILED_TO_PARSE_COVARIATE_TABLE = "Failed to parse covariate table"
FAILED_TO_PARSE_IMAGE_TYPE = "Failed to parse image type, whether its table or text"
GPT_MINI_MODEL = "gpt-4.1-mini"
EXAMPLE_COVARIATE_TABLE = """
,Japanese subjects,,,Caucasian subjects,,
,Semaglutide 0.5 mg Mean [min.-max.],Semaglutide 1.0 mg Mean [min.-max.],Placebo Mean [min.-max.],Semaglutide 0.5 mg Mean [min.-max.],Semaglutide 1.0 mg Mean [min.-max.],Placebo Mean [min.-max.]
N,8,8,6,8,8,6
Age (years),34.1 [23-44],39.1 [29-47],41.4 [27-51],33.4 [26-52],35.0 [25-51],36.5 [26-49]
Height (m),1.70 [1.62-1.81],1.73 [1.65-1.86],1.70 [1.65-1.75],1.81 [1.74-1.90],1.83 [1.72-1.95],1.76 [1.71-1.81]
Body weight (kg),63.9 [58.6-68.9],64.3 [55.3-74.3],62.4 [56.8-67.9],74.9 [66.5-86.0],73.5 [61.4-86.6],69.8 [62.9-73.8]
BMI (kg/m2),22.3 [20.0-24.7],21.4 [20.1-24.5],21.7 [20.4-23.6],22.9 [21.2-24.9],22.1 [20.0-24.5],22.6 [20.0-24.5]
"""  # noqa E501
IMAGE_SAVE_PATH = "data/image.png"
INCORRECT_VALUE = -999999
ATTRIBUTE_TYPES = ["var_stat", "unit", "var", "stat", "max", "min"]
NUMBER_COLUMNS = ["val", "min", "max", "var"]
STRING_COLUMNS = ["var_stat", "unit", "stat"]
GROUP_COLUMN = "group_name"


@dataclass(frozen=True)
class ErrorCode:
    CLAUDE_IMAGE_PROCESSING_FAILED = "Failed to process image with Claude."
    CLAUDE_RESPONSE_PARSING_FAILED = "Failed to parse response from Claude."

    OPENAI_IMAGE_PROCESSING_FAILED = "Failed to process image with OpenAI."
    OPENAI_RESPONSE_PARSING_FAILED = "Failed to parse response from OpenAI."

    QUESTION_REPHRASING_FAILED = "Failed to rephrase question."
    CONTEXT_SUMMARIZATION_FAILED = "Failed to summarize context."
    CONTEXT_QA_FAILED = "Failed to get ANSWER from Contexts."

    LINE_FORMER_FAILED = "Failed to run lineformer model"


MANDATORY_COV_COLUMNS = [
    "Trial_ARM",
    "COV",
    "COV_VAL",
    "COV_UNIT",
    "COV_STAT",
    "COV_MIN",
    "COV_MAX",
    "COV_VAR",
    "COV_VAR_STAT",
    "group_name",
]

COV_VALUE_MAPPING = {
    "na": "N/A",
    "null": "N/A",
    "none": "N/A",
    "sd": "SD",
    "n": "Number",
    "num": "Number",
    "%": "Percentage",
    "pct": "Percentage",
    "perc": "Percentage",
    "percent": "Percentage",
    "percentage": "Percentage",
    "percentage (%)": "Percentage",
    "kg/m²": "kg/m2",
    "µmol/l": "umol/l",
    "year": "years",
    "y": "years",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "kg": "kilograms",
    "g": "grams",
    "mg": "milligrams",
    "l": "liters",
    "ml": "milliliters",
    "µg": "ug",
    "ng": "nanograms",
    "pg": "picograms",
}

paper_dependent_fields = {
    "au",
    "ti",
    "jr",
    "py",
    "vl",
    "is",
    "pg",
    "pubmedid",
    "la",
    "regid",
    "regnm",
    "tp",
    "ts",
    "doi",
    "doi_url",
    "doi url",
    "cit_url",
    "cit url",
    "std ind",
    "std trt",
    "std trt class",
    "comments",
}

COV_DEFAULT_TABLE_DEFINITION = [
    {
        "name": "Trial_ARM",
        "description": """
        The full description of the trial arm including both the
        subject type and treatment.
        For example, 'Japanese subjects_Semaglutide 0.5 mg'.
        """,
        "d_type": "string",
        "c_type": "root",
        "c_header": "Trial_ARM",
    },
    {
        "name": "group_name",
        "description": """The treatment group name within the trial arm.
            For example, 'Semaglutide 0.5 mg', 'Placebo', etc.""",
        "d_type": "string",
        "c_type": "root",
        "c_header": "group_name",
    },
    {
        "name": "COV",
        "description": """Array of covariate names reported for this trial arm.
        For example, ['Age (years)', 'Height (m)', 'Body weight (kg)', 'BMI (kg/m2)'].
        """,
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV",
    },
    {
        "name": "COV_VAL",
        "description": """Array of values for each covariate in the COV array.
            Each position corresponds to the same position in the COV array.""",
        "d_type": "list[float]",
        "c_type": "array",
        "c_header": "COV_VAL",
    },
    {
        "name": "COV_UNIT",
        "description": """Array of units for each covariate value.
            For example, ['years', 'm', 'kg', 'kg/m2'].
            Positions align with the COV array.""",
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_UNIT",
    },
    {
        "name": "COV_STAT",
        "description": """
        Array of statistical measures for each covariate value.
        For example, ['mean', 'mean', 'mean', 'mean'] or ['N/A', 'mean', 'N/A', 'mean'].
        Positions align with the COV array.
        """,
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_STAT",
    },
    {
        "name": "COV_MIN",
        "description": """Array of minimum values for each covariate.
            Positions align with the COV array.""",
        "d_type": "list[float]",
        "c_type": "array",
        "c_header": "COV_MIN",
    },
    {
        "name": "COV_MIN_UNIT",
        "description": """Array of units for the minimum values of each covariate.
            For example, ['years', 'm', 'kg', 'kg/m2'].
            Positions align with the COV array.""",
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_MIN_UNIT",
    },
    {
        "name": "COV_MAX",
        "description": """Array of maximum values for each covariate.
            Positions align with the COV array.""",
        "d_type": "list[float]",
        "c_type": "array",
        "c_header": "COV_MAX",
    },
    {
        "name": "COV_MAX_UNIT",
        "description": """Array of units for the maximum values of each covariate.
            For example, ['years', 'm', 'kg', 'kg/m2'].
            Positions align with the COV array.""",
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_MAX_UNIT",
    },
    {
        "name": "COV_VAR",
        "description": """
        Array of variance values for each covariate, such as standard deviation.
        Can contain null when not available. Positions align with the COV array.
        """,
        "d_type": "list[float]",
        "c_type": "array",
        "c_header": "COV_VAR",
    },
    {
        "name": "COV_VAR_UNIT",
        "description": """Array of units for the variance values of each covariate.
            For example, ['years', 'm', 'kg', 'kg/m2'].
            Positions align with the COV array.""",
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_VAR_UNIT",
    },
    {
        "name": "COV_VAR_STAT",
        "description": """
        Array specifying the type of variance reported for each covariate,
        such as ['SD', 'N/A', 'N/A', 'N/A']. Positions align with the COV array.
        """,
        "d_type": "list[string]",
        "c_type": "array",
        "c_header": "COV_VAR_STAT",
    },
]


EXAMPLE_TABLE_COVARIATE_OUTPUT = """
{{
  "data": [
    {{
      "Trial_ARM": "Japanese subjects_Semaglutide 0.5 mg ",
      "group_name": "Semaglutide 0.5 mg",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [34.1, 1.7, 63.9, 22.3],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["mean", "mean", "mean", "mean"],
      "COV_MIN": [23.0, 1.62, 58.6, 20.0],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [44.0, 1.81, 68.9, 24.7],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [6.5, null, null, null],
      "COV_VAR_UNIT": ["years", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["SD", "N/A", "N/A", "N/A"],
    },
    {
      "Trial_ARM": "Japanese subjects_Semaglutide 1.0 mg",
      "group_name": "Semaglutide 1.0 mg",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [39.1, 1.73, 64.3, 21.4],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["mean", "mean", "N/A", "mean"],
      "COV_MIN": [29.0, 1.65, 55.3, 20.1],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [47.0, 1.86, 74.3, 24.5],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [null, null, null, null],
      "COV_VAR_UNIT": ["N/A", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["N/A", "N/A", "N/A", "N/A"]
    },
    {
      "Trial_ARM": "Japanese subjects_Placebo",
      "group_name": "Placebo",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [41.4, 1.7, 62.4, 21.7],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["N/A", "mean", "N/A", "mean"],
      "COV_MIN": [27.0, 1.65, 56.8, 20.4],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [51.0, 1.75, 67.9, 23.6],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [null, null, null, null],
      "COV_VAR_UNIT": ["N/A", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["N/A", "N/A", "N/A", "N/A"]
    },
    {
      "Trial_ARM": "Caucasian subjects_Semaglutide 0.5 mg",
      "group_name": "Semaglutide 0.5 mg",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [33.4, 1.81, 74.9, 22.9],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["N/A", "mean", "N/A", "mean"],
      "COV_MIN": [26.0, 1.74, 66.5, 21.2],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [52.0, 1.9, 86.0, 24.9],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [null, null, null, null],
      "COV_VAR_UNIT": ["N/A", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["N/A", "N/A", "N/A", "N/A"]
    },
    {
      "Trial_ARM": "Caucasian subjects_Semaglutide 1.0 mg",
      "group_name": "Semaglutide 1.0 mg",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [35.0, 1.83, 73.5, 22.1],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["mean", "mean", "N/A", "mean"],
      "COV_MIN": [25.0, 1.72, 61.4, 20.0],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [51.0, 1.95, 86.6, 24.5],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [null, null, null, null],
      "COV_VAR_UNIT": ["N/A", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["N/A", "N/A", "N/A", "N/A"]
    },
    {
      "Trial_ARM": "Caucasian subjects_Placebo",
      "group_name": "Placebo",
      "COV": ["Age (years)", "Height (m)", "Body weight (kg)", "BMI (kg/m2)"],
      "COV_VAL": [36.5, 1.76, 69.8, 22.6],
      "COV_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_STAT": ["N/A", "mean", "mean", "N/A"],
      "COV_MIN": [26.0, 1.71, 62.9, 20.0],
      "COV_MIN_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_MAX": [49.0, 1.81, 73.8, 24.5],
      "COV_MAX_UNIT": ["years", "m", "kg", "kg/m2"],
      "COV_VAR": [null, null, null, null],
      "COV_VAR_UNIT": ["N/A", "N/A", "N/A", "N/A"],
      "COV_VAR_STAT": ["N/A", "N/A", "N/A", "N/A"]
    }}
  ]
}}
"""

MAX_RETRIES = 5

COVARIATE_ERROR_MESSAGE = "Error occured when extracting covariate: "
AUTOFILL_ERROR_MESSAGE = "AI Autofill failed. "
ADVERSE_EVENT_ERROR_MESSAGE = "Adverse Event Digitization failed. "
SEPARATOR = "\n" + "-" * 100 + "\n"
