from app.core.auto.chat_model import AutoChatModel
from app.v3.endpoints.dosing_table.configs import settings as dosing_settings

FAILED_TO_PARSE_DOSING_TABLE = "Failed to parse dosing table"
DOSING_TABLE_ERROR_MESSAGE = "Dosing table error: "

NO_FIGURE_INITIAL_RAG_QUESTIONS = [
    "what are the trail arms?",
    "what is the study design?",
    "what is the randomization method?",
    "what are the procedures?",
    "what is the methodology/methods used in the study?",
    "what is the inter-dose interval?",
    "what is the route of administration?",
    "what is the dose escalation pattern?",
]

DOSING_TABLE_COLUMN_ORDER = [
    "GROUP",
    "ROUTE",
    "ARM_TIME",
    "ARM_TIME_UNIT",
    "AMT",
    "AMT_UNIT",
    "II",
    "II_UNIT",
    "ADDL",
    "STD_TRT",
]
chain_configs_with_figure = {
    "callbacks": [],
    "tags": ["dosing_table", "with_figure"],
}
chain_configs_without_figure = {
    "callbacks": [],
    "tags": ["dosing_table", "without_figure"],
}

context_agent = AutoChatModel.from_model_name(
    model_name=dosing_settings.CONTEXT_GENERATOR_LLM, temperature=0.2
)
