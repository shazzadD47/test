from enum import StrEnum


class ReportGenerationType(StrEnum):
    AI_ASSISTANT = "ai_assistant"
    AI_EDIT = "ai_edit"
    AI_INSIGHTS = "ai_insights"


class ReportAgents(StrEnum):
    ASSISTANT_AGENT = "assistant_agent"
    EDIT_AGENT = "edit_agent"
    INSIGHT_AGENT = "insight_agent"
