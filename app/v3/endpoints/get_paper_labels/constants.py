from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    CLAUDE_RESPONSE_PARSING_FAILED = "Failed to parse response from Claude."
    OPENAI_RESPONSE_PARSING_FAILED = "Failed to parse response from OpenAI."

    QUESTION_REPHRASING_FAILED = "Failed to rephrase question."
    CONTEXT_SUMMARIZATION_FAILED = "Failed to summarize context."
    CONTEXT_QA_FAILED = "Failed to get ANSWER from Contexts."


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
