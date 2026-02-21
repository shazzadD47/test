from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from app.configs import settings
from app.core.auto import AutoCallbackHandler, AutoChatModel
from app.core.callbacks.anthropic import AnthropicTokenUsageCallbackHandler
from app.v3.endpoints.iterative_autofill.chains import prepare_question_rephrasing_chain
from app.v3.endpoints.iterative_autofill.configs import settings as iaf_settings


@dataclass(frozen=True)
class ErrorCode:
    CLAUDE_RESPONSE_PARSING_FAILED = "Failed to parse response from Claude."
    OPENAI_RESPONSE_PARSING_FAILED = "Failed to parse response from OpenAI."

    QUESTION_REPHRASING_FAILED = "Failed to rephrase question."
    CONTEXT_SUMMARIZATION_FAILED = "Failed to summarize context."
    CONTEXT_QA_FAILED = "Failed to get ANSWER from Contexts."
    MAX_INPUT_TOKENS_EXCEEDED = "Max input tokens exceeded."


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

MAX_RETRIES = 5
SEPARATOR = f"\n{'-' * 100}\n\n"
TOP_K = 20  # number of documents to retrieve from vector store
ITERATIVE_AUTOFILL_ERROR_MESSAGE = "Iterative autofill failed."

claude_usage = AnthropicTokenUsageCallbackHandler()
openai_usage = OpenAICallbackHandler()
usage_handler = AutoCallbackHandler.from_model_name(iaf_settings.OPENAI_MODEL_NAME)

if isinstance(usage_handler, OpenAICallbackHandler):
    usage_handler = openai_usage
elif isinstance(usage_handler, AnthropicTokenUsageCallbackHandler):
    usage_handler = claude_usage

llm = AutoChatModel.from_model_name(
    iaf_settings.OPENAI_MODEL_NAME,
    temperature=0.2,
    max_tokens=iaf_settings.OPENAI_MAX_TOKENS,
    callbacks=[usage_handler],
)

llm_claude = ChatAnthropic(
    model=iaf_settings.CLAUDE_MODEL_ID,
    temperature=0.2,
    max_tokens_to_sample=iaf_settings.CLAUDE_MAX_TOKENS,
    api_key=settings.ANTHROPIC_API_KEY,
    callbacks=[claude_usage],
)

rephrase_chain = prepare_question_rephrasing_chain(llm=llm)
