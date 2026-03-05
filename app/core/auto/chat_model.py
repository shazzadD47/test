from anthropic import Anthropic
from google.genai.client import Client as GoogleClient
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.core.auto.factory import BaseAutoModel, BaseAutoSdkModel
from app.core.auto.sdk_models import AnthropicSdkModel, GoogleSdkModel, OpenAISdkModel

OPENAI_MODEL_MAPPING = {
    "gpt-4.1": ChatOpenAI,
    "gpt-4.1-mini": ChatOpenAI,
    "gpt-4o": ChatOpenAI,
    "gpt-4o-2024-08-06": ChatOpenAI,
    "chatgpt-4o-latest": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "gpt-4-turbo": ChatOpenAI,
    "gpt-4": ChatOpenAI,
    "gpt-3.5-turbo": ChatOpenAI,
    "gpt-3.5-turbo-1106": ChatOpenAI,
    "o3-mini": ChatOpenAI,
    "o4-mini": ChatOpenAI,
    "o3": ChatOpenAI,
    "gpt-5": ChatOpenAI,
    "gpt-5-mini": ChatOpenAI,
    "gpt-5.2": ChatOpenAI,
}

ANTHROPIC_MODEL_MAPPING = {
    "claude-opus-4-6": ChatAnthropic,
    "claude-sonnet-4-6": ChatAnthropic,
    "claude-sonnet-4-5": ChatAnthropic,
    "claude-haiku-4-5": ChatAnthropic,
    "claude-opus-4-0": ChatAnthropic,
    "claude-sonnet-4-20250514": ChatAnthropic,
    "claude-sonnet-4-0": ChatAnthropic,
    "claude-3-7-sonnet-latest": ChatAnthropic,
    "claude-3-7-sonnet-20250219": ChatAnthropic,
    "claude-3-5-sonnet-20240620": ChatAnthropic,
    "claude-3-opus-20240229": ChatAnthropic,
    "claude-3-sonnet-20240229": ChatAnthropic,
    "claude-3-haiku-20240307": ChatAnthropic,
}

GOOGLE_GENAI_MODEL_MAPPING = {
    "gemini-3.1-pro": ChatGoogleGenerativeAI,
    "gemini-3.1-pro-preview": ChatGoogleGenerativeAI,
    "gemini-3.1-flash": ChatGoogleGenerativeAI,
    "gemini-3.1-flash-preview": ChatGoogleGenerativeAI,
    "gemini-3.1-flash-lite": ChatGoogleGenerativeAI,
    "gemini-3.1-flash-lite-preview": ChatGoogleGenerativeAI,
    "gemini-3-pro-preview": ChatGoogleGenerativeAI,
    "gemini-3-flash-preview": ChatGoogleGenerativeAI,
    "gemini-2.5-flash": ChatGoogleGenerativeAI,
    "gemini-2.5-flash-lite": ChatGoogleGenerativeAI,
    "gemini-2.5-pro": ChatGoogleGenerativeAI,
    "gemini-2.5-flash-preview-05-20": ChatGoogleGenerativeAI,
    "gemini-2.5-flash-preview-04-17": ChatGoogleGenerativeAI,
    "gemini-2.5-pro-preview-06-05": ChatGoogleGenerativeAI,
    "gemini-2.5-pro-preview-05-06": ChatGoogleGenerativeAI,
    "gemini-2.5-pro-preview-03-25": ChatGoogleGenerativeAI,
    "gemini-2.5-pro-exp-03-25": ChatGoogleGenerativeAI,
    "gemini-2.0-flash": ChatGoogleGenerativeAI,
    "gemini-2.0-flash-001": ChatGoogleGenerativeAI,
    "gemini-2.0-flash-lite": ChatGoogleGenerativeAI,
    "gemini-2.0-flash-lite-001": ChatGoogleGenerativeAI,
    "gemini-1.5-pro": ChatGoogleGenerativeAI,
}

OPENAI_SDK_MAPPING = {k: OpenAI for k in OPENAI_MODEL_MAPPING.keys()}

ANTHROPIC_SDK_MAPPING = {k: Anthropic for k in ANTHROPIC_MODEL_MAPPING.keys()}

GOOGLE_GENAI_SDK_MAPPING = {k: GoogleClient for k in GOOGLE_GENAI_MODEL_MAPPING.keys()}

OPENAI_SDK_MODEL_MAPPING = {k: OpenAISdkModel for k in OPENAI_MODEL_MAPPING.keys()}

ANTHROPIC_SDK_MODEL_MAPPING = {
    k: AnthropicSdkModel for k in ANTHROPIC_MODEL_MAPPING.keys()
}

GOOGLE_GENAI_SDK_MODEL_MAPPING = {
    k: GoogleSdkModel for k in GOOGLE_GENAI_MODEL_MAPPING.keys()
}

MODEL_MAPPING = {
    **OPENAI_MODEL_MAPPING,
    **ANTHROPIC_MODEL_MAPPING,
    **GOOGLE_GENAI_MODEL_MAPPING,
}

SDK_MAPPING = {
    **OPENAI_SDK_MAPPING,
    **ANTHROPIC_SDK_MAPPING,
    **GOOGLE_GENAI_SDK_MAPPING,
}

SDK_MODEL_MAPPING = {
    **OPENAI_SDK_MODEL_MAPPING,
    **ANTHROPIC_SDK_MODEL_MAPPING,
    **GOOGLE_GENAI_SDK_MODEL_MAPPING,
}


class AutoChatModel(BaseAutoModel):
    _model_mapping = MODEL_MAPPING


class AutoSdkChatModel(BaseAutoSdkModel):
    _model_mapping = SDK_MAPPING
    _sdk_model_mapping = SDK_MODEL_MAPPING
