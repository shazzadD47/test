import os

from fastapi import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel, BaseLanguageModel

from app.core.auto.token_config import MAX_TOKEN_MAPPING
from app.core.utils.hooks import hook_manager


class _BaseFactory:
    _model_mapping: dict = None

    def __init__(self) -> None:
        raise OSError(
            f"{self.__class__.__name__} should not be instantiated directly. "
            f"Use {self.__class__.__name__}.from_model_name() instead."
        )

    @classmethod
    def from_model_name(cls, model_name: str, **kwargs):
        if cls._model_mapping is None:
            raise NotImplementedError(
                f"{cls.__name__} does not have a model mapping. "
                f"Please define a `_model_mapping` class attribute."
            )

        model_class = cls._model_mapping.get(model_name)

        if not model_class:
            raise ValueError(
                f"Model name '{model_name}' is not valid for {cls.__name__}."
            )

        return cls._create_model(model_class, model_name, **kwargs)

    @classmethod
    def _create_model(
        cls, model_class, model_name: str, **kwargs
    ) -> BaseLanguageModel | BaseChatModel:
        raise NotImplementedError(
            f"{cls.__name__}.create_model() must be implemented in a subclass."
        )

    @classmethod
    def register(cls, model_name: str, model_class: BaseLanguageModel):
        if cls._model_mapping is None:
            cls._model_mapping = {}

        cls._model_mapping[model_name] = model_class


class BaseAutoModel(_BaseFactory):
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs):
        if cls._model_mapping is None:
            raise NotImplementedError(
                f"{cls.__name__} does not have a model mapping. "
                f"Please define a `_model_mapping` class attribute."
            )

        model_class = cls._model_mapping.get(model_name)

        if "max_tokens" not in kwargs:
            max_tokens = MAX_TOKEN_MAPPING.get(model_name, 4096)
            kwargs["max_tokens"] = max_tokens

        if "temperature" in kwargs and model_name.startswith("o"):
            del kwargs["temperature"]

        if not model_class:
            raise ValueError(
                f"Model name '{model_name}' is not valid for {cls.__name__}."
            )

        return cls._create_model(model_class, model_name, **kwargs)

    @classmethod
    def _create_model(
        cls, model_class, model_name: str, **kwargs
    ) -> BaseLanguageModel | BaseChatModel:
        return model_class(model=model_name, **kwargs)


class _BaseSdkModel:
    _model_api_key_var_name: str = None

    def __init__(self, model_class, model_name: str, **kwargs):
        try:
            self.client = model_class(api_key=os.getenv(self._model_api_key_var_name))
        except Exception:
            raise ValueError(
                f"Set {self._model_api_key_var_name}" f" env variable for {model_name}."
            )
        self.model_name = model_name
        self.kwargs = kwargs
        if "callbacks" in kwargs:
            self.callbacks = kwargs.pop("callbacks")
        else:
            self.callbacks = []

    def _create_response(self):
        raise NotImplementedError(
            f"{self.__class__.__name__}._create_response() "
            "must be implemented in a subclass."
        )

    def stream(self):
        raise NotImplementedError(
            f"{self.__class__.__name__}.stream() " "must be implemented in a subclass."
        )

    def invoke(self):
        response = self._create_response()
        self.execute_callbacks(response)
        self.trigger_hooks(response)
        return response

    def execute_callbacks(self, response):
        for callback in self.callbacks:
            if hasattr(callback, "on_llm_end"):
                callback.on_llm_end(response)
            else:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(
                        "Error executing callback "
                        f"{callback.__class__.__name__} "
                        f"for model {self.model_name}: {e}"
                    )

    def trigger_hooks(self, response):
        try:
            hook_manager.trigger(
                "on_llm_end", response=response, model_name=self.model_name
            )
        except Exception as e:
            logger.error("Error triggering hook " f"for model {self.model_name}: {e}")


class BaseAutoSdkModel(BaseAutoModel):
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs):
        if cls._model_mapping is None:
            raise NotImplementedError(
                f"{cls.__name__} does not have a model mapping. "
                f"Please define a `_model_mapping` class attribute."
            )
        if cls._sdk_model_mapping is None:
            raise NotImplementedError(
                f"{cls.__name__} does not have a sdk model mapping. "
                f"Please define a `_sdk_model_mapping` class attribute."
            )

        model_class = cls._model_mapping.get(model_name)
        sdk_model_class = cls._sdk_model_mapping.get(model_name)

        if not model_class:
            raise ValueError(
                f"Model name '{model_name}' is not valid for {cls.__name__}."
            )
        if not sdk_model_class:
            raise ValueError(
                f"Sdk model name '{model_name}' is not valid for {cls.__name__}."
            )

        return cls._create_model(model_class, sdk_model_class, model_name, **kwargs)

    @classmethod
    def _create_model(
        cls, model_class, sdk_model_class, model_name: str, **kwargs
    ) -> _BaseSdkModel:
        return sdk_model_class(model_class, model_name, **kwargs)


class BaseAutoCallbackHandler(_BaseFactory):
    @classmethod
    def _create_model(
        cls, model_class, model_name: str, **kwargs
    ) -> BaseCallbackHandler:
        return model_class(**kwargs)
