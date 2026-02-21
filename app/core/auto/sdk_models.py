from app.core.auto.factory import _BaseSdkModel


class AnthropicSdkModel(_BaseSdkModel):
    _model_api_key_var_name = "ANTHROPIC_API_KEY"

    def _create_response(self):
        response = self.client.messages.create(model=self.model_name, **self.kwargs)
        return response


class GoogleSdkModel(_BaseSdkModel):
    _model_api_key_var_name = "GOOGLE_API_KEY"

    def _create_response(self):
        response = self.client.models.generate_content(
            model=self.model_name, **self.kwargs
        )
        return response

    def stream(self):
        response = self.client.models.generate_content_stream(
            model=self.model_name, **self.kwargs
        )
        yield from response

    def invoke_through_stream(self):
        response = self.stream()
        result = ""
        for chunk in response:
            result += chunk.text
        return result


class OpenAISdkModel(_BaseSdkModel):
    _model_api_key_var_name = "OPENAI_API_KEY"

    def _create_response(self):
        response = self.client.responses.create(model=self.model_name, **self.kwargs)
        return response
