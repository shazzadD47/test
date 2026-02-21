def return_llm_sdk_cost(handler, response, model_name: str):
    """
    Hook function to capture and process LLM SDK cost information.

    This function is called when an LLM operation ends and is responsible for
    extracting usage metadata from the response and passing it to the handler
    for cost tracking purposes.

    Args:
        handler: The callback handler that will process the usage metadata.
        response: The response object from the LLM SDK containing usage information.
        model_name (str): The name of the model that was used for the LLM operation.
    """
    handler.on_llm_end(response, model_name)
