import json
from typing import Annotated

from fastapi import APIRouter, Form
from openai import OpenAI

from app.configs import settings

graph_prompt_router = APIRouter(tags=["V3_Text2graph"])


def convert_to_lowercase(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            output_dict[key] = [[item.lower() for item in sublist] for sublist in value]
        else:
            output_dict[key] = value.lower()
    return output_dict


@graph_prompt_router.post("/graph-prompt-test/")
async def graph_prompt_test(
    unstructured_text: Annotated[str, Form()],
    test_prompt: Annotated[str | None, Form()] = None,
):

    if test_prompt is None:
        test_prompt = """
            You are a Quantitative Systems Pharmacology Expert , your job is
            to mathematically model biological mechanisms.

            Given a text delimited by triple backticks., extrapolate as many
            relationships as possible from it and provide a list of updates.
            Don't do it excessively but add additional entities and relationships
            if it provides interesting context.

            Be very thorough with providing  relationships and ensure
            that there are no duplicates due to case or similar wording.

            If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2].
            The relationship is directed, so the order matters.
            the list should be a python list.

            if there is no relationship just send empty list like this: []

"""

    message_content = f"""
            {test_prompt}
            Example:
            text: Alice is Bob's roommate. Rob is Bob's classmate.
            output:{{"answer":[[
            "Alice", "roommate", "Bob"],["Bob", "classmate", "Rob"]]}}

            text: ```{unstructured_text}```
            output:
        """

    client = OpenAI()
    chat_completion, *_ = client.chat.completions.create(
        model=settings.GPT_4_TEXT_MODEL,
        messages=[
            {"role": "system", "content": "Please output valid JSON"},
            {"role": "user", "content": message_content},
        ],
        response_format={"type": "json_object"},
    ).choices
    content = chat_completion.message.content
    reply = json.loads(content)
    reply = convert_to_lowercase(reply)

    return reply
