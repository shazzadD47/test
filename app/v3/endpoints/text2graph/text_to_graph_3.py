import json
from typing import Annotated

from fastapi import APIRouter, Form
from openai import OpenAI

text_to_graph_router_3 = APIRouter(tags=["V3_Text2graph"])


def convert_to_lowercase(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            output_dict[key] = [[item.lower() for item in sublist] for sublist in value]
        else:
            output_dict[key] = value.lower()
    return output_dict


@text_to_graph_router_3.post("/text-to-graph/")
async def text_to_graph_3(unstructured_text: Annotated[str, Form()]):

    # Prompt to convert the unstructured text into graph
    message_content = f"""
            You are a Quantitative Systems Pharmacology Expert , your job is to
            mathematically model biological mechanisms.

            Given a text delimited by triple backticks,
            extrapolate as many relationships as possible from it and provide a
            list of updates. Don't do it excessively but add additional entities
            and relationships if it provides interesting context.
            Be very thorough with providing  relationships and ensure
            that there are no duplicates due to case or similar wording.

            If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2].
            The relationship is directed, so the order matters.
            the list should be a python list.

            if there is no relationship just send empty list like this: []


        output:{{
        "answer":[[
        ENTITY 1, RELATIONSHIP, ENTITY 2],[ENTITY 1, RELATIONSHIP, ENTITY 2]
        ]}}
        text: ```{unstructured_text}```

        """

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message_content}],
        temperature=1,
    )
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "model": response.model,
    }

    load2 = json.loads(response.choices[0].message.content)
    load2["usage"] = usage
    return load2
