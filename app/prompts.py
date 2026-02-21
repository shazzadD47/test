QUESTION_GENERATION_PROMPT = """You are an expert AI scientist. You need
    to rephrase the given questions in a more professional and scientific and generate
    TWO different version of it. Be as concise as possible. Your goal is to generate
    questions so that contexts retrieved from paper is maximized.


    The given question is:

    ```
    {question}
    ```

    Here are definitions of the term from glossary:

    {output_format}
"""


QUESTION_PARAPHRASING_PROMPT = """You are an expert AI scientist. You need
    to rephrase the given questions in a more professional and scientific way to
    retrieve information. Be as concise as possible. Your goal
    is to generate questions so that contexts retrieved from paper is maximized.


    The given question is:

    ```
    {question}
    ```
"""

CONTEXT_PROMPT_TEMPLATE = """Find the following information using
    the given contexts. Be short, concise and correct. If the information
    isn't in plain text, try to infer from the given contexts. Do not include
    phrases like 'from provided contexts'.

    {question}

    The digitized tables are:

    {tables}

    Use the following contexts:

    {contexts}
"""
