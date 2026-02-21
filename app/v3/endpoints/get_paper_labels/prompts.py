QUESTION_PARAPHRASING_PROMPT = """You are an expert pharmacometrics scientist.
    Your MAIN task is to rephrase the given question in a way that improves clarity,
    coherence and comprehensibility of the question without changing the meaning of the
    question. Include arm/treatment names in the rephrased question if they are present
    in the original question. Ensure that the rephrased question preserves the meaning
    of the original question.

    Add after `Suggestion regarding where to find:` about possbile locations
    (maximum 3) of the target information inside the research paper/figure/table/other.
    If the original question contains such location information,
    add that in the suggestion. Add numbering (1.,2.,3.,...) for multiple
    possible positions.

    Do not provide any explanation regarding output.
    Do not add 'Rephrased Question:' in the output.

    The given question is:
    {question}

    Do not add any explanation or decorative text.
"""

QA_ON_CONTEXT_PROMPT = """
    You are an expert pharmacometrics scientist.
    Find the following information using the given contexts.

    Question:
    {question}

    Give specific answers to the questions.
    Do not add any decorative text or explanation of your
    answer. Do not repeat the question in the answer.

    If the information isn't in plain text or not provided explicitly
    in contexts, try to infer from the given contexts.

    Use the following contexts:
    {contexts}
"""

ANSWER_FROM_CONTEXT_PROMPT = """
    You are a pharmacometrics scientist. You need to
    find some specific information from given contexts
    based on the questions asked. Use the given contexts properly
    to answer all the questions. If the information can not be found
    in plain text, use your research and reasoning skills to infer
    the information from the contexts.

    Questions with their respective contexts are:
    {contexts}

    For questions that require prior infromation, first find that
    information and then answer the question based on it. For
    example, if the question asks you to answer the treatment class
    of an arm, first find the arm name and then answer the treatment
    class of that arm.

    If the question asks for a specific value, be specific.
    Otherwise, if the question asks for a summarization/explanation/description,
    be elaborate.

    If any information is missing in the contexts, then return "N/A"
    for string values and null for numerical values.
    Do not make up outputs on your own - your answers
    must be from the provided contexts.

    Output Format:
    {output_instructions}
    Strictly follow the output format.
    Output MUST be in VALID JSON format. Do not add any
    explanation or decorative text.

"""
