BEGINNING_PROMPT = """
    <task>
    **Your task:**
    - Based on the given inputs, answer the given questions
    about the following label:

    <label_details>
    {label_details}
    </label_details>

    <questions>
    {questions}
    </questions>

    Total number of questions: {total_questions}
"""

INSTRUCTION_BEGINNING_PROMPT = """
     <instructions>
     You must follow the given instructions:

    **Instructions:**
    {special_user_instructions}

    **Must generate answers for all the questions:**
        You must provide answers for all the given questions.

        Derive all the answers using the given inputs.
        Do not make up any information.

    **Order of the answers:**
    - The answers should be provided in the order of the questions.

"""

COMPREHENSIVENESS_PROMPT = """
    ** Comprehensiveness:**
        Remember that you are conveying the information to a human user.
        Therefore, each answer should be well formatted with proper
        sections and subsections (if required), easily understandable,
        comprehensive and detailed containing all information asked in
        the question unless otherwise specified in the User Instructions
        or the label details.
"""

INSTRUCTION_ENDING_PROMPT = """

    </instructions>
"""


DEPENDENT_ANSWER_FORMAT_PROMPT = """
    <answer_format>
    **Format of the answers:**
    Provide answers of each question in the "answers" field starting
    from the first question and moving to the last question.

    The first item in the "answers" field are all the possible
    answers to the first question,
    the second item in the "answers" field are all the possible
    answers to the second question,
    and so on. The length of the 'answers' field must be equal to the
    total number of questions.

    If there are multiple possible answers for a single question,
    join all of them with commas. Do not miss any possible answer.

    -------------------------------------------------------------------------------
    **Format:**
    {{"answers": [
    "all_answers_for_question_1",
    "all_answers_for_question_2",
    "all_answers_for_question_3", ... , "all_answers_for_question_n"]}}
    -------------------------------------------------------------------------------
    **Example:**
    Questions:
    1. What is the moon of Mars?
    2. What is the moon of Earth?

    Answer format:
    {{"answers": ["Phobos, Deimos <citation>...</citation>", \
    "Moon <citation>...</citation>"]}}
    -------------------------------------------------------------------------------

    b. **Must support your answers with proper citations:**
    You must support your answers with proper citations. Each piece of
    information in the answer that is coming from the provided inputs
    must be supported with a citation inside the <citation>...</citation> tag.
    </answer_format>
"""


INDEPENDENT_NON_ROOT_ANSWER_FORMAT_PROMPT = """
    <answer_format>
    **Format of the answers:**
    Provide answers to all the questions in a ordered
    list manner. If there is a single question, just
    provide the answer - no need to mention the question number.

    **Format:**
    If there are multiple questions:
    1. answer for question 1 <citation>...</citation>
    2. answer for question 2 <citation>...</citation>
    3. answer for question n <citation>...</citation>
    ...
    If there is a single question:
    answer for question 1 <citation>...</citation>

    **Must support your answers with proper citations:**
    You must support your answers with proper citations. Each piece of
    information in the answer that is coming from the provided materials
    must be supported with a citation inside the <citation>...</citation> tag.

    **Multiple answers for a single question:**
    If there are multiple possbile answers for a single question,
    provide all of the answers. Do not miss any possible answer.

    For example, if the question is "What is the moon of Jupiter?",
    and the answer will be "Europa, Io, Ganymede, Callisto, ... , nth moon".
    </answer_format>
"""
