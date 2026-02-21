from app.v3.endpoints.general_extraction.prompts.parts import (
    BEGINNING_PROMPT,
    COMPREHENSIVENESS_PROMPT,
    DEPENDENT_ANSWER_FORMAT_PROMPT,
    INSTRUCTION_BEGINNING_PROMPT,
    INSTRUCTION_ENDING_PROMPT,
)

LABEL_QUESTION_GENERATION_FROM_ROOT_LABELS_PROMPT = """
    <task>
    **Your task:**
    Your task is to analyze the label details and generate {total_questions}
    precise questions that will enable to extract all information regarding
    the label for every answer of the root label.

    You will be provided with:

        1.  `<label_details>`: This tag contains follwoing details about the
        **Primary Label**:
        * `name`: The name of the Primary Label.
        * `description`: A description of what the Primary Label represents.
        * `dependent_on`: A list of the **Root Label names** that the Primary
        Label is dependent on.

        <label_details>
        {label_details}
        </label_details>
        ```

    2.  `<combination_of_answers_of_root_label>`: This contains all the
        combinations of answers for the root labels identified in the `dependent_on`
        field of the Primary Label.

        <combination_of_answers_of_root_label>
        {combination_of_answers_of_root_label}
        </combination_of_answers_of_root_label>


    <instructions>
    **Instructions:**
     * Directly Pertain to the Primary Label:
        The core subject of the questions **must** be the Primary Label itself.
        To understand its scope and define the information to be found,
        you **must** refer to the Primary Label's `name` and `description`.
        The generated questions must contain what the user is trying to extract
        from the document about the label.

    * Create Questions for each combination of the Root Labels:
        Each combination of the root labels are provided in the
        <combination_of_answers_of_root_label>...</combination_of_answers_of_root_label>
        tag. For each combination of the root labels, the created question should
        incorporate the information of that combination of the answers of the
        root labels.

        If an answer of the root label is empty/null/'N/A' in a combination,
        create question for that combination without incorporating
        the information of that specific answer.

        But the number of created questions must be equal to the number of
        combinations of the root answers.

    * Variety in questions:
        Do not generate repeated questions that ask the same things
        in a different way. Only generate the exact amount of questions
        required to know about everything user wants to know about the label.
    </instructions>

    <output_format>
    **Output Format:**
    Questions should be well formatted and understandable by a human.
    Your output should be a list of {total_questions} question strings.
    Do not include any introductory phrases, concluding remarks, or any text
    outside of this JSON object.
    </output_format>


    <example>
    **Example 1:**

    Given:

    <label_details>
    <name>Software Bug Resolution Time</name>
    <description>Time taken to resolve reported software bugs, from reporting
    to closure.</description>
    <dependent_on>
    ["Bug Priority", "Software Module"]
    </dependent_on>
    </label_details>

    <combination_of_answers_of_root_label>
    Bug Priority,Software Module
    Critical,Module 1
    Critical,Module 2
    Critical,Module 3
    Critical,Module 4
    High,Module 1
    High,Module 2
    High,Module 3
    High,Module 4
    Medium,Module 1
    Medium,Module 2
    Medium,Module 3
    Medium,Module 4
    Low,Module 1
    Low,Module 2
    Low,Module 3
    Low,Module 4
    </combination_of_answers_of_root_label>

    Generated Output should be:
    [
    "What is the average resolution time for Critical bugs in Module 1?",
    "What is the average resolution time for Critical bugs in Module 2?",
    "What is the average resolution time for Critical bugs in Module 3?",
    "What is the average resolution time for Critical bugs in Module 4?",

    "What is the average resolution time for High priority bugs in Module 1?",
    "What is the average resolution time for High priority bugs in Module 2?",
    "What is the average resolution time for High priority bugs in Module 3?",
    "What is the average resolution time for High priority bugs in Module 4?",

    "What is the average resolution time for Medium priority bugs in Module 1?",
    "What is the average resolution time for Medium priority bugs in Module 2?",
    "What is the average resolution time for Medium priority bugs in Module 3?",
    "What is the average resolution time for Medium priority bugs in Module 4?",

    "What is the average resolution time for Low priority bugs in Module 1?",
    "What is the average resolution time for Low priority bugs in Module 2?",
    "What is the average resolution time for Low priority bugs in Module 3?",
    "What is the average resolution time for Low priority bugs in Module 4?",
    ]

    Explanation for Example 1: These questions target "Software Bug Resolution
    Time" (Primary Label name/description) and aim to "Analyze average
    resolution time by bug priority and by software module". They combine
    different values from the "Bug Priority" and "Software Module" Root Labels,
    ensuring the questions are focused on the aspects detailed in the Primary
    Label's analysis field.
    </example>

    -------------------------------------------------------------------------------
    </task>
"""

LABEL_CONTEXT_GENERATION_ROOT_PROMPT = """
    <task>
    **Your task:**
    - Based on the given inputs, create a table with rows containing
    all possible values for the following label:

    <label_details>
    {label_details}
    </label_details>

    <output_format>
    **Output format:**
    - Your output must be exactly a single JSON object with one field: "rows".
    - "rows" value must be an array where each array item is one possible value
    of the label.
    - Include every possible value you can find; do not omit any.
    - The table should be as complete as possible. Any missing value of the
    label from the table will result in a faulty table, leading to serious harm.
    - Do not put multiple values in a single row.
    - Row values must be unique. Do not repeat the same value in multiple rows.
    - row value validation: Each row value must have citations inside the
    <citation>...</citation> tag if you use the information from the given inputs.

    Examine the following example for reference:
    <example>
    **Given:**
    <label_details>
    label_name: moon
    label_description: moons of jupyter
    </label_details>

    **Generated Output should be:**
    {{
    "rows":
    [
        "Io <citation>...</citation>",
        "Europa <citation>...</citation>",
        "Ganymede <citation>...</citation>",
        "Callisto <citation>...</citation>",
        "Amalthea <citation>...</citation>",
        "Himalia <citation>...</citation>",
        "Elara <citation>...</citation>",
        "Pasiphae <citation>...</citation>",
        "Sinope <citation>...</citation>",
        "Lysithea <citation>...</citation>",
        "Carme <citation>...</citation>",
        "Ananke <citation>...</citation>",
        "Leda <citation>...</citation>",
        "Thebe <citation>...</citation>",
        "Adrastea <citation>...</citation>",
        "Metis <citation>...</citation>",
        "Callirrhoe <citation>...</citation>",
        "Themisto <citation>...</citation>",
        "Megaclite <citation>...</citation>",
        "Taygete <citation>...</citation>",
        "Chaldene <citation>...</citation>",
        "Harpalyke <citation>...</citation>",
        "Kalyke <citation>...</citation>",
        "Iocaste <citation>...</citation>",
        "Erinome <citation>...</citation>",
        "Isonoe <citation>...</citation>",
        "Praxidike <citation>...</citation>",
        "Autonoe <citation>...</citation>",
        "Thyone <citation>...</citation>",
        "Hermippe <citation>...</citation>",
        "Aitne <citation>...</citation>",
        "Eurydome <citation>...</citation>",
        "Euanthe <citation>...</citation>",
        "Euporie <citation>...</citation>",
        "Orthosie <citation>...</citation>",
        "Sponde <citation>...</citation>",
        "Kale <citation>...</citation>",
        "Pasithee <citation>...</citation>",
        "Hegemone <citation>...</citation>",
        "Mneme <citation>...</citation>",
        "Aoede <citation>...</citation>",
        "Thelxinoe <citation>...</citation>",
        "Arche <citation>...</citation>",
        "Kallichore <citation>...</citation>",
        "Helike <citation>...</citation>",
        "Carpo <citation>...</citation>",
        "Eukelade <citation>...</citation>",
        "Cyllene <citation>...</citation>",
        "Kore <citation>...</citation>",
        "Herse <citation>...</citation>",
        "S/2010 J1 <citation>...</citation>",
        "S/2010 J2 <citation>...</citation>",
        "Dia <citation>...</citation>",
        "S/2016 J1 <citation>...</citation>",
        "S/2003 J18 <citation>...</citation>",
        "S/2011 J2 <citation>...</citation>",
        "Eirene <citation>...</citation>",
        "Philophrosyne <citation>...</citation>",
        "S/2017 J1 <citation>...</citation>",
        "Eupheme <citation>...</citation>",
        "S/2003 J19 <citation>...</citation>",
        "Valetudo <citation>...</citation>",
        "S/2017 J2 <citation>...</citation>",
        "S/2017 J3 <citation>...</citation>",
        "Pandia <citation>...</citation>",
        "S/2017 J5 <citation>...</citation>",
        "S/2017 J6 <citation>...</citation>",
        "S/2017 J7 <citation>...</citation>",
        "S/2017 J8 <citation>...</citation>",
        "S/2017 J9 <citation>...</citation>",
        "Ersa <citation>...</citation>",
        "S/2011 J1 <citation>...</citation>",
        "S/2003 J2 <citation>...</citation>",
        "S/2003 J4 <citation>...</citation>",
        "S/2003 J9 <citation>...</citation>",
        "S/2003 J10 <citation>...</citation>",
        "S/2003 J12 <citation>...</citation>",
        "S/2003 J16 <citation>...</citation>",
        "S/2003 J23 <citation>...</citation>",
        "S/2003 J24 <citation>...</citation>",
        "S/2011 J3 <citation>...</citation>",
        "S/2016 J3 <citation>...</citation>",
        "S/2016 J4 <citation>...</citation>",
        "S/2018 J2 <citation>...</citation>",
        "S/2018 J3 <citation>...</citation>",
        "S/2018 J4 <citation>...</citation>",
        "S/2021 J1 <citation>...</citation>",
        "S/2021 J2 <citation>...</citation>",
        "S/2021 J3 <citation>...</citation>",
        "S/2021 J4 <citation>...</citation>",
        "S/2021 J5 <citation>...</citation>",
        "S/2021 J6 <citation>...</citation>",
        "S/2022 J1 <citation>...</citation>",
        "S/2022 J2 <citation>...</citation>",
        "S/2022 J3 <citation>...</citation>"
    ]
    }}
    --------------------------------------------------------
    </example>
    </output_format>

    {special_user_instructions}

    </task>
"""

LABEL_CONTEXT_GENERATION_PROMPT = f"""
{BEGINNING_PROMPT}
{DEPENDENT_ANSWER_FORMAT_PROMPT}
{INSTRUCTION_BEGINNING_PROMPT}
{COMPREHENSIVENESS_PROMPT}
{INSTRUCTION_ENDING_PROMPT}
"""
