SYSTEM_PROMPT = """
    You are an expert pharmacometrics scientist.
"""
QA_ON_CONTEXT_ADDITIONAL_INSTRUCTIONS = """
    Additional output instructions:
    You must cite the chunks you used to derive your answer.
    Information derived from chunks in the answer must have
    in-line citations. For example,
    If you used chunks 1 and 8 - you must cite them as <cite>[1,8]</cite>.
    total_number_of_chunks = {total_number_of_chunks}
    The cited chunks must be in the range of 1 to total_number_of_chunks
    inclusive.
    All answers must be cited.
    Strictly follow the output format.

    The answer should be well-explained, detailed and comprehensive.
    You MUST not add any information outside of the provided contexts.
    Do not repeat the prompt in the answer.
    Try your best to adhere to the instructions in the prompt.

    If answer cannot be found in the provided contexts,
    then retrim "Required information was not found in the paper"
    + a detailed explanation about why the answer cannot be found and
    what information was actually found in the provided contexts
    including any close matches found in the provided contexts
    (MUST include in-text citations for the found information).
    You MUST NOT make up any information or provide
    any information outside of the provided contexts.
"""


QA_ON_CONTEXT_PROMPT = """
You are an expert pharmacometrics scientist.
Find the answers to the following prompts using the given contexts.

Prompts:
<prompts>
{questions}
</prompts>

Contexts:
<contexts>
{contexts}
</contexts>
"""


NESTED_LABEL_CONTEXT_INSTRUCTION = """
    Some labels may be dependent on the values of other labels
    These parent labels' answers are provided which should be used
    in conjunction with the contexts to answer the prompt.

    Parent label answers:
    <parent_label_answers>
    {parent_label_answers}
    </parent_label_answers>

    The 'Related How' field describes how the current label is related
    to the parent label which you can use to create your answer.

    For example, if the question asks for treatment dosage of a specific
    arm, and the related label is 'arm' and the 'Related How' field is
    'Dosage for that specific arm', then use the 'arm' label's answers to
    find the specific dosage for that arm. So if arm is
    ['Tirzepatide', 'Placebo'],find the dosage for Tirzepatide and Placebo.
"""


ANSWER_FROM_CONTEXT_PROMPT = """
You are an expert pharmacometrics scientist. You need to
extract the correct information about labels from the
given contexts.

Information about some labels might already be given
which you can use to find the answers for the other labels.
Already given information:
<labels_with_answers>
{labels_with_answers}
</labels_with_answers>
The given information can be updated by you to give the
answers for the other labels.

Labels with their respective contexts are:
<contexts>
{contexts}
</contexts>

Instructions:
1. If there are multiple correct values,
create a new entry for each correct value.

If the label has a correct answer for different values of
related labels, then create a new entry for each correct answer
for all combinations of values of related labels.

For example, Let's assume the target label is "T2D_STATUS_PERCENT_BL"
(type 2 diabetes percentage, datatype: float) and its related labels
are "trial_arm" (datatype: string) and "gender" (datatype: string).
Values of related labels are:
trial_arm: "1. A, 2. B"
gender: "1. Male, 2. Female"
Then return the answer for "T2D_STATUS_PERCENT_BL" in following format:
[{{"trial_arm": "A", "gender": "Male", "T2D_STATUS_PERCENT_BL": 10}},
 {{"trial_arm": "B", "gender": "Male", "T2D_STATUS_PERCENT_BL": 20}},
 {{"trial_arm": "A", "gender": "Female", "T2D_STATUS_PERCENT_BL": 30}},
 {{"trial_arm": "B", "gender": "Female", "T2D_STATUS_PERCENT_BL": 40}}]
Answer format explanation:
T2D_STATUS_PERCENT_BL is dependent on trial_arm and gender.
It is 10 for trial_arm A and gender Male,
20 for trial_arm B and gender Male, and so on so forth.
Hence, the related labels are broken to their single values as the
target label T2D_STATUS_PERCENT_BL contains a correct answer
for different values of related label.
So, in the final output, trial_arm will be two separate values ["A", "B"]
and gender will be two separate values ["Male", "Female"].

2. Use the given contexts properly to extract the information.
All the information must be from the provided contexts.

5. If any information is missing in the contexts,
then return null for numerical values.

6. Do not make up outputs on your own - your answers
must be from the provided contexts.

Strictly follow the instructions.

Your answer must be in VALID JSON format.
Do not add any explanation or decorative text.
"""

QUESTION_PARAPHRASING_PROMPT = """
    You are an expert pharmacometrics scientist.
    Your MAIN task is to rephrase the given question in a way
    that improves the  clarity, coherence and comprehensibility
    of the question, and helps to extract the necessary answer
    from the paper while keeping intact all the original
    information and instructions in the original question.
    Include arm/treatment names in the rephrased question if
    they are present in the original question. Ensure that the
    rephrased question preserves the meaning of the original
    question and contains all information of the original question.

    Add after `Suggestion regarding where to find:` about possbile locations
    (maximum 3) of the target information inside the research paper/figure/table/other.
    If the original question contains such location information,
    add that in the suggestion. Add numbering (1.,2.,3.,...) for multiple
    possible positions.

    Do not provide any explanation regarding output.

    The given question is:
    <question>
    {question}
    </question>
"""
