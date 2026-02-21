CONTEXT_PROMPT_TEMPLATE_DOC = """Find the following information using
    the given document. Be as much elaborative as possible. If the
    information isn't in plain text, try to infer from the given document.
    Do not include phrases like 'from provided document'. Be concise. No
    need to include any irrelevant or decorative text.

    {question}

    Clearly state the steps of each arm. For example, for an arm X 1 mg, if
    the doses are given in 3 steps, clearly state the steps like 0.25 mg in
    week 1 to 2, 0.5 mg in week 3 to 4, 0.25 mg in week 5 to 6.

    Return the required information only. Nothing else. Never include any
    decorative or irrelevant text. If you do not find an information in the
    document, return N/A for string values and null for numerical values.
"""

CONTEXT_PROMPT_TEMPLATE_RAG = """Find the following information using
    the given contexts. Be as much elaborative as possible. If the
    information isn't in plain text, try to infer from the given contexts.
    Do not include phrases like 'from provided contexts'. Be concise. No
    need to include any irrelevant or decorative text.

    <question>
    {question}
    </question>

    Use the following contexts:
    <contexts>
    {contexts}
    </contexts>

    Clearly state the steps of each arm. For example, for an arm X 1 mg, if
    the doses are given in 3 steps, clearly state the steps like 0.25 mg in
    week 1 to 2, 0.5 mg in week 3 to 4, 0.25 mg in week 5 to 6.
    Ensure the placebo lead-in group is explicitly listed as its own row in
    the CSV, with the correct re-anchored ARM_TIME.

    Return the required information only. Nothing else. Never include any
    decorative or irrelevant text. If you do not find an information in the
    context, return N/A for string values and null for numerical values.
"""

IMAGE_EXPLANATION_PROMPT_CLAUDE = """You are given an image. At first
    describe the image including what are the x and y axis represent, what
    are the ticks, units etc and find the following information from the
    image.
    - What are x axis ticks like [..., -1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
    - What is x axis unit like days, weeks
    If the figure shows a lead-in or run-in period (e.g., placebo lead-in)
    starting at a negative week on the x-axis, re-anchor the timeline so that
    the lead-in starts at ARM_TIME = 0 and the subsequent treatment period
    starts at ARM_TIME = [lead-in duration].
    For example, if the x-axis shows -5 → 0 for lead-in, and 0 → 52 for
    treatment, output the ARM_TIME as 0 for lead-in and 5 for treatment,
    and also the treatment duration will be increased drom 52 to 57.
    Ensure the placebo lead-in group is explicitly listed as its own row in
    the CSV, with the correct re-anchored ARM_TIME.
    Do not use negative values for ARM_TIME in the CSV. The smallest
    ARM_TIME should be 0
    Verify your described x-axis now begins at 0 (not –5, –1, etc.)

    Then do the following things:

    From the given image find out what are the does arms. At which day what
    amount of dose were given for each arm. If the amount of doses are
    separated by shads of colors, try to infer the time and amount looking
    at the shades. At first find the key visits then do the listing. Give
    careful attention to the dose giving pattern, starting time and ending
    time (EOT) of the doses. Correctly predict if starts at day 0 or 1 or
    any other day. Try to give the time in range like day 0 to 1, 1 to 5,
    etc.


    If you find a table in the figure, at first digitize the table exactly
    and then answer the question.

    Our goal is to find the following information:
    - Dosing arm/group
    - Dose
    - Route of administration
    - Interval between doses
    - Total number of doses

    Use the following contexts:
    <contexts>
    {contexts}
    </contexts>

    Clearly identify the dosing pattern and steps. Take help from the
    contexts, but the image is the main source of information. If the image
    states that a dose has 2 steps like week 1 to 2 and week 3 to 10 but
    the context states that the dose has 3 steps like week 1 to 2, week 3
    to 5, and week 6 to 10, then the image is the correct.

    Some terms:
    EOT = End of Treatment
    FU = Follow-up

    Also include route of administration and interval between doses for each
    arm.
"""

DOSE_CALCULATION_PROMPT = """Calculate total how many times doses are given
    in each time step. For example 40mg X treatment: 0 to 2 weeks 10mg twice
    a day. So total doses in 0 to 2 weeks are 28 times.

    Group together the same amount of doses/ same arms. For example,
    treatment Y twice a day: week 1 10 mg, week 2 10 mg, week 3 15 mg. So
    group together 10 mg doses and 15 mg doses. Means 10 mg week 1 to 2
    doses are 14 * 2 = 28 doses and 15 mg week 3 doses are 7 * 2 = 14 doses.
    Another example: Placebo week 1 to 4 twice a day total 56 doses, week 5
    to 8 twice a day total 56 doses total 112 doses.



    Also calculate the interval between doses for each arm. For example, 40mg
    X treatment: 0 to 2 weeks 10mg twice a day. So the interval between doses
    is 12 hours or 0.5 days, 10mg X drug thrice daily the interval between
    doses is 8 hours.

    Remember placebo are also given as dose. So include them in the
    calculation.

    Use the following information and explanation of the image:
    <contexts>
    {contexts}
    </contexts>

    Retain as much information as possible from the given information and
    explanation. Make sure your calculations are correct.
"""

INFORMATION_EXTRACTION_PROMPT = """
Find the required information using the given contexts. Return the
information for each line/arm separately.

Use the given contexts in the best way possible. If the information can not
be found in plain text, use your research and reasoning skills to infer the
information from the contexts.

The given contexts are:
<contexts>
{contexts}
</contexts>

Return only the required information. Do not include any decorative or
irrelevant text. Return the information in a structured JSON format
according to the expected schema.

Use good names for the treatments. Some examples of bad names and good names
are:
| ----------------------------------- | ------------------ | ------------------ |
| bad name                            | good name          | standard treatment |
| ----------------------------------- | ------------------ | ------------------ |
| Semaglutide 2.4 mg s.c. once-weekly | Semaglutide 2.4 mg | Semaglutide        |
| Placebo twice a day                 | Placebo            | Placebo            |
| X 10 mg twice a day                 | X 10 mg            | X                  |
| ----------------------------------- | ------------------ | ------------------ |

Always give the abbreviations of route instead of full name. Look at the
following table for abbreviations:

| full name       | abbreviation |
| --------------- | ------------ |
| oral            | oral         |
| subcutaneous    | sc           |
| intravenous     | iv           |
| intramuscular   | im           |

AMT for placebo is always 0. Give unit match with the unit of the other
doses.

If there are multiple correct values for a single field and list is not
expected, create a new field for each value. For example, if there are
multiple values for "dose", i.e. [1, 2, 3, 4, 5] for a single treatment
arm such as "A", create a new field for each value. Like ("treatment": "A",
"dose": 1), ("treatment": "A", "dose": 2), and so on.

ADDL is the ADDITIONAL DOSES column. It's the additional dose given after
the last dose. ADDL is 1 less than the total number of doses given. For
example, if 3 doses are given, ADDL is 2. If 4 doses are given, ADDL is 3.

Group together the same amount of doses/ same arms. For example, treatment Y
twice a day: week 1 10 mg, week 2 10 mg, week 3 15 mg. So group together
10 mg doses and 15 mg doses. Means 10 mg week 1 to 2 doses are 14 * 2 = 28
doses and 15 mg week 3 doses are 7 * 2 = 14 doses. Another example: Placebo
week 1 to 4 twice a day total 56 doses, week 5 to 8 twice a day total 56
doses. So group together week 1 to 8 doses with total 112 doses.

If start time is not given, try to infer from the given information. If you
can not infer, then assume it's 0.

Carefully determine the interval between doses. For example, if the doses
are given twice a day, the interval between doses is 12 hours. If the doses
are given once a day, the interval between doses is 24 hours or 1 day, if
the dose are given thrice daily, the interval between doses is 8 hours.

Match time units of all rows of a same column. For example, if one row is
in days and another row is in weeks, convert all rows to same unit,
preferably to smaller unit like days. Also, II Unit must match ARM time
unit.
"""
