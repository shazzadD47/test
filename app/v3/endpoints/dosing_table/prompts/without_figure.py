STUDY_DESIGN_UNDERSTANDING_PROMPT_RAG = """
You are an expert in pharmacology research. You
are given a list of contexts that are related to a pharmacology research paper.
Your task is to understand the study design of the paper and extract the
information about the study design.

Contexts:
<contexts>
{contexts}
</contexts>

Instructions:

Analyze the contexts and extract the following information about the study
design:
1. The trail arms/groups
2. The type of study design
3. The number of groups and the names of the groups
4. Dosing regimen and route of adminstration
5. Dose of each group
6. Clearly state the dose escalation steps of each arm. For example, for an
  arm X 1 mg, if the doses are given in 3 steps, clearly state the steps
  like 0.25 mg in week 1 to 2, 0.5 mg in week 3 to 4, 0.25 mg in week 5
  to 6.
  If the paper has lead-in or run-in period (e.g., placebo lead-in),
  Ensure the placebo lead-in group is explicitly listed as its own row in
  the CSV
7.Duration of each group at each escalation step like week 1 to 2, week 3
  to 4, etc.
8. Inter-dose interval like 1 day, 8 hours, etc.
9. Steps of the study design
10. If start time is not perticularly mentioned, assume it to be day 0.


Some general guidelines:
- Our goal is to extract the dosing table from the contexts.
- Crearly give the time frame of each arm. For example, for an arm X 1 mg,
    if the doses are given in 3 steps, clearly state the steps like 0.25 mg
    in week 1 to 2, 0.5 mg in week 3 to 4, 0.25 mg in week 5 to 6.
- Give carefull attention to the escalation steps and make sure you do not
  miss a step.
- Provide as much information as possible.
- Your output will be used to generate a dosing table by an LLM.
"""

STUDY_DESIGN_UNDERSTANDING_PROMPT_DOC = """
You are an expert in pharmacology research. You are given a document related
to a pharmacology research paper. Your task is to understand the study design
of the document and extract the information about the study design.

Analyze the document and extract the following information about the study
design:
1. The trail arms/groups
2. The type of study design
3. The number of groups and the names of the groups
4. Dosing regimen and route of adminstration
5. Dose of each group
6. Clearly state the dose escalation steps of each arm. For example, for an
  arm X 1 mg, if the doses are given in 3 steps, clearly state the steps
  like 0.25 mg in week 0 to 3, 0.5 mg in week 4 to 7, 0.25 mg in week 8
  to 12 or End of the treatment.
7. Duration of each group at each escalation step like week 1 to 2, week 3
  to 4, etc.
8. Inter-dose interval like 1 day, 8 hours, etc.
9. Steps of the study design
10. If start time is not perticularly mentioned, assume it to be day 0.

LEAD IN:
If the paper describes a lead-in or run-in period (e.g., placebo lead-in)
starting before treatment, re-anchor the timeline so that the lead-in starts
at ARM_TIME = 0.
The subsequent treatment phase should then start at ARM_TIME equal to the
lead-in duration.
For example, if the lead-in is described from week -5 to 0 and treatment
from week 0 to 52, output ARM_TIME as 0 for lead-in and 5 for treatment,
and adjust the treatment duration accordingly from 52 to 57.
Ensure the placebo lead-in group is explicitly listed as its own row in the
CSV with the correct re-anchored ARM_TIME.
The smallest ARM_TIME must always be 0. Verify that the x-axis you describe
or report begins at 0.

DOSE ESCALTION:
  Clearly state the dose escalation steps of each arm. For example, for an
  arm X 1 mg, if the doses are given in 3 steps, clearly state the steps
  like 0.25 mg in week 0 to 3, 0.5 mg in week 4 to 7, 0.25 mg in week 8
  to 12, or end of the treatment
ARM_TIME:
Ensure that ARM_TIME reflects each escalation step accurately.
If the escalation occurs every fixed interval (e.g., every 4 weeks), the
ARM_TIME should increment accordingly:
Example: If escalation starts at week 0 → ARM_TIME = 0, 4, 8, 12.
If the paper specifies a different starting week (e.g., week 1), adjust
accordingly → ARM_TIME = 1, 5, 9, 13.
If the paper does not specify any starting time, default to starting at 0
and build the escalation timeline based on the given interval.
Make sure all escalation points follow the stated or assumed start time
consistently throughout the timeline

TOTAL DOSE CALCULATION:
    Calculate total how many times doses are given for each
    time step. For example 40mg X treatment: 0 to 2 weeks 10mg twice a day.
    So total doses in 0 to 2 weeks are 28 times.

    Group together the same amount of doses/ same arms. For example,
    treatment Y twice a day: week 1 10 mg, week 2 10 mg, week 3 15 mg. So
    group together 10 mg doses and 15 mg doses. Means 10 mg week 1 to 2
    doses are 14 * 2 = 28 doses and 15 mg week 3 doses are 7 * 2 = 14
    doses. Another example: Placebo week 1 to 4 twice a day total 56 doses,
    week 5 to 8 twice a day total 56 doses total 112 doses.
    For example, Treatment Z thrice a day for 52 weeks. Since there are 7
    days in a week, the total number of doses is calculated as 52 weeks × 7
    days × 3 doses per day = 1,092 doses.
    Therefore, Treatment Z administered thrice daily for 52 week results in
    a total of 1,092 doses.
    Make sure you capture the Start Time and End Time properly from the
    Methodology section of the paper

II:
    Also calculate the interval between doses for each arm. For example,
    40mg X treatment: 0 to 2 weeks 10mg twice a day. So the interval
    between doses is 12 hours or 0.5 days, 10mg X drug thrice daily the
    interval between doses is 8 hours.

ADDL CALCULATION:
   ADDL is the ADDITIONAL DOSES column. It's the additional dose given
   after the last dose. ADDL is 1 less than the total number of doses
   given. For example, if 3 doses are given, ADDL is 2. If 4 doses are
   given, ADDL is 3.
   ADDL= Total dose -1
Some general guidelines:
- Our goal is to extract the dosing table from the document.
- Crearly give the time frame of each arm. For example, for an arm X 1 mg,
    if the doses are given in 3 steps, clearly state the steps like 0.25 mg
    in week 1 to 2, 0.5 mg in week 3 to 4, 0.25 mg in week 5 to 6.
- Give carefull attention to the escalation steps and make sure you do not
  miss a step.
- Provide as much information as possible.
- Your output will be used to generate a dosing table by an LLM.
"""
