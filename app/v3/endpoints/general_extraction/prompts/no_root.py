from app.v3.endpoints.general_extraction.prompts.parts import (
    BEGINNING_PROMPT,
    COMPREHENSIVENESS_PROMPT,
    INDEPENDENT_NON_ROOT_ANSWER_FORMAT_PROMPT,
    INSTRUCTION_BEGINNING_PROMPT,
    INSTRUCTION_ENDING_PROMPT,
)

LABEL_QUESTION_GENERATION_PROMPT = """
    <task>
    **Your task:**
    Create a specific question from the label name and description
    given in label_details tag that will allow to extract all information
    regarding the label.

    <label_details>
    {label_details}
    </label_details>

    <output_format>
    **Output format:**
    Do not include any decorative or irrelevant text.
    Only provide the created question.
    </output_format>

    <example>
    **Example:**
    <label_details>
    label_name: Moon
    label_description: Moons of Jupiter
    </label_details>

    **Generated Output should be:**
    "What are the moons of Jupiter?"
    </example>
    </task>
"""

LABEL_CONTEXT_GENERATION_NO_ROOT_PROMPT = f"""
{BEGINNING_PROMPT}
{INDEPENDENT_NON_ROOT_ANSWER_FORMAT_PROMPT}
{INSTRUCTION_BEGINNING_PROMPT}
{COMPREHENSIVENESS_PROMPT}
{INSTRUCTION_ENDING_PROMPT}
"""
