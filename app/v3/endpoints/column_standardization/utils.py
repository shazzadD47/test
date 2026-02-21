from langchain_core.prompts import PromptTemplate

from app.v3.endpoints.column_standardization.prompts import (
    COLUMN_STANDARDIZATION_SYSTEM_PROMPT,
    COLUMN_STANDARDIZATION_USER_PROMPT,
    UNIT_STANDARDIZATION_SYSTEM_PROMPT,
    UNIT_STANDARDIZATION_USER_PROMPT,
)


def create_column_standardization_prompt(
    column_name: str,
    column_description: str | None = None,
    user_instruction: str | None = None,
) -> PromptTemplate:
    """
    Create a customized prompt template for column standardization.

    Automatically detects if the column is a unit column (ends with '_UNIT')
    and uses the appropriate prompt template.

    Args:
        column_name: Name of the column being standardized
        column_description: Optional description of what the column contains
        user_instruction: Optional user-specific standardization instructions

    Returns:
        PromptTemplate configured for the specific column
    """
    # Determine if this is a unit column
    is_unit_column = column_name.lower().endswith("_unit")

    # Build description section
    description_section = ""
    if column_description:
        description_section = f"\n- **Column Description**: {column_description}"

    # Build user instruction section with emphasis
    instruction_section = ""
    if user_instruction:
        instruction_section = f"""

## 🎯 USER INSTRUCTIONS (HIGHEST PRIORITY)
**You must follow these instructions precisely:**

{user_instruction}

**These instructions override all default rules above.**
"""

    # Select appropriate prompts based on column type
    if is_unit_column:
        system_prompt = UNIT_STANDARDIZATION_SYSTEM_PROMPT
        user_prompt = UNIT_STANDARDIZATION_USER_PROMPT
    else:
        system_prompt = COLUMN_STANDARDIZATION_SYSTEM_PROMPT
        user_prompt = COLUMN_STANDARDIZATION_USER_PROMPT

    # Combine system and user prompts
    full_template = system_prompt + "\n\n" + user_prompt

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["values_str"],
        template=full_template,
        partial_variables={
            "column_name": column_name,
            "column_description_section": description_section,
            "user_instruction_section": instruction_section,
        },
    )

    return prompt
