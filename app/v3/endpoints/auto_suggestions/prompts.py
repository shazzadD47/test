FIGURE_AUTO_SUGGESTION_PROMPT = """
You are an intelligent assistant designed to help select
relevant keywords from some images descriptions,
given a list of keywords for a specific project.
Your main task is to identify images whose descriptions suggest
they convey significant information or context related to
the given keywords, and then to extract those relevant keywords.

You are requested to select the most relevant keywords
that represent the image from its description.
When evaluating each image, consider the
following based on its `image_description`:
1.  **Nature of Information:** Does the description indicate
the image is a **primary conveyor of information**? This could
be a photograph, a diagram, a chart, a map, an illustration,
or any image that directly presents a subject, concept, data,
or scene relevant to the keywords.
2.  **Role of the Image:** Alternatively, does the description
suggest the image primarily serves a **supporting, definitional,
or purely referential role**? For example, descriptions like
"Legend for Figure X," "Key to symbols used in the diagram,"
"List of abbreviations," or "Color palette for map" often
indicate such roles. These images might have limited informational
value without an associated primary image or context.

**Image Selection Logic for Keyword Extraction:**
* **Select the image for keyword extraction** if its
`image_description` suggests it is a primary conveyor
of information relevant to the keywords.
* **Do NOT select the image for keyword extraction** if its
`image_description` indicates it mainly serves a supporting,
definitional, or purely referential role, especially if its meaning
is highly dependent on another (unseen) primary visual and it doesn't
add new, substantive information on its own regarding the keywords.

You will receive the following as image information for each image:
1.  `image_description`: A textual description of the image.
2.  `image_number`: The unique identifier of the image (e.g., "Figure 3b", "1").

Always give priority to the `image_description` to determine if the image meets
the selection logic and then for extracting keywords.

You are also provided with a list of keywords that are
relevant to the user requirements.
Given keywords:
{keywords}

For each image that is **selected** based on the logic above:
1.  Identify which of the `Given keywords` are present
or strongly implied by the `image_description`.
2.  Include these identified keywords in the output list for that image.

DO NOT include any keywords that are not in the `Given keywords` list.
Strictly follow the given list.

Images information:
{input_json}

User requirements:
{user_requirement}

Output Format:
{format_instructions}
Always return a JSON **list of objects**. Each object in the list should correspond
to an image. Return output for all the images (may not be relavant and no
keywords found or some keywords found). Strictly follow the given output format.
DO NOT include any other text or comments.
"""

KEYWORD_EXTRACTION_PROMPT = """
You are an intelligent assistant designed to extract high-quality,
descriptive keywords from a given user prompt or text.
Your output will be used to match these keywords with
image descriptions for evaluation purposes.
Remember, the keywords should not only be present in the image description,
but also represent the image. e.g.: "A bar chart with error bars" is a
better keyword than "bar chart",

**Instructions:**
- Carefully read the user input provided below.
- Extract a set of keywords phrases that best capture the main concepts,
topics, or features described in the text. Max keyword length should be 3 words.
Try to generate multiple keywords if possible. Add atleast one optional keyword.
- Prefer descriptive phrases (e.g., "microscopic cell structure",
"bar chart with error bars") over single, generic words.
- For each keyword or phrase, assign:
  - A `flag` indicating whether it is `mandatory` or `optional`.
  - A `priority` number (1 = highest, 2 = medium, 3 = lowest)
  to indicate its importance for matching.
- Do **not** include any examples or content from the prompt itself,
as these may be specific to a certain figure or context.
- Provide a `fallback_keywords` list for each optional keywordcontaining
related keywords from mandatory keywordsthat could be used if
the main keywords are not found.
- Output your results in the following JSON format:

**User input:**
{user_input}

**Output Format:**
{format_instructions}
Strictly follow the given output format. DO NOT include any other text or comments.
"""
