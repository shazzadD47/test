FIND_INFO_FROM_FIGURE_PROMPT_SYSTEM_MESSAGE = """
    You are a very powerful AI assistant
    who is capable of extracting information from plot image accurately.
"""

FIND_LEGEND_NAMES_PROMPT = """
    Find the valid legend names inside the given plot image.
    plot image type: {plot_image_type}

    if plot image type is "N/A", firstfind the plot type of the given image.
    plot type can be: line/bar/box/scatter/kaplan-meier-curve/other

    Instructions:
    1. Legends Must be from inside the given plot image.
    NEVER hallucinate any legend names.

    2. Number of legends must be equal to the number of
    lines/bars in the plot image.

    3. If legends have colors/marker colors,
    they must match with the colors in the lines/bars. If not,
    NEVER return them as legend names.
    Otherwise, NEVER return them as legend names.

    3. For bar plots, if you do not find any legends,
    return the names of each bar as legends.
    Otherwise, return legend names concatenated via '_'
    with bar names. Bar names are the labels for each bar
    in the bar plot.

    4.  If you do not find any legends and no bar names
    for bar plot, return [line_1,line_2,...line_N] where
    N is the number of distinct lines/bars in the given plot image.
    Strictly adhere to the above instructions.

    5. Check captions, footnote or other texts in the plot image
    for finding legend names.

    The output must be in valid JSON format.
    Strictly follow the output format.
    Do not add any extra information or
    decorative text in the output.
"""

FIND_NUMBER_OF_LINES_PROMPT = """
    Find the number of distinct lines/bars in the given plot image.
    The output must be in valid JSON format.
    Strictly follow the output format.
    Do not add any extra information or decorative text in the output.
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

    Do not provide any explanation regarding output.
    Do not put 'Rephrased question:' or 'Output:' in the output.

    The given question is:
    <question>
    {question}
    </question>
"""

FIGURE_INFO_EXTRACTION_PROMPT = """
    Your main task is to answer the questions from the plot image.
    You need to observe the image carefully.
    Provide special care to the number of lines.

    Figure Data:
    <figure_data>
    {figure_data}
    </figure_data>
    Figure data may contain figure number,
    summary, caption, footnote, etc.
    It may also contain the caption and footnote of the
    parent of this figure. You MUST try your best to utilize
    all this information to extract the required information.

    Questions:
    <questions>
    {questions}
    </questions>

    The figure might contain float values which you must extract
    with full precision. You MUST try your best to answer all
    questions from the plot image.
    If you do not find the information in the image,
    return N/A for string values and null
    for numerical values.

    Despite your best efforts, if you cannot find the answer in
    the figure, return 'N/A' for string values and null for
    numerical values.
    Output MUST be in VALID JSON format.
    Do not add any explanation or decorative text.
"""


LEGEND_MAPPING_PROMPT = """
    You are a legend mapping expert.
    You will be given with the legends list from the AI model
    and the legends list from the autofill pipeline.
    Legends from autofill pipeline are correct.
    Legends from AI models are not correct and it gives legend
    names close to the original one.
    You need to map each legend from the autofill pipeline to
    the legend from the AI pipeline
    that closely matches with the legend from the autofill pipeline.
    You are also provided with the plot image. Carefully look into the
    image for any other legend mapping information that might not be
    interpretable from the legend text only.
    ai_model_legends: {ai_model_legends}
    autofill_legends: {autofill_legends}
    You must output in the following format:
    {output_format_instructions}
    Output MUST be in VALID JSON format.
    DO NOT modify any name in the output of
    ai_model_legends and autofill_legends from the input one.
    Strictly follow the output format.
    Do not add any extra information or
    decorative text in the output.
"""

VISION_AGENT_BAR_PLOT_PROMPT = """
    Select the top of vertical bar of '{BAR_INFO}' in the bar graph.
"""


GEMINI_AXIS_VALUES_EXTRACTION_PROMPT = """
      You are an expert data analyst specializing in interpreting plots.

      I am providing a plot image. Analyze the plot carefully and extract
      the numeric values for:

      - minimum and maximum values of the x-axis labels
      - minimum and maximum values of the y-axis labels

      **Guidelines for Plots:**
      - The x-axis and y-axis may show positive and negative ranges.
        Ensure both positive and negative extremes are captured accurately.
      - Do not approximate any axis label values from plotted bars,
        lines, or data points.
      - If the x-axis has **categorical labels** or **no visible numeric labels**,
        assign sequential numeric indices starting from 0 up to (N - 1),
        where N is the total number of bars or categories.
      - If there is a broken axis, use the **larger visible portion**
        of the axis for extracting min and max.
      - Extract values so that all labels from min to max follow
        a **linear numeric pattern** based on the axis scale.
      - Consider only visible tick labels, gridlines,
        or axis structure for numeric extraction.
        Do not include any additional text, explanation, or formatting.
      """
