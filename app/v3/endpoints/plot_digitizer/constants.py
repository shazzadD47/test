from dataclasses import dataclass

from app.core.auto import AutoChatModel
from app.v3.endpoints.plot_digitizer.chains import (
    prepare_question_rephrasing_chain,
)
from app.v3.endpoints.plot_digitizer.configs import settings as plot_digitizer_settings


@dataclass(frozen=True)
class ErrorCode:
    CLAUDE_IMAGE_PROCESSING_FAILED = "Failed to process image with Claude."
    CLAUDE_RESPONSE_PARSING_FAILED = "Failed to parse response from Claude."

    OPENAI_IMAGE_PROCESSING_FAILED = "Failed to process image with OpenAI."
    OPENAI_RESPONSE_PARSING_FAILED = "Failed to parse response from OpenAI."

    QUESTION_REPHRASING_FAILED = "Failed to rephrase question."
    CONTEXT_SUMMARIZATION_FAILED = "Failed to summarize context."
    CONTEXT_QA_FAILED = "Failed to get ANSWER from Contexts."

    LINE_FORMER_FAILED = "Failed to run lineformer model"


@dataclass(frozen=True)
class Florence2Request:
    TIMEOUT = 210
    individual_line_black_threshold = 30
    point_black_threshold = 6
    point_radius = 12


@dataclass(frozen=True)
class ChartDete:
    EXTRACTION_THRESHOLD = 0.3
    AXES_DETECTION_THERESHOLD = 0.4
    LEGEND_PICK_THEREHOLD = 0.4
    PATCH_FIELD_NAME = "legend_patch"
    LABEL_FIELD_NAME = "legend_label"
    UPLOAD_PATH = "plot-digitizer/chart_dete"


@dataclass(frozen=True)
class OCR:
    LABLE_EXTRACT_Y_THRESHOLD = 11
    PAD = 5


@dataclass(frozen=True)
class LineFormerRequest:
    # Time in second
    TIMEOUT = 30

    # extend line
    INTERPOLATE_LENGTH = 30
    NEW_POINT_NUM = -5
    NEW_POINT_START = 5

    # interpolate function
    def interpolate_func(x, a, b):
        return a * x + b

    # color plate for line drawing
    COLOR_PLATE = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 255),
        (0, 165, 255),
        (128, 0, 255),
        (255, 192, 203),
        (42, 42, 165),
        (0, 128, 128),
        (128, 128, 0),
        (0, 215, 255),
        (75, 0, 130),
        (80, 127, 255),
    ]

    # filter line distance threshold
    AVERAGE_DISTANCE_THRESHOLD = 5

    # line draw pixel width
    PIXEL_LENGTH = 12

    # white threshold value
    WHITE_THRESHOLD = 240

    # line remove short amount
    SHORT_AMOUNT = 10

    # black count signal amplitude filter
    SIGNAL_AMPLITUDE = 3

    # detect points
    SLIDING_WINDOW = 10


@dataclass(frozen=True)
class ChartType:
    BOX_PLOT = "box"
    BAR_PLOT = "bar"
    KAPLAN_MEIER_CURVE = "kaplan-meier-curve"
    LINE_PLOT = "line"
    SCATTER_PLOT = "scatter"
    SPIDER_PLOT = "spider-plot"
    OTHER = "other"


@dataclass(frozen=True)
class AxisMinMax:
    example_url = (
        "https://delineate.nyc3.digitaloceanspaces.com/delineate-staging/"
        "plot-extraction/6712924c15938c4df2f6e0f2/6720c3ee282a64c83dd12599/"
        "resized_image_w6hdqvlpb36idhwrs3fqhu25.png"
    )

    example_out = {
        "points": [
            {"label": "ymin", "x": 103, "y": 312},
            {"label": "ymax", "x": 103, "y": 23},
            {"label": "xmin", "x": 103, "y": 313},
            {"label": "xmax", "x": 733, "y": 313},
        ]
    }
    manual_x_tick_detection_pad = 10
    plot_area_max_close_distance = 20
    Y_AXIS_KERNEL_SHAPE_PERCENT = 0.50
    Y_AXIS_TICK_KERNEL_START = 7
    X_AXIS_KERNEL_SHAPE_PERCENT = 0.50
    X_AXIS_TICK_KERNEL_START = 7
    Y_AXIS_WHITEOUT_SECTION = ["y_title", "xlabel", "ylabel"]
    X_AXIS_WHITEOUT_SECTION = ["x_title", "xlabel", "ylabel"]
    X_AXIS_AREA_WIDTH = 10
    GRAY_FILTER_LEVEL = 240


@dataclass(frozen=True)
class SubstitutionDigitizer:
    GRID_MAX_COLUMN = 3
    GRID_TEXT_SPACING = 10
    GRID_PADDING = 20
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    SUBSTITUTION_PAD = 20
    SUBSTITUTION_LEGENDS = [
        "ready",
        "that",
        "security",
        "remember",
        "institution",
        "participant",
        "simple",
        "team",
        "plant",
        "front",
        "person",
        "market",
        "position",
        "meeting",
        "network",
        "standard",
        "practice",
        "access",
        "argue",
        "senior",
        "edge",
        "long",
        "check",
        "even",
        "wrong",
        "pattern",
        "community",
        "recent",
        "guy",
        "everything",
        "thousand",
        "stay",
        "book",
        "reason",
        "cause",
    ]


@dataclass(frozen=True)
class ResolutionCheck:
    DIGITIZER_MAX_RESOLUTION = 800
    UPLOAD_PATH = "plot-digitizer/resolution_check"


@dataclass(frozen=True)
class CategoricalAxis:
    SCORE_THRESHOLD = 0.5
    X_AXIS_LABEL_PADDING = 3  # Pixel


SUPPORTED_PLOTS_FOR_DIGITIZATION = {
    ChartType.LINE_PLOT,
    ChartType.SCATTER_PLOT,
    ChartType.KAPLAN_MEIER_CURVE,
    ChartType.SPIDER_PLOT,
}
ERROR_BAR_SUPPORTED_PLOTS = {
    ChartType.LINE_PLOT,
}
SEPARATOR = "\n" + "-" * 100 + "\n"
MAX_RETRIES = 3

AUTOFILL_ERROR_MESSAGE = "AI Autofill failed. "
DIGITIZATION_ERROR_MESSAGE = "AI Digitization failed. "
SPIDER_PLOT_MIN_LINE_NUM = 20

LLM = AutoChatModel.from_model_name(plot_digitizer_settings.LLM_NAME, temperature=0.2)
rephrase_chain = prepare_question_rephrasing_chain(llm=LLM)

SPIDER_PLOT_DEF = """
    Instruction on how to detect spider plots:
    A spider plot is a variation
    of a line plot. It will meet either of these following 2
    conditions:
    1. At least one legend has multiple lines
    that have the same color/marker style.
    2. If there are no legends, there will be multiple lines
    originating from a common or closely spaced starting
    point.
    If that is true for the given plot, classify it as
    spider-plot.
"""

VISION_AGENT_TIME_OUT = 60

CATEGORICAL_SKIP_PLOTS = {ChartType.SPIDER_PLOT, ChartType.BAR_PLOT}
