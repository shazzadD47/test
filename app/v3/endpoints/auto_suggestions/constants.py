from dataclasses import dataclass


@dataclass(frozen=True)
class AutoConnectionSkipWords:
    FOOTNOTE_FILTER_WORDS = ["footnote", "footnotes"]
    CAPTION_FILTER_WORDS = ["caption", "captions"]


@dataclass(frozen=True)
class AISelectionInputTypes:
    CHART = "chart"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    LEGEND = "legend"


@dataclass(frozen=True)
class DatabaseFigureTypes:
    PLOT = "plot"
    TABLE = "table"
    EQUATION = "equation"
    LEGEND = "legend"
    IMAGE = [PLOT, TABLE]
