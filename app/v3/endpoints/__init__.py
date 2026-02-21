from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"


class ExtractionType(Enum):
    GENERAL_EXTRACTION = "ai-general-extraction"
    COVARIATE_EXTRACTION = "ai-covariate-extraction"
    PLOT_AUTOFILL_EXTRACTION = "ai-plot-autofill-extraction"
    ITERATIVE_AUTOFILL_EXTRACTION = "ai-iterative-autofill-extraction"
    PAPER_LABELS_EXTRACTION = "ai-paper-labels-extraction"
    DOSING_EXTRACTION = "ai-dosing-extraction"
