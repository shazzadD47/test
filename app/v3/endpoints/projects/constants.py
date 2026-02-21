from dataclasses import dataclass


@dataclass(frozen=True)
class AutoConnection:
    FOOTNOTE_FILTER_WORDS = ["footnote", "footnotes"]
    CAPTION_FILTER_WORDS = ["caption", "captions"]
