LANGSMITH_TAGS = ["general-extraction"]

BASE_QUESTION = (
    "Generally arms refers to the treatments in QSP."
    " what are the arms in these experiments?"
    " List those arms. Be short and concise."
)

# error messages
ROOT_CHOICES_NOT_ALLOWED = "Root choices are not allowed when is_root is true."
NO_ROOT_CHOICES_PROVIDED = "`is_root` is false but no root choice is provided"


paper_dependent_fields = {
    "au",
    "ti",
    "jr",
    "py",
    "vl",
    "is",
    "pg",
    "pubmedid",
    "la",
    "regid",
    "regnm",
    "tp",
    "ts",
    "doi",
    "doi_url",
    "doi url",
    "cit_url",
    "cit url",
}
