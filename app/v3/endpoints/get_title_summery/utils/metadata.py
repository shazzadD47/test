from typing import Any

from app.core.database.models import FileDetails


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metadata_to_str(
    metadata: dict[str, Any],
    ignore_keys: list[str] | None = None,
    keep_only_keys: list[str] | None = None,
) -> list[str]:
    """
    Process metadata and return a list of formatted strings suitable for RAG
    applications.

    Args:
        metadata (Dict[str, Any]): The metadata JSON object.
        ignore_keys (Optional[List[str]]): Keys to ignore in the metadata.
        keep_only_keys (Optional[List[str]]): Keys to keep in the metadata.

    Returns:
        List[str]: A list of formatted strings containing the metadata information.

    Raises:
        ValueError: If both ignore_keys and keep_only_keys are provided.
    """
    if ignore_keys and keep_only_keys:
        raise ValueError("Cannot specify both ignore_keys and keep_only_keys")

    result = []

    def is_empty(value: Any) -> bool:
        """Check if a value is empty (None, empty string, or only whitespace)."""
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and not value:
            return True
        return False

    def to_string(value: Any) -> str:
        """Recursively convert complex structures to string."""
        if isinstance(value, dict):
            return ", ".join(
                f"{k}: {to_string(v)}" for k, v in value.items() if not is_empty(v)
            )
        elif isinstance(value, list):
            return ", ".join(map(to_string, filter(lambda x: not is_empty(x), value)))
        else:
            return str(value)

    def pluralize(word: str) -> str:
        """Simple pluralization function."""
        if word.endswith("y"):
            return word[:-1] + "ies"
        elif word.endswith("s"):
            return word + "es"
        else:
            return word + "s"

    def format_key(key: str, is_plural: bool) -> str:
        """Format the key, pluralizing if necessary."""
        formatted_key = key.replace("_", " ").capitalize()
        return pluralize(formatted_key) if is_plural else formatted_key

    def process_key(key: str, value: Any) -> None:
        if is_empty(value):
            return

        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            formatted_key = format_key(key, len(value) > 1)
            chunk = [f"The {formatted_key} of the paper are:"]
            for item in value:
                item_str = to_string(item)
                chunk.append(f"- {item_str}")
            result.append("\n".join(chunk))
        elif isinstance(value, list):
            formatted_key = format_key(key, len(value) > 1)
            result.append(
                f"{formatted_key} "
                f"{'are' if len(value) > 1 else 'is'}: "
                f"{to_string(value)}"
            )
        elif isinstance(value, dict):
            nested_chunk = []
            for sub_key, sub_value in value.items():
                if not is_empty(sub_value):
                    nested_chunk.append(f"{sub_key}: {to_string(sub_value)}")
            if nested_chunk:
                result.append(
                    f"The {format_key(key, False)} of the paper is: "
                    + ", ".join(nested_chunk)
                )
        else:
            formatted_key = format_key(key, False)
            result.append(f"The {formatted_key} of the paper is: {to_string(value)}")

    for key, value in metadata.items():
        if ignore_keys and key in ignore_keys:
            continue
        if keep_only_keys and key not in keep_only_keys:
            continue
        process_key(key, value)

    return result


def prepare_file_details(
    flag_id: str,
    project_id: str,
    user_id: str,
    file_extension: str,
    metadata: dict[str, Any],
    title: str | None = None,
    doi: str | None = None,
    doi_url: str | None = None,
    summary: str | None = None,
    supplementary_id: str | None = None,
) -> FileDetails:
    return FileDetails(
        flag_id=flag_id,
        supplementary_id=supplementary_id,
        project_id=project_id,
        user_id=user_id,
        file_extension=file_extension,
        title=title,
        doi=doi,
        doi_url=doi_url,
        publication_type=metadata.get("type"),
        summary=summary if (summary and summary.strip()) else None,
        authors=metadata.get("author") or [],
        funder=metadata.get("funder") or [],
        publisher=metadata.get("publisher"),
        published_date=metadata.get("published") or {},
        issue=metadata.get("issue"),
        issue_date=metadata.get("issued") or {},
        volume=metadata.get("volume"),
        page=metadata.get("page"),
        license=metadata.get("license") or [],
        container_title=metadata.get("container-title"),
        ISSN=metadata.get("ISSN") or [],
        reference_count=safe_int(metadata.get("references-count"))
        or safe_int(metadata.get("reference-count")),
        references=metadata.get("reference") or [],
        referenced_by_count=safe_int(metadata.get("is-referenced-by-count")),
        raw_metadata=metadata,
    )
