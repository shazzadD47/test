import os
import traceback
from itertools import product

from app.constants import NUMERICAL_D_TYPES
from app.utils.download import download_files_from_flag_id
from app.utils.utils import check_if_null
from app.v3.endpoints.general_extraction.logging import celery_logger as logger


def check_if_unit_label(label: str) -> bool:
    return label.lower().strip().endswith("_unit")


def check_if_numerical_label(label: dict) -> bool:
    return label["d_type"].lower().strip() in NUMERICAL_D_TYPES


def get_unit_label_prefix(unit_label_name: str) -> str:
    """
    Extract the prefix from a unit label by removing the _unit suffix
    (case-insensitive), preserving the original case of the prefix.

    e.g., "weight_UNIT" -> "weight", "Weight_unit" -> "Weight"
    """
    return unit_label_name[:-5]


def find_matching_unit_labels(
    label_name: str,
    all_label_names: list[str],
) -> list[str]:
    """
    Find unit label names in all_label_names that correspond to the given
    label name. Returns all matches — both exact case matches (unit label
    prefix == label_name) and case-insensitive matches — with exact matches
    listed first.

    For example, if label_name is "ARM_NUMBER" and all_label_names contains
    ["ARM_NUMBER_UNIT", "ARM_NUMBER_unit", "arm_number_unit"], all three are
    returned. The downstream `is_best_numerical_match_for_unit_label` guard
    prevents cross-contamination when multiple numerical labels compete for
    unit labels.
    """
    exact_matches = []
    case_insensitive_matches = []
    label_name_lower = label_name.lower().strip()

    for name in all_label_names:
        if not check_if_unit_label(name):
            continue
        prefix = get_unit_label_prefix(name)
        if prefix == label_name:
            exact_matches.append(name)
        elif prefix.lower().strip() == label_name_lower:
            case_insensitive_matches.append(name)

    return exact_matches + case_insensitive_matches


def reassign_unit_label_roots(
    table_structure: list[dict],
) -> list[dict]:
    """
    If a unit label is assigned as root and has a numerical counterpart
    in the table structure, transfer the root designation to the numerical
    counterpart instead. Unit labels derive their answers from the numerical
    counterpart's 'unit' field, so the numerical label should be the root.

    Also demotes ALL matching unit labels for that counterpart (not just
    the one that triggered the check), using find_matching_unit_labels
    with exact-case-first preference.
    """
    table_structure_name_hash = {label["name"]: label for label in table_structure}
    all_label_names = list(table_structure_name_hash.keys())

    for label in table_structure:
        if label["c_type"] == "root" and check_if_unit_label(label["name"]):
            prefix = get_unit_label_prefix(label["name"])
            prefix_lower = prefix.lower().strip()

            # Find the numerical counterpart — exact case match first
            numerical_counterpart = None
            for other_label in table_structure:
                if other_label["name"] == prefix and check_if_numerical_label(
                    other_label
                ):
                    numerical_counterpart = other_label
                    break

            # Case-insensitive fallback
            if numerical_counterpart is None:
                for other_label in table_structure:
                    if other_label[
                        "name"
                    ].lower().strip() == prefix_lower and check_if_numerical_label(
                        other_label
                    ):
                        numerical_counterpart = other_label
                        break

            if numerical_counterpart is not None:
                # Transfer root to the numerical counterpart
                numerical_counterpart["c_type"] = "root"

                # Demote ALL matching unit labels for this counterpart
                # using the same exact-case-first matching logic
                matching_unit_labels = find_matching_unit_labels(
                    numerical_counterpart["name"], all_label_names
                )
                for unit_name in matching_unit_labels:
                    unit_label_data = table_structure_name_hash[unit_name]
                    if unit_label_data["c_type"] == "root":
                        unit_label_data["c_type"] = "general"
    return table_structure


def is_best_numerical_match_for_unit_label(
    numerical_label_name: str,
    unit_label_name: str,
    table_structure: list[dict],
) -> bool:
    """
    Check if the given numerical label is the best (exact-case-first) match
    for a unit label.

    A numerical label is the best match if:
    - It is an exact case match (prefix of unit label == numerical label name), OR
    - It is a case-insensitive match AND no exact-case numerical label exists
      in the table structure for that unit label.

    This prevents a case-insensitive fallback numerical label (e.g. "Weight")
    from writing to a unit label (e.g. "weight_unit") when the exact-case
    numerical label ("weight") already exists.
    """
    prefix = get_unit_label_prefix(unit_label_name)

    # Exact match — always the best
    if prefix == numerical_label_name:
        return True

    # Case-insensitive match — only best if no exact-match numerical exists
    return all(
        not (label["name"] == prefix and check_if_numerical_label(label))
        for label in table_structure
    )


def check_if_unit_label_has_numerical_non_unit_label(
    unit_label: str,
    table_structure: list[dict],
) -> bool:
    """
    Check if a unit label has a corresponding non-unit label that is numerical.
    Unit labels should only be skipped from direct LLM extraction when their
    corresponding non-unit label is numerical, because the unit answer is derived
    from the NumericalAnswer schema's 'unit' field. If the corresponding label
    is not numerical, the unit label must be extracted independently.

    Prefers exact case match (prefix matches label name exactly) over
    case-insensitive match.
    """
    prefix = get_unit_label_prefix(unit_label)
    prefix_lower = prefix.lower().strip()

    # Exact case match first
    for label in table_structure:
        if label["name"] == prefix:
            return check_if_numerical_label(label)

    # Case-insensitive fallback
    for label in table_structure:
        if label["name"].lower().strip() == prefix_lower:
            return check_if_numerical_label(label)
    return False


def create_custom_instructions_prompt(
    custom_instruction: str = None,
) -> str:
    if not check_if_null(custom_instruction):
        return f"""
        ** Special Instructions from User:**
        These take precedence over all other instructions.
        You MUST follow these instructions.
        <special_user_instructions>
        - {custom_instruction}
        </special_user_instructions>
        """
    return ""


def process_pdf_file(
    flag_id: str,
) -> dict:
    file_details = {}
    try:
        files_result = download_files_from_flag_id(flag_id, return_supplementaries=True)
        # Handle the dict result from download_files_from_flag_id
        if isinstance(files_result, dict):
            main_file = files_result.get("main_file")
            if main_file and os.path.exists(main_file):
                file_details["pdf_path"] = main_file
            else:
                file_details["pdf_path"] = "N/A"

            # Handle supplementary files
            supplementaries = files_result.get("supplementaries", [])
            valid_supplementaries = [
                path for path in supplementaries if path and os.path.exists(path)
            ]
            file_details["supplementary_paths"] = valid_supplementaries
        else:
            # Fallback for backwards compatibility if a string path is returned
            if files_result and os.path.exists(files_result):
                file_details["pdf_path"] = files_result
            else:
                file_details["pdf_path"] = "N/A"
            file_details["supplementary_paths"] = []
    except Exception as e:
        logger.info(f"Error getting file from flag id: {e}")
        logger.info(f"Traceback: {traceback.format_exc()}")
        file_details["pdf_path"] = "N/A"
        file_details["supplementary_paths"] = []
    return file_details


def unroll_root_answers(root_labels: list[str], root_label_answers: list[dict]) -> dict:
    """
    Unroll root answers
    Args:
        root_labels: List[str]: Root labels
        root_label_answers: list[dict]: Root label answers
    Returns:
        dict: Unrolled root answers
    """
    if len(root_labels) == 0 or len(root_label_answers) == 0:
        return {}

    root_label_answers_unrolled = [
        tuple(row[label] for label in root_labels) for row in root_label_answers
    ]
    root_label_answers_unrolled = reverse_product(root_label_answers_unrolled)
    root_label_answers_unrolled = dict(zip(root_labels, root_label_answers_unrolled))
    return root_label_answers_unrolled


def roll_root_answers(root_answers: dict) -> list[dict]:
    """
    Roll root answers
    Args:
        root_answers: dict: Root answers
    Returns:
        list[dict]: Rolled root answers
    """
    labels = list(root_answers.keys())
    answers = list(root_answers.values())
    rolled_answers = product(*answers)
    rolled_answers = [dict(zip(labels, answer)) for answer in rolled_answers]
    return rolled_answers


def reverse_product(product_list: list[tuple]) -> tuple[list, ...]:
    """
    Reverses the operation of itertools.product using a robust iterative approach.

    Given a list of tuples representing the Cartesian product of several
    iterables, this function reconstructs and returns the original iterables.
    This version correctly handles edge cases with repeating elements or patterns
    by analyzing the structure of the product's prefixes.

    Args:
        product_list: The list of tuples generated by itertools.product.

    Returns:
        A tuple containing the lists that were the original inputs to the
        product function. Returns an empty tuple if the input is empty.
    """
    # If the input list is empty, return an empty tuple of iterables.
    if not product_list:
        return ()  # type: ignore

    reconstructed_iterables = []
    current_product = product_list

    # The number of original iterables is the length of any tuple in the product.
    num_iterables = len(product_list[0]) if product_list else 0

    # Iteratively find each of the original iterables, from last to first.
    for _ in range(num_iterables):
        if not current_product:
            break

        # --- Find the last of the remaining original iterables ---
        # The key insight is that the last iterable cycles completely before any
        # element in the preceding iterables changes. We find its length by
        # finding the first tuple where the "prefix" (all elements except the
        # last) is different from the first tuple's prefix.

        # 1. Determine the length of the next iterable.
        if len(current_product[0]) == 1:
            # If we're down to the last (original first) iterable,
            # its length is the whole list.
            len_next_iterable = len(current_product)
        else:
            first_prefix = current_product[0][:-1]
            len_next_iterable = len(current_product)  # Default if prefix never changes
            for i, t in enumerate(current_product):
                if i > 0 and t[:-1] != first_prefix:
                    len_next_iterable = i
                    break

        # 2. Extract the iterable using the determined length.
        next_iterable = [t[-1] for t in current_product[:len_next_iterable]]

        # if only one unique value, then make it a single element list
        if len(set(next_iterable)) == 1:
            next_iterable = [next_iterable[0]]
            len_next_iterable = 1

        # 3. Add the found iterable to the front of our results list.
        reconstructed_iterables.insert(0, next_iterable)

        # --- Prepare for the next iteration ---
        # If this was the last iterable we needed to find, we're done.
        if len(reconstructed_iterables) == num_iterables:
            break

        # The next (smaller) product is the list of unique prefixes. We get this
        # by stepping through the current product list with a step size equal
        # to the length of the iterable we just found.
        current_product = [t[:-1] for t in current_product[::len_next_iterable]]

    return tuple(reconstructed_iterables)
