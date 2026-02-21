from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from app.v3.endpoints.agent_chat.schema import AgentChatRequest


def is_valid_request(request: dict | str) -> bool:
    if not isinstance(request, dict):
        return False

    try:
        AgentChatRequest(**request)
    except ValidationError:
        return False

    return True


def convert_to_title_case(text: str) -> str:
    return text.replace("_", " ").title()


def _get_excel_sheet(
    file_path: str, sheet_name: str | None = None, n_rows: int | None = None
) -> pd.DataFrame:
    """
    Reads a specific sheet from an Excel file into a pandas DataFrame.
    Defaults to the first sheet if no sheet_name is provided.
    If n_rows is provided, limits to that many rows.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=n_rows)

        if isinstance(df, dict):
            df = next(iter(df.values()))
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read sheet '{sheet_name or 'default'}' from {file_path}: {e}"
        )


def excel_to_csv(file_path: str, n_rows: int = 30) -> list[str]:
    """
    Converts up to `n_rows` (default: 30) random rows from
    each sheet of an Excel file into CSVs.
    Keeps all columns.
    Uses _get_excel_sheet() for reading sheets.
    """
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    file_path = Path(file_path)
    result_file_paths = []

    if len(sheets) <= 1:
        csv_file_path = file_path.with_name(f"{file_path.stem}_preview.csv")
        df = _get_excel_sheet(str(file_path))

        # Randomly sample up to n_rows rows
        if len(df) > n_rows:
            df = df.sample(n=n_rows, random_state=42)

        df.to_csv(csv_file_path, index=False)
        result_file_paths.append(csv_file_path)

    else:
        file_name = file_path.stem
        parent_dir = file_path.parent

        sheets_dir = parent_dir / f"{file_name}_preview"
        sheets_dir.mkdir(parents=True, exist_ok=True)

        for sheet_name in sheets:
            csv_file_path = sheets_dir / f"{sheet_name}_preview.csv"

            sheet_df = _get_excel_sheet(str(file_path), sheet_name=sheet_name)

            # Randomly sample up to n_rows rows
            if len(sheet_df) > n_rows:
                sheet_df = sheet_df.sample(n=n_rows, random_state=42)

            sheet_df.to_csv(csv_file_path, index=False)
            result_file_paths.append(csv_file_path)

    return [str(fp) for fp in result_file_paths]
