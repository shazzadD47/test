from pathlib import Path

import nbformat
from nbconvert import PDFExporter

from app.v3.endpoints.get_title_summery.logging import logger


def _read_notebook_file(file_path: str | Path) -> str | Path:
    """Read and convert notebook files to bytes."""
    logger.debug(f"Reading {file_path}...")

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    if not file_path.is_file():
        raise IsADirectoryError(f"File {file_path} is a directory")

    with open(file_path, encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    notebook_content = _truncate_cell_outputs(notebook_content)

    pdf_exporter = PDFExporter()
    pdf_data, _ = pdf_exporter.from_notebook_node(notebook_content)
    file_path_pdf = file_path.with_suffix(".pdf")
    with open(file_path_pdf, "wb") as f:
        f.write(pdf_data)

    return file_path_pdf


def _truncate_cell_outputs(
    notebook_content: nbformat.NotebookNode,
) -> nbformat.NotebookNode:
    """
    Truncate cell outputs to maximum 25 lines
    (first 10 + middle 10 + last 5).
    """
    for cell in notebook_content.cells:
        if cell["cell_type"] == "code" and "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "stream" and output.name == "stdout":
                    # calculate the number of lines in the output,
                    # truncate to 1st 10 lines, middle 10 lines at fixed intervals,
                    # and last 5 lines
                    lines = output.text.split("\n")
                    total_lines = len(lines)
                    if total_lines > 25:
                        first_10_lines = "\n".join(lines[:10])
                        first_10_lines_text = f"First 10 lines: {first_10_lines}"
                        last_5_lines = "\n".join(lines[-5:])
                        last_5_lines_text = f"Last 5 lines: {last_5_lines}"
                        # Get 10 lines from the middle section at fixed intervals
                        middle_section = lines[10:-5]
                        if len(middle_section) <= 10:
                            middle_10_lines = middle_section
                        else:
                            # Select 10 lines at evenly spaced intervals
                            step = len(middle_section) / 10
                            middle_indices = [int(i * step) for i in range(10)]
                            middle_10_lines = [
                                middle_section[i] for i in middle_indices
                            ]
                        middle_10_lines_text = "\n".join(middle_10_lines)
                        middle_10_lines_text = f"Middle lines: {middle_10_lines_text}"
                        output.text = "\n".join(
                            [
                                first_10_lines_text,
                                middle_10_lines_text,
                                last_5_lines_text,
                            ]
                        )
    return notebook_content
