"""
Quality control checks for merged CSV output.

Runs registered checks per key (from qc_list). Each key can have multiple
checks; each failing check produces one QCError with error_source="qc".
"""

from io import StringIO

import httpx
import pandas as pd

from app.configs import settings
from app.v3.endpoints.merging.checks import get_checks_for_key
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.schemas import QCError


def get_qc_csv_from_backend(file_id: str, token: str | None = None) -> str:
    """Fetch CSV content for quality control from the main backend."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{settings.BACKEND_BASE_URL}/v3/quality-control/{file_id}"
    with httpx.Client(timeout=60) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["csv"]


def run_quality_checks(csv: str, qc_list: list[str]) -> list[QCError]:
    """
    Run quality checks on CSV content. Parses CSV to a DataFrame, runs each
    key's registered checks, and returns one QCError per failing check.
    """
    if not csv.strip():
        return []
    df = pd.read_csv(StringIO(csv))
    errors: list[QCError] = []
    for qc_name in qc_list:
        checks = get_checks_for_key(qc_name)
        logger.info(f"Checks for key {qc_name}: {checks}")
        for check_fn in checks:
            err = check_fn(df, qc_name)
            if err is not None:
                errors.append(err)
    return errors
