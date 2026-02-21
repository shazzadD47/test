"""
V1 merge pipeline: transform tables, build observation + dose event data,
merge covariate, then paper_labels; backfill study/arm columns, rename and output.

Five table types only: plot, dosing, covariate, observation_table, paper_labels.
All paper_labels are merged via _merge_paper_labels() (no null type).

Pipeline steps (before -> after):

  1. Transform: raw tables -> clean names, group, aliases (file_name/doi_url).
  2. Build obs/dose/cov: tables_by_type -> obs_all (plot + observation_table),
     dose, cov, paper_labels; obs_all order: observation_table rows then plot rows.
  3. Arm duration & backfill: dose gains arm_dur; missing group in dose filled from obs.
  4. Event data: obs_all + dose -> one DataFrame with event_type OBS/DOSE.
  5. Merge covariate (on group).
  6. Backfill: FROM_OBS_COLS from OBS into DOSE by group; arm-level ffill/bfill.
  7. Uppercase columns, EVID/MDV, fill DOSE from OBS by group, post-fill by GROUP.
  8. Paper labels: merge on FILE_NAME via _merge_paper_labels(); fill study columns.
  9. Rename to final schema, coalesce duplicates, FINAL_COLUMNS order.

All merges are left joins; source tables are not modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.v3.endpoints.merging.constants import (
    ARM_LEVEL_COLS,
    FINAL_COLUMNS,
    FROM_OBS_COLS,
    POST_FILL_COLS,
    RENAME_MAP,
)
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.merge_transform import (
    clean_names,
    drop_unnamed,
    ensure_group_from_arm_trt,
    na_strings_to_nan,
    normalize_aliases,
    std_group,
)
from app.v3.endpoints.merging.schemas import SingleError, TablesByType


def _transform_tables_by_type(tables_by_type: TablesByType) -> None:
    """
    Apply clean_names, drop_unnamed, normalize_aliases, ensure_group_from_arm_trt,
    std_group to every table. Modifies tables_by_type in place.
    """

    def _transform_single_table(df: pd.DataFrame) -> pd.DataFrame:
        df = clean_names(df)
        df = drop_unnamed(df)
        df = normalize_aliases(df)
        df = ensure_group_from_arm_trt(df)
        df = std_group(df)
        return df

    for key in ("plot", "dosing", "covariate", "observation_table", "paper_labels"):
        for index, df in enumerate(tables_by_type.get(key) or []):
            tables_by_type[key][index] = _transform_single_table(df.copy())


def _build_obs_dose_cov(
    tables_by_type: TablesByType,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    bool,
]:
    """
    Build obs_all = concat(observation_table, plot), dose, cov, paper_labels.

    Before: tables_by_type with plot, observation_table, dosing, covariate,
    paper_labels. After: obs_all (observation_table rows first, then plot rows,
    each with _obs_origin 0/1 for sorting), dose, cov, paper_labels; use_two_file_flow
    is True when observation_table is present.
    """
    plot_list = tables_by_type.get("plot") or []
    observation_table_list = tables_by_type.get("observation_table") or []
    dosing_list = tables_by_type.get("dosing") or []
    cov_list = tables_by_type.get("covariate") or []
    paper_labels_list = tables_by_type.get("paper_labels") or []

    obs = pd.concat(plot_list, ignore_index=True, sort=False) if plot_list else None
    obs_table = (
        pd.concat(observation_table_list, ignore_index=True, sort=False)
        if observation_table_list
        else None
    )
    if obs is not None:
        obs = obs.copy()
        obs["_obs_origin"] = 1
    if obs_table is not None:
        obs_table = obs_table.copy()
        obs_table["_obs_origin"] = 0
    obs_parts = [x for x in (obs, obs_table) if x is not None]
    obs_all = pd.concat(obs_parts, ignore_index=True, sort=False) if obs_parts else None
    dose = (
        pd.concat(dosing_list, ignore_index=True, sort=False) if dosing_list else None
    )
    cov = pd.concat(cov_list, ignore_index=True, sort=False) if cov_list else None
    paper_labels = (
        pd.concat(paper_labels_list, ignore_index=True, sort=False)
        if len(paper_labels_list) > 1
        else (paper_labels_list[0] if paper_labels_list else None)
    )
    use_two_file_flow = obs_table is not None
    return obs_all, dose, cov, paper_labels, use_two_file_flow


def _compute_arm_dur(dose: pd.DataFrame) -> pd.DataFrame:
    """Compute arm_dur and arm_dur_unit per group.

    How values are computed:
    - Per group, rows in arm_time order. For each dosing row with amt > 0:
      - If ii or addl is set: contribution to arm_dur is ii * (addl + 1).
      - If ii == 0 and addl == 0: contribution is time until next dose in
        group (next arm_time - current arm_time), or 0 for the last row.
    - Placebo arms (total amt == 0): one row per group; arm_dur = ii*(addl+1).
    - arm_dur_unit is set from arm_time_unit (e.g. "weeks", "day").

    Raises ValueError if a placebo group has more than one row.
    """
    dose = dose.copy()
    dose["arm_dur"] = np.nan
    dose["arm_dur_unit"] = np.nan
    dose["amt"] = pd.to_numeric(dose.get("amt"), errors="coerce")
    dose["ii"] = pd.to_numeric(dose.get("ii"), errors="coerce")
    dose["addl"] = (
        pd.to_numeric(dose.get("addl"), errors="coerce").fillna(0).astype(int)
    )
    dose["arm_time"] = pd.to_numeric(dose.get("arm_time"), errors="coerce")

    for grp, g in dose.groupby("group", sort=False):
        arm_dur = 0.0
        if g["amt"].fillna(0).sum() == 0 and len(g) > 1:
            raise ValueError("More than 1 Placebo Dosing Row")
        if g["amt"].fillna(0).sum() == 0:
            r = g.iloc[0]
            ii = 0.0 if pd.isna(r["ii"]) else float(r["ii"])
            arm_dur = ii * (int(r["addl"]) + 1)
        else:
            g_sorted = g.sort_values("arm_time", kind="mergesort")
            for pos, (_, r) in enumerate(g_sorted.iterrows()):
                if pd.isna(r["amt"]) or r["amt"] == 0:
                    continue
                ii = 0.0 if pd.isna(r["ii"]) else float(r["ii"])
                addl = int(r["addl"]) if pd.notna(r["addl"]) else 0
                if ii == 0 and addl == 0:
                    if pos < len(g_sorted) - 1:
                        cur_t = r["arm_time"]
                        nxt_t = g_sorted.iloc[pos + 1]["arm_time"]
                        val = (
                            float(nxt_t - cur_t)
                            if (pd.notna(cur_t) and pd.notna(nxt_t) and nxt_t > cur_t)
                            else 0.0
                        )
                    else:
                        val = 0.0
                else:
                    val = ii * (addl + 1)
                arm_dur += val
        dose.loc[dose["group"] == grp, "arm_dur"] = arm_dur
    dose["arm_dur_unit"] = dose.get("arm_time_unit")
    return dose


def _dose_group_backfill_from_obs(
    dose: pd.DataFrame, obs: pd.DataFrame
) -> pd.DataFrame:
    """Fill missing group in dose from obs by file_name/doi_url/doi.

    Left join semantics. Before: dose group=NaN, obs has group. After: dose
    group filled from obs match on file_name. Merge keys: file_name, doi_url,
    or doi.
    """
    if "group" not in dose.columns or dose["group"].notna().all():
        return dose
    missing_g = dose["group"].isna()
    if not missing_g.any():
        return dose
    keys = []
    if "file_name" in obs.columns and "file_name" in dose.columns:
        keys.append("file_name")
    if "doi_url" in obs.columns and "doi_url" in dose.columns:
        keys.append("doi_url")
    if "doi" in obs.columns and "doi" in dose.columns:
        keys.append("doi")
    if not keys and "filename" in obs.columns and "filename" in dose.columns:
        keys.append("filename")
    if not keys:
        return dose
    obs_map = obs.dropna(subset=keys + ["group"])
    if obs_map.empty:
        return dose
    obs_map = obs_map.groupby(keys, dropna=False)["group"].first().reset_index()
    dose = dose.merge(obs_map, on=keys, how="left", suffixes=("", "_from_obs"))
    if "group_from_obs" in dose.columns:
        dose.loc[missing_g, "group"] = dose.loc[missing_g, "group_from_obs"]
        dose = dose.drop(columns=["group_from_obs"], errors="ignore")
    return dose


def _ci_variance_aliases(obs: pd.DataFrame) -> pd.DataFrame:
    """Add dv_var_upper/dv_var_lower from uci/varu/lci if missing."""
    if "dv_var_upper" not in obs.columns and "uci" in obs.columns:
        obs["dv_var_upper"] = obs["uci"]
    elif "dv_var_upper" not in obs.columns and "varu" in obs.columns:
        obs["dv_var_upper"] = obs["varu"]
    if "dv_var_lower" not in obs.columns and "lci" in obs.columns:
        obs["dv_var_lower"] = obs["lci"]
    return obs


def _event_data_and_group_name(
    obs: pd.DataFrame | None,
    dose: pd.DataFrame | None,
    use_two_file_flow: bool = True,
) -> pd.DataFrame:
    """Build event_data = concat(obs, dose) with event_type; set group_name default.

    Matches merge_script: only set group_name = group when use_two_file_flow
    (merge1 / observation-table present); else rely on group_name_dose.

    Before:
        obs: rows with group, (no event_type)   |  dose: rows with group,
        group_name (or group_name_dose)
    After:
        event_data: all rows with event_type="OBS" or "DOSE"; one group_name
        column (from group or coalesced: group_name where notna, else
        group_name_dose). Empty DataFrame if both inputs empty.
    """
    parts = []
    if obs is not None and not obs.empty:
        o = obs.copy()
        o["event_type"] = "OBS"
        if "_obs_origin" not in o.columns:
            o["_obs_origin"] = 1
        parts.append(o)
    if dose is not None and not dose.empty:
        d = dose.copy()
        if "group_name" in d.columns and "group_name_dose" not in d.columns:
            d = d.rename(columns={"group_name": "group_name_dose"})
        d["event_type"] = "DOSE"
        d["_obs_origin"] = 2
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    event_data = pd.concat(parts, ignore_index=True, sort=False)
    if (
        use_two_file_flow
        and "group_name" not in event_data.columns
        and "group" in event_data.columns
    ):
        event_data["group_name"] = event_data["group"]
    if "group_name" in event_data.columns and "group_name_dose" in event_data.columns:
        event_data["group_name"] = event_data["group_name"].where(
            event_data["group_name"].notna(), event_data["group_name_dose"]
        )
    elif "group_name_dose" in event_data.columns:
        event_data = event_data.rename(columns={"group_name_dose": "group_name"})
    return event_data


def _merge_cov_and_restore_dv(
    final_data: pd.DataFrame, cov: pd.DataFrame
) -> pd.DataFrame:
    """Left merge cov on group; restore dv_var_upper/dv_var_lower from _obs."""
    final_data = final_data.merge(cov, on="group", how="left", suffixes=("", "_cov"))
    for c in ["dv_var_upper", "dv_var_lower"]:
        if f"{c}_obs" in final_data.columns and c not in final_data.columns:
            final_data[c] = final_data[f"{c}_obs"]
    return final_data


def _merge_null_table(
    result: pd.DataFrame,
    table_name: str,
    null_df: pd.DataFrame,
    errors: list[SingleError],
) -> pd.DataFrame:
    """
    Left-merge null table into result: on group, or file_name (and stu_number),
    or append as extra rows and log warning. Uses lowercase column names for
    key detection.
    """
    result_cols = {str(c).strip().lower() for c in result.columns}
    null_cols = {str(c).strip().lower() for c in null_df.columns}

    if "group" in result_cols and "group" in null_cols:
        result = result.merge(
            null_df,
            on="group",
            how="left",
            suffixes=("", f"_from_{table_name[:20]}"),
        )
        return result

    if "file_name" in result_cols and (
        "file_name" in null_cols or "filename" in null_cols
    ):
        if "filename" in null_cols and "file_name" not in null_cols:
            fn_col = next(
                c for c in null_df.columns if str(c).strip().lower() == "filename"
            )
            null_df = null_df.rename(columns={fn_col: "file_name"})
        merge_keys = ["file_name"]
        if "stu_number" in result_cols and "stu_number" in null_cols:
            merge_keys.append("stu_number")
        result = result.merge(
            null_df,
            on=merge_keys,
            how="left",
            suffixes=("", f"_from_{table_name[:15]}"),
        )
        return result

    if "filename" in result_cols and (
        "file_name" in null_cols or "filename" in null_cols
    ):
        fn_result = next(
            c for c in result.columns if str(c).strip().lower() == "filename"
        )
        fn_null = next(
            c
            for c in null_df.columns
            if str(c).strip().lower() in ("file_name", "filename")
        )
        if fn_result != fn_null:
            null_df = null_df.rename(columns={fn_null: fn_result})
        result = result.merge(
            null_df,
            on=[fn_result],
            how="left",
            suffixes=("", f"_from_{table_name[:15]}"),
        )
        return result

    errors.append(
        {
            "error_name": "Table appended (no common key)",
            "error_message": (
                f"Table '{table_name}' had no common key (group or "
                "FILE_NAME/STU_NUMBER) with the merged result; appended as "
                "extra rows to preserve data."
            ),
        }
    )
    for c in result.columns:
        if c not in null_df.columns:
            null_df[c] = np.nan
    for c in null_df.columns:
        if c not in result.columns:
            result[c] = np.nan
    return pd.concat([result, null_df], ignore_index=True, sort=False)


def _fill_dose_from_obs(final_data: pd.DataFrame) -> pd.DataFrame:
    """Fill DOSE rows from OBS within same group for FROM_OBS_COLS."""
    if "event_type" not in final_data.columns or "group" not in final_data.columns:
        return final_data
    obs_mask = final_data["event_type"].eq("OBS")
    dose_mask = final_data["event_type"].eq("DOSE")
    for col in FROM_OBS_COLS:
        if col not in final_data.columns:
            continue
        src = final_data.loc[obs_mask, ["group", col]].dropna(subset=["group"])
        if src.empty:
            continue
        m = src.dropna(subset=[col]).groupby("group")[col].first()
        if m.empty:
            continue
        final_data.loc[dose_mask, col] = final_data.loc[dose_mask, col].where(
            final_data.loc[dose_mask, col].notna(),
            final_data.loc[dose_mask, "group"].map(m),
        )
    return final_data


def _backfill_arm_level(final_data: pd.DataFrame) -> pd.DataFrame:
    """Back-fill arm-level columns by group (ffill/bfill)."""
    if "group" not in final_data.columns:
        return final_data
    for col in ARM_LEVEL_COLS:
        if col not in final_data.columns:
            continue
        final_data[col] = final_data.groupby("group")[col].transform(
            lambda x: x.where(x.notna(), x.ffill().bfill())
        )
    for col in ["arm_dur", "arm_dur_unit"]:
        if col in final_data.columns:
            final_data[col] = final_data.groupby("group")[col].transform("first")
    return final_data


# Study-level columns filled from paper labels (script: merge1 paper merge)
_PAPER_FILL_COLS = [
    "STU_NUMBER",
    "TRIAL_NAME",
    "AU",
    "TITLE",
    "JOURNAL_NAME",
    "YEAR",
    "VL",
    "IS",
    "PG",
    "LA",
    "REGID",
    "REGNM",
    "TP",
    "TS",
    "DOI_URL",
    "CIT_URL",
    "STD_IND",
    "STD_TRT",
    "STD_TRT_CLASS",
    "PUBMED_ID",
    "TRIAL_ID",
    "COUNTRY",
    "SPONSOR_TYPE",
    "BLINDING",
    "NEW_ONSET_T1DM",
    "ENDPOINT_DURATION_MONTHS",
    "SOC_DESCRIPTION",
    "STUDY_DESIGN",
    "NOTES",
]

# Map common paper_labels column names to final schema (typo → canonical).
_PAPER_COLUMN_ALIASES: dict[str, str] = {
    "BLINIDING": "BLINDING",
}


def _merge_paper_labels(
    final_data: pd.DataFrame,
    paper_labels: pd.DataFrame | None,
    use_two_file_flow: bool,
) -> pd.DataFrame:
    """Merge paper labels on FILE_NAME (and STU_NUMBER if needed); fill study cols.

    All paper_labels tables are extraction data for the same paper: study-level
    info is broadcast so it appears on every row of the merged result. Columns
    from paper (PUBMED_ID, JOURNAL_NAME, TITLE, YEAR, BLINDING, etc.) are filled
    and kept. Rows in pl without FILE_NAME (e.g. study-level-only tables) are
    broadcast to all rows.
    """
    if paper_labels is None or paper_labels.empty:
        return final_data
    if "FILE_NAME" not in final_data.columns:
        return final_data
    pl = paper_labels.copy()
    pl.columns = pl.columns.str.upper()
    # Normalize common paper column names (e.g. typo BLINIDING → BLINDING)
    for old_name, new_name in _PAPER_COLUMN_ALIASES.items():
        if old_name in pl.columns and new_name not in pl.columns:
            pl = pl.rename(columns={old_name: new_name})

    # Merge on FILE_NAME when pl has it; otherwise broadcast all pl columns to every row
    pl_has_file_name = "FILE_NAME" in pl.columns
    if pl_has_file_name:
        merge_keys = ["FILE_NAME"]
        if (
            "STU_NUMBER" in pl.columns
            and pl["FILE_NAME"].duplicated().any()
            and "STU_NUMBER" in final_data.columns
        ):
            merge_keys = ["FILE_NAME", "STU_NUMBER"]
        final_data = final_data.merge(
            pl, on=merge_keys, how="left", suffixes=("", "_PAPER")
        )
        for base_col in _PAPER_FILL_COLS:
            paper_col = f"{base_col}_PAPER"
            if paper_col not in final_data.columns:
                continue
            if base_col not in final_data.columns:
                final_data[base_col] = final_data[paper_col]
            else:
                final_data[base_col] = final_data[base_col].where(
                    final_data[base_col].notna(), final_data[paper_col]
                )
        paper_drop = [c for c in final_data.columns if c.endswith("_PAPER")]
        final_data = final_data.drop(columns=paper_drop, errors="ignore")
    else:
        # No FILE_NAME in pl: single paper, broadcast every pl column to all rows
        for col in pl.columns:
            if pl[col].notna().any():
                first_val = pl[col].dropna().iloc[0]
                final_data[col] = first_val
                logger.debug("Broadcast paper_labels column %s to all rows", col)

    # Broadcast study-level-only rows: columns from pl that are all NaN (e.g. from
    # a table with no FILE_NAME when we had multiple paper tables).
    # Get first value from pl for those columns.
    pl_cols = set(pl.columns)
    for col in pl_cols:
        if col not in final_data.columns:
            continue
        if final_data[col].isna().all() and pl[col].notna().any():
            first_val = pl[col].dropna().iloc[0]
            final_data[col] = first_val
            logger.debug("Broadcast paper_labels column %s to all rows", col)

    if "GROUP" in final_data.columns:
        for c in POST_FILL_COLS:
            if c not in final_data.columns:
                continue
            final_data[c] = final_data.groupby("GROUP")[c].transform(
                lambda x: x.where(x.notna(), x.ffill().bfill())
            )
    for c in ["STD_TRT", "STD_TRT_CLASS"]:
        if c in final_data.columns and "GROUP" in final_data.columns:
            final_data[c] = final_data.groupby("GROUP")[c].transform(
                lambda x: x.where(x.notna(), x.ffill().bfill())
            )
    if (
        use_two_file_flow
        and "EVENT_TYPE" in final_data.columns
        and "FILE_NAME" in final_data.columns
    ):
        dose_mask = final_data["EVENT_TYPE"].eq("DOSE")
        for c in ["STD_TRT", "STD_TRT_CLASS"]:
            if c not in final_data.columns:
                continue
            m = (
                final_data.loc[~dose_mask, ["FILE_NAME", c]]
                .dropna()
                .drop_duplicates(subset=["FILE_NAME"])
                .set_index("FILE_NAME")[c]
            )
            if not m.empty:
                final_data.loc[dose_mask, c] = final_data.loc[dose_mask, c].where(
                    final_data.loc[dose_mask, c].notna(),
                    final_data.loc[dose_mask, "FILE_NAME"].map(m),
                )
    return final_data


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one column per name; coalesce values from duplicate-name columns."""
    cols = list(df.columns)
    first_idx: dict[str, int] = {}
    keep = [True] * len(cols)
    for i, c in enumerate(cols):
        if c in first_idx:
            j = first_idx[c]
            a = df.iloc[:, j]
            b = df.iloc[:, i]
            df.iloc[:, j] = a.where(a.notna(), b)
            keep[i] = False
        else:
            first_idx[c] = i
    return df.iloc[:, [i for i, k in enumerate(keep) if k]].copy()


def run_merge_flow(
    tables_by_type: TablesByType,
    errors: list[SingleError],
) -> tuple[pd.DataFrame | None, list[SingleError]]:
    """
    Run full merge pipeline. Before: tables_by_type, errors. After:
    (final_data, errors) with FINAL_COLUMNS order and NA normalized; or
    (None, errors).
    """
    try:
        # Step 1: Transform all tables (clean names, group, aliases)
        _transform_tables_by_type(tables_by_type)
    except Exception as e:
        logger.exception(f"Transform step failed: {e}")
        errors.append({"error_name": "Transform failed", "error_message": str(e)})
        return None, errors

    obs_all, dose, cov, paper_labels, use_two_file_flow = _build_obs_dose_cov(
        tables_by_type
    )

    # Step 2: Two-file flow only — convert literal "na"/"n/a" to NaN
    if use_two_file_flow:
        if obs_all is not None and not obs_all.empty:
            obs_all = na_strings_to_nan(obs_all)
        if dose is not None and not dose.empty:
            dose = na_strings_to_nan(dose)
        if cov is not None and not cov.empty:
            cov = na_strings_to_nan(cov)
        if paper_labels is not None and not paper_labels.empty:
            paper_labels = na_strings_to_nan(paper_labels)

    total_tables = (
        (1 if obs_all is not None else 0)
        + (1 if dose is not None else 0)
        + (1 if cov is not None else 0)
        + (1 if paper_labels is not None and not paper_labels.empty else 0)
    )
    if total_tables == 0:
        errors.append(
            {
                "error_name": "No tables",
                "error_message": "No tables could be loaded; nothing to merge.",
            }
        )
        return None, errors

    # Step 3: Arm duration per group; fill missing dose group from obs (file_name/doi)
    try:
        if dose is not None and not dose.empty:
            dose = _compute_arm_dur(dose)
        if dose is not None and obs_all is not None and not obs_all.empty:
            dose = _dose_group_backfill_from_obs(dose, obs_all)
    except Exception as e:
        logger.warning(f"ARM_DUR or dose backfill failed: {e}")
        errors.append(
            {"error_name": "ARM_DUR / dose backfill", "error_message": str(e)}
        )

    if obs_all is not None and not obs_all.empty:
        obs_all = _ci_variance_aliases(obs_all)
        for c in ["dv_var_upper", "dv_var_lower"]:
            if c in obs_all.columns:
                obs_all = obs_all.rename(columns={c: f"{c}_obs"})

    # Step 4: event_data = concat(obs_all, dose) with event_type OBS/DOSE
    event_data = _event_data_and_group_name(
        obs_all, dose, use_two_file_flow=use_two_file_flow
    )
    if (
        event_data.empty
        and cov is None
        and (paper_labels is None or paper_labels.empty)
    ):
        errors.append(
            {
                "error_name": "No event data",
                "error_message": "No observation or dosing data to merge.",
            }
        )
        return None, errors

    if event_data.empty:
        if cov is not None and not cov.empty:
            final_data = cov.copy()
        elif paper_labels is not None and not paper_labels.empty:
            final_data = paper_labels.copy()
        else:
            return None, errors
    else:
        final_data = event_data.copy()

    # Step 5: Merge covariate (on group)
    if cov is not None and not cov.empty and not final_data.empty:
        try:
            final_data = _merge_cov_and_restore_dv(final_data, cov)
        except Exception as e:
            errors.append({"error_name": "Merge covariate", "error_message": str(e)})

    # Step 6: Backfill FROM_OBS_COLS, arm-level; sort; EVID/MDV; fill DOSE from OBS
    final_data = _fill_dose_from_obs(final_data)
    final_data = _backfill_arm_level(final_data)
    final_data["arm_time"] = pd.to_numeric(final_data.get("arm_time"), errors="coerce")
    if "event_type" in final_data.columns:
        final_data["_ord"] = final_data["event_type"].map({"DOSE": 0, "OBS": 1})
        sort_cols = ["group", "arm_time", "_ord"]
        if "_obs_origin" in final_data.columns:
            sort_cols = ["group", "_obs_origin", "arm_time", "_ord"]
        final_data = final_data.sort_values(by=sort_cols).drop(
            columns=["_ord", "_obs_origin"], errors="ignore"
        )
    else:
        sort_cols = [c for c in ["group", "arm_time"] if c in final_data.columns]
        if sort_cols:
            final_data = final_data.sort_values(by=sort_cols)

    final_data.columns = final_data.columns.str.upper()
    final_data["EVID"] = np.nan
    final_data["MDV"] = np.nan
    if "EVENT_TYPE" in final_data.columns:
        final_data.loc[final_data["EVENT_TYPE"] == "DOSE", ["EVID", "MDV"]] = [1, 1]
        final_data.loc[final_data["EVENT_TYPE"] == "OBS", ["EVID", "MDV"]] = [0, 0]
    else:
        final_data["EVID"] = 0
        final_data["MDV"] = 0
    final_data["EVID"] = final_data["EVID"].astype("Int64")
    final_data["MDV"] = final_data["MDV"].astype("Int64")

    if "EVENT_TYPE" in final_data.columns and "GROUP" in final_data.columns:
        obs_m = final_data["EVENT_TYPE"].eq("OBS")
        dose_m = final_data["EVENT_TYPE"].eq("DOSE")
        obs_groups = set(final_data.loc[obs_m, "GROUP"].dropna().unique())
        for col in [
            "FILE_NAME",
            "STU_NUMBER",
            "TRIAL_NAME",
            "GROUP_NAME",
            "TITLE",
            "JOURNAL_NAME",
            "YEAR",
            "DOI_URL",
            "CIT_URL",
        ]:
            if col not in final_data.columns:
                continue
            src = final_data.loc[obs_m, ["GROUP", col]].dropna(subset=["GROUP"])
            if src.empty:
                continue
            m = src.dropna(subset=[col]).groupby("GROUP")[col].first()
            if m.empty:
                continue
            final_data.loc[dose_m, col] = final_data.loc[dose_m, col].where(
                final_data.loc[dose_m, col].notna(),
                final_data.loc[dose_m, "GROUP"].map(m),
            )
        dose_only = dose_m & ~final_data["GROUP"].isin(obs_groups)
        if dose_only.any():
            if "FILE_NAME" in final_data.columns:
                final_data.loc[dose_only, "FILE_NAME"] = np.nan
            if "DOI_URL" in final_data.columns:
                final_data.loc[dose_only, "DOI_URL"] = np.nan

    if "GROUP" in final_data.columns:
        for c in POST_FILL_COLS:
            if c not in final_data.columns:
                continue
            final_data[c] = final_data.groupby("GROUP")[c].transform(
                lambda x: x.where(x.notna(), x.ffill().bfill())
            )
    for c in ["STD_TRT", "STD_TRT_CLASS"]:
        if c in final_data.columns and "GROUP" in final_data.columns:
            final_data[c] = final_data.groupby("GROUP")[c].transform(
                lambda x: x.where(x.notna(), x.ffill().bfill())
            )

    if "EVENT_TYPE" in final_data.columns and "FILE_NAME" in final_data.columns:
        dose_mask = final_data["EVENT_TYPE"].eq("DOSE")
        for c in ["STD_TRT", "STD_TRT_CLASS"]:
            if c not in final_data.columns:
                continue
            m = (
                final_data.loc[~dose_mask, ["FILE_NAME", c]]
                .dropna()
                .drop_duplicates(subset=["FILE_NAME"])
                .set_index("FILE_NAME")[c]
            )
            if not m.empty:
                final_data.loc[dose_mask, c] = final_data.loc[dose_mask, c].where(
                    final_data.loc[dose_mask, c].notna(),
                    final_data.loc[dose_mask, "FILE_NAME"].map(m),
                )

    # Step 7: Paper labels merge via _merge_paper_labels(); rename to final schema
    if "FILENAME" in final_data.columns:
        if "FILE_NAME" in final_data.columns:
            final_data["FILE_NAME"] = final_data["FILE_NAME"].where(
                final_data["FILE_NAME"].notna(), final_data["FILENAME"]
            )
        final_data = final_data.drop(columns=["FILENAME"], errors="ignore")
    if "DOI" in final_data.columns and "DOI_URL" in final_data.columns:
        final_data["DOI_URL"] = final_data["DOI_URL"].where(
            final_data["DOI_URL"].notna(), final_data["DOI"]
        )
        final_data = final_data.drop(columns=["DOI"], errors="ignore")

    final_data = _merge_paper_labels(final_data, paper_labels, use_two_file_flow)
    final_data = final_data.rename(
        columns={k: v for k, v in RENAME_MAP.items() if k in final_data.columns}
    )
    final_data = _coalesce_duplicate_columns(final_data)

    if (
        "GROUP_NAME_DOSE" in final_data.columns
        and "GROUP_NAME" in final_data.columns
        and "EVENT_TYPE" in final_data.columns
    ):
        dose_mask = final_data["EVENT_TYPE"].eq("DOSE")
        final_data.loc[dose_mask, "GROUP_NAME"] = final_data.loc[
            dose_mask, "GROUP_NAME"
        ].where(
            final_data.loc[dose_mask, "GROUP_NAME"].notna(),
            final_data.loc[dose_mask, "GROUP_NAME_DOSE"],
        )
        final_data = final_data.drop(columns=["GROUP_NAME_DOSE"], errors="ignore")
    if "GROUP_NAME" not in final_data.columns and "GROUP" in final_data.columns:
        final_data["GROUP_NAME"] = final_data["GROUP"]
    if "TRIAL_NAME" not in final_data.columns and "REGNM" in final_data.columns:
        final_data["TRIAL_NAME"] = final_data["REGNM"]
    if (
        "AFF_BSA_STAT" not in final_data.columns
        and "AFF_BSA_BL_STAT" in final_data.columns
    ):
        final_data["AFF_BSA_STAT"] = final_data["AFF_BSA_BL_STAT"]

    # Coerce integer-like columns so they output without decimals (e.g. 21719096)
    for col in ("PUBMED_ID", "YEAR"):
        if col in final_data.columns:
            num = pd.to_numeric(final_data[col], errors="coerce")
            final_data[col] = num.astype("Int64")

    for col in FINAL_COLUMNS:
        if col not in final_data.columns:
            final_data[col] = np.nan
    final_data = final_data[FINAL_COLUMNS]
    final_data = final_data.replace(r"(?i)^na$", "NA", regex=True)
    return final_data, errors
