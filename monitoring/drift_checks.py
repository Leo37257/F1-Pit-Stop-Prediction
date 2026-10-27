"""Model-free drift checks with traffic-light thresholds.

Evidently draws the dashboards; this module gives the same question a number and a verdict that a test, a DAG task
or a CI step can act on. It needs only numpy, pandas and scipy and does not touch the champion model.

PSI (population stability index) compares binned distributions:  < 0.10 stable, 0.10-0.25 moderate shift,
>= 0.25 major shift (the usual credit-risk rule of thumb). The two-sample KS statistic is reported next to it
for numeric columns because it is insensitive to how the bins were chosen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN = 0.10
PSI_ALERT = 0.25
F1_DROP_WARN = 0.03      # absolute macro-F1 drop versus the reference score
F1_DROP_ALERT = 0.07
_EPS = 1e-4              # floor for empty bins so the log never sees zero


def _frac(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return np.clip(counts / total, _EPS, None)


def psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    r, c = _frac(np.asarray(ref_counts, float)), _frac(np.asarray(cur_counts, float))
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI with decile edges taken from the reference, so the reference is uniform by construction."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if np.unique(ref).size <= bins:          # binary or few-valued column (incl. constant): compare value shares
        return psi_categorical(pd.Series(ref), pd.Series(cur))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_counts(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref, cur = reference.astype("string").fillna("<NA>"), current.astype("string").fillna("<NA>")
    cats = sorted(set(ref) | set(cur))
    rc = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
    cc = cur.value_counts().reindex(cats, fill_value=0).to_numpy()
    return psi_from_counts(rc, cc)


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")
    return float(ks_2samp(ref, cur).statistic)


def classify_psi(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    return "alert" if psi >= PSI_ALERT else "warn" if psi >= PSI_WARN else "ok"


def feature_drift_table(reference: pd.DataFrame, current: pd.DataFrame,
                        numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """One row per feature: psi, ks (numeric only), status. Sorted by psi, worst first."""
    rows = []
    for col in numeric:
        p = psi_numeric(reference[col], current[col])
        rows.append({"feature": col, "kind": "numeric", "psi": p, "ks": ks_statistic(reference[col], current[col]),
                     "status": classify_psi(p)})
    for col in categorical:
        p = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "kind": "categorical", "psi": p, "ks": np.nan, "status": classify_psi(p)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def performance_status(reference_f1: float, current_f1: float) -> dict:
    drop = round(float(reference_f1 - current_f1), 4)
    status = "alert" if drop >= F1_DROP_ALERT else "warn" if drop >= F1_DROP_WARN else "ok"
    return {"f1_drop": drop, "status": status}


def worst_status(statuses) -> str:
    order = {"unknown": 0, "ok": 1, "warn": 2, "alert": 3}
    return max(statuses, key=lambda s: order.get(s, 0), default="ok")
