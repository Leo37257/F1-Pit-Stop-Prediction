"""Feature engineering ported from notebooks/EDA.ipynb (add_features cell).

All features are scoped to (Year, Race, Driver, Stint) so nothing leaks across
pit stops, races, or years. Rolling features apply shift(1) before rolling so
the current lap is never part of its own feature.
"""

from __future__ import annotations

import pandas as pd

GROUP_KEY = ["Year", "Race", "Driver", "Stint"]
SORT_KEY = GROUP_KEY + ["LapNumber"]

TYRELIFE_BUCKETS = {
    "bins": [-0.1, 5, 10, 15, 20, 25, 30, 40, 60, 200],
    "labels": ["0_5", "6_10", "11_15", "16_20", "21_25", "26_30", "31_40", "41_60", "60plus"],
}


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(SORT_KEY).reset_index(drop=True)
    g = df.groupby(GROUP_KEY, sort=False)

    for lag in (1, 2, 3):
        df[f"LapTime_lag{lag}"] = g["LapTime (s)"].shift(lag)
        df[f"LapTime_Delta_lag{lag}"] = g["LapTime_Delta"].shift(lag)
        df[f"Position_lag{lag}"] = g["Position"].shift(lag)

    for w in (3, 5):
        df[f"LapTime_roll{w}_mean"] = g["LapTime (s)"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"LapTime_roll{w}_std"] = g["LapTime (s)"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).std()
        )
        df[f"LapTimeDelta_roll{w}_mean"] = g["LapTime_Delta"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    df["TyreLife_bucket"] = pd.cut(
        df["TyreLife"],
        bins=TYRELIFE_BUCKETS["bins"],
        labels=TYRELIFE_BUCKETS["labels"],
    ).astype(str)

    df["IsEarlyRace"] = (df["RaceProgress"] < 0.25).astype(int)
    df["IsLateRace"] = (df["RaceProgress"] > 0.70).astype(int)

    df["StintMin_sofar"] = g["LapTime (s)"].cummin()
    df["LapTime_vs_StintMin"] = df["LapTime (s)"] - df["StintMin_sofar"]

    return df
