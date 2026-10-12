"""Load alerts, build the hourly grid, the target for Kyiv City, and features known at each forecast time t."""

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data") / "official_data_en.csv"
REGION = "Kyiv City"
LOCAL_TZ = "Europe/Kyiv"
HORIZON = pd.Timedelta(hours=1)

NEIGHBORS = {
    "Kyivska oblast": "kyivska",
    "Chernihivska oblast": "chernihivska",
    "Zhytomyrska oblast": "zhytomyrska",
    "Vinnytska oblast": "vinnytska",
    "Cherkaska oblast": "cherkaska",
    "Poltavska oblast": "poltavska",
    "Sumska oblast": "sumska",
}
NEIGHBOR_START_WINDOW = pd.Timedelta(hours=1)

COUNT_WINDOWS = {
    "starts_24h": pd.Timedelta(hours=24),
    "starts_7d": pd.Timedelta(days=7),
    "starts_30d": pd.Timedelta(days=30),
}
NEIGHBOR_FEATURES = [
    *(f"active_{slug}" for slug in NEIGHBORS.values()),
    "neighbors_active",
    "oblasts_active_ukraine",
    "neighbor_start_1h",
]
FEATURES = [
    *COUNT_WINDOWS, "alert_active", "minutes_since_last_start", "hour_local", "dow_local",
    *NEIGHBOR_FEATURES,
]

GRID_START = pd.Timestamp("2024-09-01", tz="UTC")
# Source channel stopped on 2026-09-07 (last message 05:39 UTC). Alerts still running then were never
# written to the CSV, so the grid ends a few hours earlier to avoid false zeros in the target.
GRID_END = pd.Timestamp("2026-09-07", tz="UTC")


def load_all_alerts(path: Path = CSV_PATH) -> pd.DataFrame:
    """All regions, oblast and raion level. Hromada-level alerts are too local to describe a region."""
    # ~38% of rows in the source CSV are exact duplicates
    df = pd.read_csv(path).drop_duplicates()
    df = df[df["level"] != "hromada"].copy()
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.as_unit("ns")
    assert (df["finished_at"] >= df["started_at"]).all(), "alert ends before it starts"
    return df.sort_values("started_at").reset_index(drop=True)[["oblast", "started_at", "finished_at"]]


def load_alerts(all_alerts: pd.DataFrame, region: str = REGION) -> pd.DataFrame:
    df = all_alerts[all_alerts["oblast"] == region].reset_index(drop=True)[["started_at", "finished_at"]]
    assert not (df["started_at"] < df["finished_at"].shift()).any(), "overlapping alerts"
    return df


def make_grid(start: pd.Timestamp = GRID_START, end: pd.Timestamp = GRID_END) -> pd.DatetimeIndex:
    # UTC grid: the DST switch would otherwise create a duplicate hour
    return pd.date_range(start, end - HORIZON, freq="h", inclusive="both").as_unit("ns")


def _ns(values) -> np.ndarray:
    # asi8 depends on the datetime unit; arithmetic below assumes nanoseconds
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def build_target(alerts: pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series:
    """1 if a new alert starts in [t, t + HORIZON)."""
    s = _ns(alerts["started_at"])
    t = _ns(times)
    n_starts = np.searchsorted(s, t + HORIZON.value, side="left") - np.searchsorted(s, t, side="left")
    return pd.Series((n_starts > 0).astype(int), index=times, name="target")


def _active_at(rows: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """True where at least one alert in rows has started_at < t < finished_at (overlaps allowed)."""
    # Zero-length alerts are never active; without them finished_at <= t implies started_at < t
    rows = rows[rows["finished_at"] > rows["started_at"]]
    started = np.searchsorted(np.sort(_ns(rows["started_at"])), t, side="left")
    finished = np.searchsorted(np.sort(_ns(rows["finished_at"])), t, side="right")
    return started - finished > 0


def build_features(alerts: pd.DataFrame, all_alerts: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    s = _ns(alerts["started_at"])
    f = _ns(alerts["finished_at"])
    t = _ns(times)

    n_before = np.searchsorted(s, t, side="left")
    out = pd.DataFrame(index=times)
    for name, window in COUNT_WINDOWS.items():
        out[name] = n_before - np.searchsorted(s, t - window.value, side="left")

    last = np.maximum(n_before - 1, 0)
    has_prev = n_before > 0
    # Kyiv alerts never overlap (asserted in load_alerts), so only the latest started one can be active
    out["alert_active"] = (has_prev & (f[last] > t)).astype(int)
    out["minutes_since_last_start"] = np.where(has_prev, (t - s[last]) / 60e9, np.nan)

    local = times.tz_convert(LOCAL_TZ)
    out["hour_local"] = local.hour
    out["dow_local"] = local.dayofweek

    for oblast, slug in NEIGHBORS.items():
        out[f"active_{slug}"] = _active_at(all_alerts[all_alerts["oblast"] == oblast], t).astype(int)
    out["neighbors_active"] = out[[f"active_{slug}" for slug in NEIGHBORS.values()]].sum(axis=1)

    others = all_alerts[all_alerts["oblast"] != REGION]
    out["oblasts_active_ukraine"] = sum(
        _active_at(rows, t).astype(int) for _, rows in others.groupby("oblast")
    )

    ns = np.sort(_ns(all_alerts.loc[all_alerts["oblast"].isin(NEIGHBORS), "started_at"]))
    recent_starts = np.searchsorted(ns, t, side="left") - np.searchsorted(ns, t - NEIGHBOR_START_WINDOW.value, side="left")
    out["neighbor_start_1h"] = (recent_starts > 0).astype(int)
    return out[FEATURES]


def history_at(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """What was known at t: alerts started before t; an end after t is still unknown."""
    known = alerts[alerts["started_at"] < t].copy()
    known.loc[known["finished_at"] > t, "finished_at"] = pd.NaT
    return known


def features_from_history(history: pd.DataFrame, all_history: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Slow reference implementation, used by the leakage test."""
    starts = history["started_at"]
    row = {name: int(((starts >= t - w) & (starts < t)).sum()) for name, w in COUNT_WINDOWS.items()}
    row["alert_active"] = int(history["finished_at"].isna().any())
    row["minutes_since_last_start"] = (t - starts.max()).total_seconds() / 60 if len(history) else np.nan
    local = t.tz_convert(LOCAL_TZ)
    row["hour_local"] = local.hour
    row["dow_local"] = local.dayofweek

    ongoing = all_history.loc[all_history["finished_at"].isna(), "oblast"]
    for oblast, slug in NEIGHBORS.items():
        row[f"active_{slug}"] = int((ongoing == oblast).any())
    row["neighbors_active"] = sum(row[f"active_{slug}"] for slug in NEIGHBORS.values())
    row["oblasts_active_ukraine"] = ongoing[ongoing != REGION].nunique()

    neighbor_starts = all_history.loc[all_history["oblast"].isin(NEIGHBORS), "started_at"]
    row["neighbor_start_1h"] = int((neighbor_starts >= t - NEIGHBOR_START_WINDOW).any())
    return row
