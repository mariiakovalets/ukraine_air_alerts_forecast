"""Leakage test: features computed on the full data must equal features computed from history known at t."""

import sys

import numpy as np
import pandas as pd

from features import (
    FEATURES, HORIZON, _ns, build_features, build_target, features_from_history,
    history_at, load_alerts, load_all_alerts, make_grid,
)

N_RANDOM_HOURS = 300
SEED = 42


def sample_times(alerts: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    rng = np.random.default_rng(SEED)
    random_hours = times[rng.choice(len(times), N_RANDOM_HOURS, replace=False)]

    # Hours around alert boundaries are where leaks usually hide
    in_grid = alerts[(alerts["started_at"] >= times[0]) & (alerts["finished_at"] < times[-1])]
    picked = in_grid.sample(min(100, len(in_grid)), random_state=SEED)
    edges = pd.DatetimeIndex(pd.concat([
        picked["started_at"].dt.floor("h"),
        picked["started_at"].dt.ceil("h"),
        picked["finished_at"].dt.floor("h"),
    ]))
    return random_hours.union(edges.intersection(times))


def truncation_mismatches(alerts, all_alerts, features: pd.DataFrame, check_times) -> dict[str, int]:
    mismatches = dict.fromkeys(FEATURES, 0)
    for t in check_times:
        expected = features_from_history(history_at(alerts, t), history_at(all_alerts, t), t)
        actual = features.loc[t]
        for name in FEATURES:
            if not np.isclose(actual[name], expected[name], equal_nan=True):
                mismatches[name] += 1
    return {k: v for k, v in mismatches.items() if v}


def leaky_alert_active(alerts: pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series:
    """Deliberately wrong: 'any alert during [t, t + 1h)', which peeks into the next hour."""
    s = _ns(alerts["started_at"])
    f = _ns(alerts["finished_at"])
    t = _ns(times)
    started_before_end = np.searchsorted(s, t + HORIZON.value, side="left")
    finished_before_t = np.searchsorted(np.sort(f), t, side="right")
    return pd.Series((started_before_end - finished_before_t > 0).astype(int), index=times)


def main() -> int:
    all_alerts = load_all_alerts()
    alerts = load_alerts(all_alerts)
    times = make_grid()
    features = build_features(alerts, all_alerts, times)
    target = build_target(alerts, times)
    check_times = sample_times(alerts, times)

    assert features.index.equals(target.index)
    print(f"Grid: {times[0]} .. {times[-1]}, {len(times)} hours, positive share {target.mean():.3f}")
    print(f"Checking {len(check_times)} forecast times")

    mismatches = truncation_mismatches(alerts, all_alerts, features, check_times)
    if mismatches:
        print("FAIL: features use information from after t:", mismatches)
        return 1
    print("PASS: all features match the history-at-t reference")

    leaky = features.copy()
    leaky["alert_active"] = leaky_alert_active(alerts, times)
    caught = truncation_mismatches(alerts, all_alerts, leaky, check_times)
    if "alert_active" not in caught:
        print("FAIL: the test did not catch a deliberately leaky feature")
        return 1
    print(f"PASS: deliberately leaky feature caught ({caught['alert_active']} mismatches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
