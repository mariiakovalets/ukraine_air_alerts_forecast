"""Sanity check of neighbor features: name matching, share of hours = 1, and Kyiv start rate conditional on them."""

import pandas as pd

from features import NEIGHBORS, build_features, build_target, load_alerts, load_all_alerts, make_grid

PERIODS = {
    "train (2024-09-01 .. 2026-07-12)": ("2024-09-01", "2026-07-13"),
    "val (2026-07-13 .. 2026-08-09)": ("2026-07-13", "2026-08-10"),
}
BINARY = [*(f"active_{slug}" for slug in NEIGHBORS.values()), "neighbor_start_1h"]


def main() -> None:
    all_alerts = load_all_alerts()
    missing = set(NEIGHBORS) - set(all_alerts["oblast"])
    print("Neighbor names missing from data:", missing or "none")

    alerts = load_alerts(all_alerts)
    times = make_grid()
    X = build_features(alerts, all_alerts, times)
    y = build_target(alerts, times)

    for label, (lo, hi) in PERIODS.items():
        mask = (X.index >= pd.Timestamp(lo, tz="UTC")) & (X.index < pd.Timestamp(hi, tz="UTC"))
        Xp, yp = X[mask], y[mask]
        print(f"\n=== {label}: {mask.sum()} hours, Kyiv start share {yp.mean():.3f} ===")

        rows = []
        for col in BINARY:
            on = Xp[col] == 1
            rows.append({
                "feature": col,
                "share_hours_eq_1": on.mean(),
                "kyiv_start_rate_if_1": yp[on].mean(),
                "kyiv_start_rate_if_0": yp[~on].mean(),
                "hours_eq_1": int(on.sum()),
            })
        print(pd.DataFrame(rows).round(3).to_string(index=False))

        print("\nKyiv start rate by neighbors_active:")
        print(yp.groupby(Xp["neighbors_active"]).agg(["mean", "size"]).round(3).T.to_string())


if __name__ == "__main__":
    main()