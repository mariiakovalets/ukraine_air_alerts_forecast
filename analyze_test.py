"""Diagnostics for the final model's test predictions: separation, top-ranked hours, calibration, missed alerts."""

from pathlib import Path

import pandas as pd

from features import build_features, load_alerts, load_all_alerts, make_grid

PRED_PATH = Path("outputs") / "test_predictions.csv"
MODEL = "logreg"
BASELINE = "hour_of_day_28d"
LOCAL_TZ = "Europe/Kyiv"
TOP_SHARES = (0.05, 0.10, 0.20)
CALIBRATION_BINS = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
CONTEXT = ["alert_active", "neighbors_active", "active_kyivska", "oblasts_active_ukraine", "starts_24h"]


def main() -> None:
    pred = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    y, p = pred["target"], pred[MODEL]
    base_rate = y.mean()
    print(f"{len(pred)} hours, {int(y.sum())} with an alert start ({base_rate:.3f})")

    print(f"\n1) {MODEL} prediction by actual outcome:")
    print(p.groupby(y).describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())

    print("\n2) Hours ranked by predicted risk:")
    ranked = pred.sort_values(MODEL, ascending=False)
    rows = []
    for share in TOP_SHARES:
        for name in (MODEL, BASELINE):
            top = pred.sort_values(name, ascending=False).head(round(len(pred) * share))
            rows.append({
                "top_share": f"{share:.0%}", "model": name, "hours": len(top),
                "alert_rate": top["target"].mean(),
                "lift": top["target"].mean() / base_rate,
                "alerts_captured": top["target"].sum() / y.sum(),
            })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n3) Calibration (predicted vs actual by bin):")
    bins = pd.cut(p, CALIBRATION_BINS)
    calib = pred.groupby(bins, observed=True).agg(
        hours=("target", "size"), mean_pred=(MODEL, "mean"), actual_rate=("target", "mean"),
    )
    print(calib.round(3).to_string())

    all_alerts = load_all_alerts()
    X = build_features(load_alerts(all_alerts), all_alerts, make_grid()).loc[pred.index, CONTEXT]
    view = pred[["target", MODEL]].join(X)
    view.index = view.index.tz_convert(LOCAL_TZ).strftime("%m-%d %H:%M")

    print("\n4) 15 highest-risk hours (Kyiv time):")
    print(view.loc[ranked.index[:15].tz_convert(LOCAL_TZ).strftime("%m-%d %H:%M")].round(3).to_string())

    print("\n5) Context of alert hours, by predicted risk:")
    pos = view[view["target"] == 1]
    low = pos[MODEL] < 0.10
    summary = pd.DataFrame({
        f"pred < 0.10 ({int(low.sum())} hours)": pos[low][CONTEXT].mean(),
        f"pred >= 0.10 ({int((~low).sum())} hours)": pos[~low][CONTEXT].mean(),
        "all non-alert hours": view[view["target"] == 0][CONTEXT].mean(),
    })
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()