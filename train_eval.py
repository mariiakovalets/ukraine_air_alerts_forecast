"""Rolling-origin evaluation of baselines and models for 'will a new alert start in Kyiv City within the next hour'."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import (
    FEATURES, HORIZON, build_features, build_target, load_alerts, load_all_alerts, make_grid,
)

# name -> (start, end, refit interval). Validation refits weekly to keep the long window fast;
# the test refits daily, as the model would run in practice.
PERIODS = {
    "val": (pd.Timestamp("2026-01-05", tz="UTC"), pd.Timestamp("2026-08-10", tz="UTC"), pd.Timedelta(days=7)),
    "test": (pd.Timestamp("2026-08-10", tz="UTC"), pd.Timestamp("2026-09-07", tz="UTC"), pd.Timedelta(days=1)),
}
BASELINE_WINDOW = pd.Timedelta(days=28)
DECAY_HALF_LIFE = pd.Timedelta(days=90)
PROB_CLIP = 1e-3
SEED = 42

OUT_DIR = Path("outputs")


def make_logreg() -> Pipeline:
    calendar = ["hour_local", "dow_local"]
    numeric = [c for c in FEATURES if c not in (*calendar, "minutes_since_last_start")]
    pre = ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore"), calendar),
        ("numeric", StandardScaler(), numeric),
        ("since_start", Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
        ]), ["minutes_since_last_start"]),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=50,
        early_stopping=False, random_state=SEED,
    )


# name -> (model factory, use time-decay sample weights)
MODELS = {
    "logreg": (make_logreg, False),
    "logreg_decay90": (make_logreg, True),
    "hgb": (make_hgb, False),
    "hgb_decay90": (make_hgb, True),
}


def decay_weights(train_index: pd.DatetimeIndex, now: pd.Timestamp) -> np.ndarray:
    age = (now - train_index) / DECAY_HALF_LIFE
    return 0.5 ** age.to_numpy(dtype=float)


def fit_model(make_model, use_decay: bool, X_tr: pd.DataFrame, y_tr: pd.Series, now: pd.Timestamp):
    model = make_model()
    if not use_decay:
        return model.fit(X_tr, y_tr)
    weights = decay_weights(X_tr.index, now)
    if isinstance(model, Pipeline):
        return model.fit(X_tr, y_tr, model__sample_weight=weights)
    return model.fit(X_tr, y_tr, sample_weight=weights)


def baseline_predictions(known_y: pd.Series, known_hours: pd.Series, day_hours: pd.Series) -> dict:
    recent = known_y.index >= known_y.index[-1] + HORIZON - BASELINE_WINDOW
    y_recent, h_recent = known_y[recent], known_hours[recent]
    base_rate = y_recent.mean()
    by_hour = y_recent.groupby(h_recent).mean()
    return {
        "base_rate_28d": np.full(len(day_hours), base_rate),
        "hour_of_day_28d": day_hours.map(by_hour).fillna(base_rate).to_numpy(),
    }


def rolling_predict(X: pd.DataFrame, y: pd.Series, start, end, refit_every) -> pd.DataFrame:
    rows = []
    for day_start in pd.date_range(start, end, freq=refit_every, inclusive="left"):
        day_mask = (X.index >= day_start) & (X.index < min(day_start + refit_every, end))
        # A training row's target window [t, t + H) must be over before the first forecast of the day
        train_mask = X.index + HORIZON <= day_start
        assert X.index[train_mask].max() + HORIZON <= X.index[day_mask].min()

        X_tr, y_tr, X_day = X[train_mask], y[train_mask], X[day_mask]
        preds = baseline_predictions(y_tr, X_tr["hour_local"], X_day["hour_local"])
        for name, (make_model, use_decay) in MODELS.items():
            model = fit_model(make_model, use_decay, X_tr, y_tr, day_start)
            preds[name] = model.predict_proba(X_day)[:, 1]

        rows.append(pd.DataFrame({"target": y[day_mask], **preds}, index=X_day.index))
    return pd.concat(rows)


def score(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["target"]
    models = [c for c in pred.columns if c != "target"]
    ref = brier_score_loss(y, pred["hour_of_day_28d"].clip(PROB_CLIP, 1 - PROB_CLIP))
    table = {}
    for m in models:
        p = pred[m].clip(PROB_CLIP, 1 - PROB_CLIP)
        brier = brier_score_loss(y, p)
        table[m] = {
            "brier": brier,
            "bss_vs_hour_baseline": 1 - brier / ref,
            "log_loss": log_loss(y, p),
            "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "mean_pred": p.mean(),
        }
    return pd.DataFrame(table).T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("period", choices=PERIODS)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUT_DIR / f"{args.period}_metrics.json"
    if args.period == "test" and metrics_path.exists():
        print(f"{metrics_path} exists: the test period has already been evaluated. Refusing to run it again.")
        return 1

    all_alerts = load_all_alerts()
    alerts = load_alerts(all_alerts)
    times = make_grid()
    X = build_features(alerts, all_alerts, times)
    y = build_target(alerts, times)

    start, end, refit_every = PERIODS[args.period]
    pred = rolling_predict(X, y, start, end, refit_every)
    pred.to_csv(OUT_DIR / f"{args.period}_predictions.csv")

    table = score(pred)
    print(f"{args.period}: {start.date()} .. {end.date()}, {len(pred)} hours, positive share {pred['target'].mean():.3f}")
    print(table.round(4).to_string())
    metrics_path.write_text(json.dumps(table.to_dict(orient="index"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
