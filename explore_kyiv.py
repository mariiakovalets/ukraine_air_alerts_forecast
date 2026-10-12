"""Inspect Kyiv City alerts in the official dataset: duplicates, levels, monthly counts, hourly start rate."""

from pathlib import Path

import pandas as pd

CSV_PATH = Path("data") / "official_data_en.csv"
REGION = "Kyiv City"
MONTHS_TO_SHOW = 24
HORIZON_H = 1


def main() -> None:
    raw = pd.read_csv(CSV_PATH)
    dup_count = raw.duplicated().sum()
    print(f"Rows: {len(raw)}, exact duplicate rows: {dup_count} ({dup_count / len(raw):.0%})")

    df = raw.drop_duplicates()
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)

    kyiv = df[df["oblast"] == REGION].sort_values("started_at")
    print(f"\n{REGION}: {len(kyiv)} unique alerts")
    print("level values:")
    print(kyiv["level"].value_counts(dropna=False))
    print("non-empty raion / hromada:", kyiv["raion"].notna().sum(), "/", kyiv["hromada"].notna().sum())
    print("alerts overlapping the previous one:", (kyiv["started_at"] < kyiv["finished_at"].shift()).sum())

    # Grid in UTC so the October DST switch does not create duplicate or missing hours
    grid_start = kyiv["started_at"].min().floor("D")
    grid_end = kyiv["started_at"].max().ceil("D")
    hours = pd.date_range(grid_start, grid_end, freq="h", inclusive="left")
    start_hours = set(kyiv["started_at"].dt.floor("h"))
    target = pd.Series([h in start_hours for h in hours], index=hours, dtype=int)

    duration_min = (kyiv["finished_at"] - kyiv["started_at"]).dt.total_seconds() / 60
    month = kyiv["started_at"].dt.strftime("%Y-%m")
    monthly = pd.DataFrame({
        "alerts": kyiv.groupby(month).size(),
        "median_minutes": duration_min.groupby(month).median().round(1),
        "share_hours_with_start": target.groupby(target.index.strftime("%Y-%m")).mean().round(3),
    })
    print(f"\nMonthly stats, last {MONTHS_TO_SHOW} months (last month may be partial):")
    print(monthly.tail(MONTHS_TO_SHOW).to_string())

    last_hour = hours[-1]
    for label, days in (("last 365 days", 365), ("last 28 days", 28)):
        window = target[target.index > last_hour - pd.Timedelta(days=days)]
        print(f"Share of hours with an alert start ({label}): {window.mean():.3f}")


if __name__ == "__main__":
    main()