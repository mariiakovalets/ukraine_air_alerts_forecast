"""Download the volunteer air raid alert dataset, run freshness checks, and print Kyiv alerts for manual verification."""

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
FILE_PATH = "datasets/volunteer_data_en.csv"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_PATH}"
COMMITS_API = f"https://api.github.com/repos/{REPO}/commits?path={FILE_PATH}&per_page=1"

DATA_DIR = Path("data")
CSV_OUT = DATA_DIR / "volunteer_data_en.csv"
SNAPSHOT_OUT = DATA_DIR / "snapshot_volunteer.json"
OFFICIAL_CSV = DATA_DIR / "official_data_en.csv"

LOCAL_TZ = "Europe/Kyiv"
KYIV_REGIONS = ("Kyiv City", "Kyivska oblast")
MAX_STALENESS_HOURS = 48
DAYS_TO_CHECK = 2


def fetch_latest_commit() -> dict:
    try:
        with urllib.request.urlopen(COMMITS_API, timeout=30) as resp:
            commit = json.load(resp)[0]
    except Exception as exc:
        print(f"WARNING: could not fetch commit info ({exc}); snapshot will lack the SHA")
        return {"sha": None, "committed_at": None}
    return {"sha": commit["sha"], "committed_at": commit["commit"]["committer"]["date"]}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def print_overview(df: pd.DataFrame, now: datetime) -> None:
    print("Columns:", list(df.columns))
    print(df.head(), "\n")

    last_start = df["started_at"].max()
    staleness_h = (now - last_start).total_seconds() / 3600
    print(f"Rows: {len(df)}")
    print(f"Range: {df['started_at'].min()} -> {last_start} (UTC)")
    print(f"Latest alert started {staleness_h:.1f} h ago")
    if staleness_h > MAX_STALENESS_HOURS:
        print("WARNING: data looks stale")

    print("\nRegions containing 'Kyiv':")
    print(df.loc[df["region"].str.contains("Kyiv"), "region"].value_counts())

    print("\nShare of naive alerts (end time imputed as start + 30 min), last 6 months:")
    naive_share = (
        df.assign(month=df["started_at"].dt.strftime("%Y-%m"))
        .groupby("month")["naive"].mean()
        .tail(6)
    )
    print(naive_share.round(3))


def print_manual_check(df: pd.DataFrame, now: datetime) -> None:
    today_local = pd.Timestamp(now).tz_convert(LOCAL_TZ).normalize()
    for days_ago in range(DAYS_TO_CHECK, 0, -1):
        day_start = today_local - timedelta(days=days_ago)
        day_end = day_start + timedelta(days=1)
        for region in KYIV_REGIONS:
            rows = df[
                (df["region"] == region)
                & (df["started_at"] >= day_start)
                & (df["started_at"] < day_end)
            ].sort_values("started_at")
            print(f"\n{region}, {day_start.date()} (Kyiv local time): {len(rows)} alerts")
            if rows.empty:
                continue
            table = pd.DataFrame({
                "start": rows["started_at"].dt.tz_convert(LOCAL_TZ).dt.strftime("%H:%M"),
                "end": rows["finished_at"].dt.tz_convert(LOCAL_TZ).dt.strftime("%H:%M"),
                "minutes": ((rows["finished_at"] - rows["started_at"]).dt.total_seconds() / 60).round(1),
                "naive": rows["naive"],
            })
            print(table.to_string(index=False))


def compare_with_official(vol: pd.DataFrame) -> None:
    if not OFFICIAL_CSV.exists():
        print(f"\n{OFFICIAL_CSV} not found, skipping source comparison")
        return

    off = load(OFFICIAL_CSV)
    overlap_end = off["started_at"].max().normalize()
    overlap_start = overlap_end - pd.Timedelta(days=90)

    def kyiv_city(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
        mask = (
            (df[region_col] == "Kyiv City")
            & (df["started_at"] >= overlap_start)
            & (df["started_at"] < overlap_end)
        )
        return df[mask]

    v = kyiv_city(vol, "region")
    o = kyiv_city(off, "oblast")

    def share_overlapping(a: pd.DataFrame, b: pd.DataFrame) -> float:
        hits = sum(
            ((b["started_at"] < r.finished_at) & (b["finished_at"] > r.started_at)).any()
            for r in a.itertuples()
        )
        return hits / len(a)

    def median_minutes(df: pd.DataFrame) -> float:
        return ((df["finished_at"] - df["started_at"]).dt.total_seconds() / 60).median()

    print(f"\nKyiv City, volunteer vs official, {overlap_start.date()} .. {overlap_end.date()}:")
    print(f"  alerts:            volunteer {len(v)}, official {len(o)}")
    print(f"  overlap the other: volunteer {share_overlapping(v, o):.0%}, official {share_overlapping(o, v):.0%}")
    print(f"  median duration:   volunteer {median_minutes(v):.1f} min, official {median_minutes(o):.1f} min")


def main() -> None:
    commit = fetch_latest_commit()
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, CSV_OUT)

    now = datetime.now(timezone.utc)
    df = load(CSV_OUT)

    print_overview(df, now)
    print_manual_check(df, now)
    compare_with_official(df)

    snapshot = {
        "source": RAW_URL,
        "commit_sha": commit["sha"],
        "commit_date": commit["committed_at"],
        "downloaded_at": now.isoformat(),
        "rows": int(len(df)),
        "last_started_at": df["started_at"].max().isoformat(),
    }
    SNAPSHOT_OUT.write_text(json.dumps(snapshot, indent=2))
    print(f"\nSnapshot saved to {SNAPSHOT_OUT}")


if __name__ == "__main__":
    main()