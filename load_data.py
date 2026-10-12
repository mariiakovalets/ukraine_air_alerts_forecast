"""Download the official air raid alert dataset, record its snapshot, and run freshness checks."""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
FILE_PATH = "datasets/official_data_en.csv"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_PATH}"
COMMITS_API = f"https://api.github.com/repos/{REPO}/commits?path={FILE_PATH}&per_page=1"

DATA_DIR = Path("data")
CSV_OUT = DATA_DIR / "official_data_en.csv"
SNAPSHOT_OUT = DATA_DIR / "snapshot.json"

MAX_STALENESS_HOURS = 48


def fetch_latest_commit() -> dict:
    with urllib.request.urlopen(COMMITS_API, timeout=30) as resp:
        commit = json.load(resp)[0]
    return {
        "sha": commit["sha"],
        "committed_at": commit["commit"]["committer"]["date"],
    }


def download_csv() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, CSV_OUT)


def main() -> None:
    commit = fetch_latest_commit()
    download_csv()

    df = pd.read_csv(CSV_OUT)
    print("Columns:", list(df.columns))
    print(df.head(), "\n")

    # Dataset README states all timestamps are UTC
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)

    now = datetime.now(timezone.utc)
    last_start = df["started_at"].max()
    staleness_h = (now - last_start).total_seconds() / 3600

    print(f"Rows: {len(df)}")
    print(f"Range: {df['started_at'].min()} -> {last_start} (UTC)")
    print(f"Latest alert started {staleness_h:.1f} h ago")
    if staleness_h > MAX_STALENESS_HOURS:
        print("WARNING: data looks stale")

    print("\nAlerts per month by level (last 15 months):")
    by_month = (
        df.assign(month=df["started_at"].dt.strftime("%Y-%m"))
        .groupby(["month", "level"]).size()
        .unstack(fill_value=0)
    )
    print(by_month.tail(15))

    snapshot = {
        "source": RAW_URL,
        "commit_sha": commit["sha"],
        "commit_date": commit["committed_at"],
        "downloaded_at": now.isoformat(),
        "rows": int(len(df)),
        "last_started_at": last_start.isoformat(),
    }
    SNAPSHOT_OUT.write_text(json.dumps(snapshot, indent=2))
    print(f"\nSnapshot saved to {SNAPSHOT_OUT}")


if __name__ == "__main__":
    main()