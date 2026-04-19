#!/usr/bin/env python3
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

CSV_ROOT = Path("videorope_exp/datasets/webvid/metadata_hf_full/data/train/partitions")
DB_PATH = Path("videorope_exp/datasets/webvid/metadata/webvid_url.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

csv_files = sorted(CSV_ROOT.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"No CSV files found under: {CSV_ROOT}")

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS videos")
cur.execute("""
CREATE TABLE videos (
    videoid TEXT,
    contentUrl TEXT,
    duration TEXT,
    page_dir TEXT,
    name TEXT
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_videoid ON videos(videoid)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contentUrl ON videos(contentUrl)")

insert_sql = "INSERT INTO videos (videoid, contentUrl, duration, page_dir, name) VALUES (?, ?, ?, ?, ?)"

total = 0
for i, csv_file in enumerate(csv_files, start=1):
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append((
                (row.get("videoid") or "").strip(),
                (row.get("contentUrl") or "").strip(),
                (row.get("duration") or "").strip(),
                (row.get("page_dir") or "").strip(),
                (row.get("name") or "").strip(),
            ))
            if len(rows) >= 10000:
                cur.executemany(insert_sql, rows)
                conn.commit()
                total += len(rows)
                rows = []
        if rows:
            cur.executemany(insert_sql, rows)
            conn.commit()
            total += len(rows)

    print(f"[{i}/{len(csv_files)}] imported {csv_file.name} | total rows = {total}")

conn.close()
print(f"Done. SQLite DB saved to: {DB_PATH}")
