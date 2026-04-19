#!/usr/bin/env python3
"""Sample a fixed-size subset from a WebVid-style SQLite database.

This script is designed for large tables and uses streaming reservoir sampling,
so it does not need to load the whole table into memory.

Outputs:
- manifest.csv           : canonical sampled metadata
- ids.txt                : one id per line
- captions.jsonl         : id/caption pairs
- summary.json           : counts and URL diagnostics
- download_manifest.csv  : only written when a URL column is available

Example:
  python 01_sample_webvid_subset.py \
      --db /path/to/webvid.db \
      --table videos \
      --size 10000 \
      --seed 42 \
      --outdir /path/to/webvid_10k_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

ID_CANDIDATES = ["id", "video_id", "videoid", "filename"]
CAPTION_CANDIDATES = ["text", "caption", "title", "description", "name"]
URL_CANDIDATES = [
    "url",
    "video_url",
    "contentUrl",
    "content_url",
    "download_url",
    "video",
    "video_path",
    "filepath",
]


@dataclass
class SampleRow:
    sample_id: str
    caption: str
    url: str
    source_table: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a WebVid subset from SQLite metadata.")
    parser.add_argument("--db", type=Path, required=True, help="Path to webvid.db")
    parser.add_argument("--table", type=str, default="videos", help="Table name")
    parser.add_argument("--size", type=int, required=True, help="Target subset size, e.g. 10000")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--id-col", type=str, default=None, help="Override ID column")
    parser.add_argument("--caption-col", type=str, default=None, help="Override caption column")
    parser.add_argument("--url-col", type=str, default=None, help="Override URL column")
    parser.add_argument(
        "--require-url",
        action="store_true",
        help="Only keep rows whose URL is non-empty. Use this when a valid URL column exists.",
    )
    parser.add_argument("--log-every", type=int, default=500000, help="Progress log interval")
    return parser.parse_args()


def first_present(candidates: Sequence[str], columns: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\r\n", " ").replace("\n", " ").strip()


def is_http_like(value: str) -> bool:
    v = value.lower().strip()
    return v.startswith("http://") or v.startswith("https://")


def iter_rows(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    caption_col: str,
    url_col: Optional[str],
) -> Iterator[SampleRow]:
    columns = [id_col, caption_col] + ([url_col] if url_col else [])
    query = f"SELECT {', '.join(columns)} FROM {table}"
    cursor = conn.execute(query)
    for row in cursor:
        raw_id = row[0]
        raw_caption = row[1]
        raw_url = row[2] if url_col else ""

        sample_id = normalize_text(raw_id)
        caption = normalize_text(raw_caption)
        url = normalize_text(raw_url)

        if not sample_id or not caption:
            continue
        yield SampleRow(sample_id=sample_id, caption=caption, url=url, source_table=table)


def reservoir_sample(stream: Iterable[SampleRow], k: int, seed: int, log_every: int) -> tuple[List[SampleRow], Dict[str, int]]:
    rng = random.Random(seed)
    reservoir: List[SampleRow] = []
    seen = 0
    kept_with_url = 0
    kept_http_like = 0

    for item in stream:
        seen += 1
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(1, seen)
            if j <= k:
                reservoir[j - 1] = item

        if item.url:
            kept_with_url += 1
            if is_http_like(item.url):
                kept_http_like += 1

        if log_every > 0 and seen % log_every == 0:
            print(f"[progress] scanned {seen:,} eligible rows...")

    stats = {
        "eligible_rows_scanned": seen,
        "eligible_rows_with_nonempty_url": kept_with_url,
        "eligible_rows_with_http_like_url": kept_http_like,
    }
    return reservoir, stats


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.db))
    try:
        columns = get_columns(conn, args.table)
        id_col = args.id_col or first_present(ID_CANDIDATES, columns)
        caption_col = args.caption_col or first_present(CAPTION_CANDIDATES, columns)
        url_col = args.url_col or first_present(URL_CANDIDATES, columns)

        if not id_col:
            raise ValueError(f"Could not detect an ID column. Available columns: {columns}")
        if not caption_col:
            raise ValueError(f"Could not detect a caption column. Available columns: {columns}")

        print("=" * 80)
        print(f"DB: {args.db}")
        print(f"Table: {args.table}")
        print(f"Detected columns -> id: {id_col}, caption: {caption_col}, url: {url_col}")
        print(f"Target subset size: {args.size:,}")
        print(f"Seed: {args.seed}")
        print(f"Output dir: {args.outdir}")
        print("=" * 80)

        stream = iter_rows(conn, args.table, id_col, caption_col, url_col)
        if args.require_url:
            stream = (row for row in stream if row.url)

        sample, stats = reservoir_sample(stream, args.size, args.seed, args.log_every)
        if len(sample) < args.size:
            print(
                f"[warning] only sampled {len(sample):,} rows, fewer than requested {args.size:,}. "
                "This usually means the eligible pool is smaller than expected."
            )

        # Sort by ID for stable diff/readability.
        sample = sorted(sample, key=lambda x: x.sample_id)

        manifest_path = args.outdir / "manifest.csv"
        ids_path = args.outdir / "ids.txt"
        captions_path = args.outdir / "captions.jsonl"
        summary_path = args.outdir / "summary.json"
        download_manifest_path = args.outdir / "download_manifest.csv"

        url_nonempty_in_sample = sum(1 for row in sample if row.url)
        url_http_like_in_sample = sum(1 for row in sample if is_http_like(row.url))

        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "caption", "url", "source_table", "seed"],
            )
            writer.writeheader()
            for row in sample:
                writer.writerow(
                    {
                        "id": row.sample_id,
                        "caption": row.caption,
                        "url": row.url,
                        "source_table": row.source_table,
                        "seed": args.seed,
                    }
                )

        with ids_path.open("w", encoding="utf-8") as f:
            for row in sample:
                f.write(f"{row.sample_id}\n")

        with captions_path.open("w", encoding="utf-8") as f:
            for row in sample:
                f.write(json.dumps({"id": row.sample_id, "caption": row.caption}, ensure_ascii=False) + "\n")

        wrote_download_manifest = False
        if url_col is not None and url_nonempty_in_sample > 0:
            with download_manifest_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["url", "caption", "id"])
                writer.writeheader()
                for row in sample:
                    if row.url:
                        writer.writerow({"url": row.url, "caption": row.caption, "id": row.sample_id})
            wrote_download_manifest = True

        summary = {
            "db": str(args.db),
            "table": args.table,
            "requested_size": args.size,
            "actual_size": len(sample),
            "seed": args.seed,
            "detected_columns": {
                "id_col": id_col,
                "caption_col": caption_col,
                "url_col": url_col,
            },
            "scan_stats": stats,
            "sample_url_stats": {
                "url_nonempty_in_sample": url_nonempty_in_sample,
                "url_http_like_in_sample": url_http_like_in_sample,
            },
            "files": {
                "manifest": str(manifest_path),
                "ids": str(ids_path),
                "captions": str(captions_path),
                "summary": str(summary_path),
                "download_manifest": str(download_manifest_path) if wrote_download_manifest else None,
            },
            "can_directly_download": bool(url_col is not None and url_http_like_in_sample > 0),
            "notes": (
                "If can_directly_download is false, your current SQLite metadata is not sufficient by itself "
                "for video downloading. You need an additional metadata source that includes usable URLs, "
                "then merge it by id before download."
            ),
        }

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Saved: {manifest_path}")
        print(f"Saved: {ids_path}")
        print(f"Saved: {captions_path}")
        print(f"Saved: {summary_path}")
        if wrote_download_manifest:
            print(f"Saved: {download_manifest_path}")
        print("-" * 80)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("=" * 80)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
