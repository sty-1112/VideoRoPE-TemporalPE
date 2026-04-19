#!/usr/bin/env python3
"""Inspect a WebVid-style SQLite metadata database.

Usage:
  python 00_inspect_webvid_db.py \
      --db /path/to/webvid.db \
      --table videos \
      --limit 10 \
      --report /path/to/inspect_report.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a WebVid SQLite DB.")
    parser.add_argument("--db", type=Path, required=True, help="Path to webvid.db")
    parser.add_argument("--table", type=str, default=None, help="Table name to inspect. Default: auto-detect.")
    parser.add_argument("--limit", type=int, default=10, help="How many example rows to print.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to save a JSON report.",
    )
    return parser.parse_args()


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def pick_table(tables: Sequence[str], requested: Optional[str]) -> str:
    if requested:
        if requested not in tables:
            raise ValueError(f"Requested table '{requested}' not found. Available tables: {tables}")
        return requested
    if "videos" in tables:
        return "videos"
    if not tables:
        raise ValueError("No user tables found in the SQLite database.")
    return tables[0]


def first_present(candidates: Sequence[str], columns: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def count_nonempty(conn: sqlite3.Connection, table: str, column: str) -> int:
    query = f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) != ''"
    return int(conn.execute(query).fetchone()[0])


def count_http_like(conn: sqlite3.Connection, table: str, column: str) -> int:
    query = (
        f"SELECT COUNT(*) FROM {table} "
        f"WHERE {column} IS NOT NULL "
        f"AND (LOWER(CAST({column} AS TEXT)) LIKE 'http://%' OR LOWER(CAST({column} AS TEXT)) LIKE 'https://%')"
    )
    return int(conn.execute(query).fetchone()[0])


def fetch_examples(conn: sqlite3.Connection, table: str, columns: Sequence[str], limit: int) -> List[Dict[str, object]]:
    chosen = ", ".join(columns)
    rows = conn.execute(f"SELECT {chosen} FROM {table} LIMIT ?", (limit,)).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        tables = list_tables(conn)
        table = pick_table(tables, args.table)
        columns = get_columns(conn, table)
        total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        id_col = first_present(ID_CANDIDATES, columns)
        caption_col = first_present(CAPTION_CANDIDATES, columns)
        url_col = first_present(URL_CANDIDATES, columns)

        report = {
            "db_path": str(args.db),
            "tables": tables,
            "selected_table": table,
            "columns": columns,
            "row_count": total_rows,
            "detected_columns": {
                "id_col": id_col,
                "caption_col": caption_col,
                "url_col": url_col,
            },
            "column_stats": {},
            "examples": [],
        }

        print("=" * 80)
        print(f"DB: {args.db}")
        print(f"Tables: {tables}")
        print(f"Selected table: {table}")
        print(f"Row count: {total_rows:,}")
        print(f"Columns ({len(columns)}): {columns}")
        print("-" * 80)
        print(f"Detected id column: {id_col}")
        print(f"Detected caption column: {caption_col}")
        print(f"Detected url column: {url_col}")

        for column in [c for c in [id_col, caption_col, url_col] if c is not None]:
            nonempty = count_nonempty(conn, table, column)
            http_like = count_http_like(conn, table, column)
            report["column_stats"][column] = {
                "nonempty": nonempty,
                "http_like": http_like,
            }
            print("-" * 80)
            print(f"Column stats for '{column}':")
            print(f"  non-empty rows: {nonempty:,}")
            print(f"  http-like rows: {http_like:,}")

        example_cols = [c for c in [id_col, caption_col, url_col] if c is not None]
        if not example_cols:
            example_cols = columns[: min(len(columns), 5)]
        examples = fetch_examples(conn, table, example_cols, args.limit)
        report["examples"] = examples

        print("-" * 80)
        print(f"First {len(examples)} example rows:")
        for idx, row in enumerate(examples, start=1):
            print(f"[{idx}] {row}")
        print("=" * 80)

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved report to: {args.report}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
