#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import time
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential video downloader for WebVid manifests.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to download_manifest.csv")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per sample")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between samples")
    parser.add_argument("--max-items", type=int, default=None, help="Only download first N items")
    parser.add_argument("--start-index", type=int, default=0, help="Start row index")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if mp4 already exists")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification. Only use if many SSL EOF errors persist.",
    )
    return parser.parse_args()


def build_session(retries: int, insecure: bool) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=0,  # we handle retries ourselves per sample
        connect=0,
        read=0,
        redirect=0,
        status=0,
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Connection": "close",
        }
    )

    if insecure:
        requests.packages.urllib3.disable_warnings()  # type: ignore
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

    return session


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_one(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout: int,
    insecure: bool,
):
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    with session.get(url, stream=True, timeout=timeout, verify=not insecure) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError(f"Unexpected HTML response instead of video: {content_type}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, out_path)


def main():
    args = parse_args()

    manifest = Path(args.manifest)
    outdir = Path(args.outdir)
    videos_dir = outdir / "videos"
    meta_dir = outdir / "meta"
    captions_dir = outdir / "captions"
    logs_dir = outdir / "logs"

    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    success_path = logs_dir / "success.csv"
    failed_path = logs_dir / "failed.csv"

    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = rows[args.start_index :]
    if args.max_items is not None:
        rows = rows[: args.max_items]

    session = build_session(args.retries, args.insecure)

    success_rows: List[Dict] = []
    failed_rows: List[Dict] = []

    for idx, row in enumerate(tqdm(rows, desc="Downloading", ncols=100), start=args.start_index):
        vid = str(row.get("id", "")).strip()
        url = str(row.get("url", "")).strip()
        caption = str(row.get("caption", "")).strip()

        if not vid or not url:
            failed_rows.append(
                {
                    "index": idx,
                    "id": vid,
                    "url": url,
                    "error": "missing id or url",
                }
            )
            continue

        out_mp4 = videos_dir / f"{vid}.mp4"
        out_json = meta_dir / f"{vid}.json"
        out_txt = captions_dir / f"{vid}.txt"

        if args.skip_existing and out_mp4.exists():
            success_rows.append(
                {
                    "index": idx,
                    "id": vid,
                    "url": url,
                    "file": str(out_mp4),
                    "status": "skipped_existing",
                }
            )
            continue

        last_err = None
        ok = False

        for attempt in range(1, args.retries + 1):
            try:
                download_one(
                    session=session,
                    url=url,
                    out_path=out_mp4,
                    timeout=args.timeout,
                    insecure=args.insecure,
                )

                safe_write_text(out_txt, caption)
                safe_write_json(
                    out_json,
                    {
                        "id": vid,
                        "url": url,
                        "caption": caption,
                        "file": str(out_mp4),
                    },
                )

                success_rows.append(
                    {
                        "index": idx,
                        "id": vid,
                        "url": url,
                        "file": str(out_mp4),
                        "status": "downloaded",
                    }
                )
                ok = True
                break

            except Exception as e:
                last_err = repr(e)
                if out_mp4.exists():
                    try:
                        out_mp4.unlink()
                    except Exception:
                        pass
                partial = out_mp4.with_suffix(".mp4.partial")
                if partial.exists():
                    try:
                        partial.unlink()
                    except Exception:
                        pass
                time.sleep(1.0)

        if not ok:
            failed_rows.append(
                {
                    "index": idx,
                    "id": vid,
                    "url": url,
                    "error": last_err,
                }
            )

        time.sleep(args.sleep)

    with open(success_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "id", "url", "file", "status"])
        writer.writeheader()
        writer.writerows(success_rows)

    with open(failed_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "id", "url", "error"])
        writer.writeheader()
        writer.writerows(failed_rows)

    summary = {
        "manifest": str(manifest),
        "outdir": str(outdir),
        "requested": len(rows),
        "success": len(success_rows),
        "failed": len(failed_rows),
        "insecure": args.insecure,
    }
    safe_write_json(logs_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()