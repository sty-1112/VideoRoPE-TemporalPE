#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build local success manifest from downloaded WebVid videos.")
    parser.add_argument("--download-root", type=str, required=True, help="Root dir of sequential download output")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--min-size-kb", type=int, default=50, help="Minimum video size in KB")
    return parser.parse_args()


def main():
    args = parse_args()

    download_root = Path(args.download_root)
    videos_dir = download_root / "videos"
    captions_dir = download_root / "captions"
    meta_dir = download_root / "meta"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not videos_dir.exists():
        raise FileNotFoundError(f"videos dir not found: {videos_dir}")

    mp4_files = sorted(videos_dir.glob("*.mp4"))
    rows = []

    for mp4_path in mp4_files:
        video_id = mp4_path.stem
        size_bytes = mp4_path.stat().st_size
        if size_bytes < args.min_size_kb * 1024:
            continue

        caption_path = captions_dir / f"{video_id}.txt"
        meta_path = meta_dir / f"{video_id}.json"

        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()

        url = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                url = str(meta.get("url", "")).strip()
                if not caption:
                    caption = str(meta.get("caption", "")).strip()
            except Exception:
                pass

        if not caption:
            # 没有 caption 的样本直接跳过，避免后面训练样本不完整
            continue

        rows.append(
            {
                "id": video_id,
                "video_path": str(mp4_path.resolve()),
                "caption": caption,
                "meta_path": str(meta_path.resolve()) if meta_path.exists() else "",
                "caption_path": str(caption_path.resolve()) if caption_path.exists() else "",
                "url": url,
                "size_bytes": size_bytes,
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "video_path",
                "caption",
                "meta_path",
                "caption_path",
                "url",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "download_root": str(download_root),
        "total_mp4_found": len(mp4_files),
        "kept_rows": len(rows),
        "out_csv": str(out_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()