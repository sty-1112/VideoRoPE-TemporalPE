#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Uniformly extract frames from local video manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="CSV from 03_build_local_success_manifest.py")
    parser.add_argument("--frames-root", type=str, required=True, help="Output root for extracted frames")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to extract per video")
    parser.add_argument("--max-items", type=int, default=None, help="Debug: only process first N videos")
    parser.add_argument("--resize-short", type=int, default=448, help="Resize shorter side to this value, keep aspect ratio")
    parser.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="Frame file extension")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already extracted videos")
    parser.add_argument("--report-dir", type=str, required=True, help="Directory for success/failed reports")
    return parser.parse_args()


def resize_keep_aspect(img, short_side):
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    frames_root = Path(args.frames_root)
    report_dir = Path(args.report_dir)

    frames_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.max_items is not None:
        rows = rows[: args.max_items]

    success_rows = []
    failed_rows = []

    for row in tqdm(rows, desc="Extracting frames", ncols=100):
        video_id = row["id"]
        video_path = Path(row["video_path"])
        out_dir = frames_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        expected_files = [out_dir / f"{i:04d}.{args.ext}" for i in range(args.num_frames)]
        if args.skip_existing and all(p.exists() for p in expected_files):
            success_rows.append(
                {
                    "id": video_id,
                    "video_path": str(video_path),
                    "frames_dir": str(out_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("cv2.VideoCapture failed to open video")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError(f"invalid total_frames={total_frames}")

            indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)

            saved = 0
            for i, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"failed reading frame_idx={frame_idx}")

                frame = resize_keep_aspect(frame, args.resize_short)

                out_path = out_dir / f"{i:04d}.{args.ext}"
                if args.ext == "jpg":
                    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(str(out_path), frame)
                saved += 1

            cap.release()

            if saved != args.num_frames:
                raise RuntimeError(f"saved={saved}, expected={args.num_frames}")

            success_rows.append(
                {
                    "id": video_id,
                    "video_path": str(video_path),
                    "frames_dir": str(out_dir),
                    "status": "ok",
                }
            )

        except Exception as e:
            failed_rows.append(
                {
                    "id": video_id,
                    "video_path": str(video_path),
                    "error": repr(e),
                }
            )

    success_csv = report_dir / "extract_success.csv"
    failed_csv = report_dir / "extract_failed.csv"
    summary_json = report_dir / "extract_summary.json"

    with success_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "video_path", "frames_dir", "status"])
        writer.writeheader()
        writer.writerows(success_rows)

    with failed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "video_path", "error"])
        writer.writeheader()
        writer.writerows(failed_rows)

    summary = {
        "manifest": str(manifest_path),
        "requested": len(rows),
        "success": len(success_rows),
        "failed": len(failed_rows),
        "frames_root": str(frames_root),
        "num_frames": args.num_frames,
        "ext": args.ext,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()