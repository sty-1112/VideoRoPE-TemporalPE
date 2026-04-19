#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Render manifest rows into mp4 videos.")
    parser.add_argument("--single-manifest", type=str, required=True)
    parser.add_argument("--videoweave-manifest", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-items-single", type=int, default=None)
    parser.add_argument("--max-items-vw", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_first_frame_shape(frame_paths: List[str]):
    img = cv2.imread(frame_paths[0])
    if img is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = img.shape[:2]
    return w, h


def render_one_video(frame_paths: List[str], out_path: Path, fps: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = read_first_frame_shape(frame_paths)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    try:
        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            ih, iw = img.shape[:2]
            if (iw, ih) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


def process_manifest(rows: List[Dict], out_dir: Path, fps: float, skip_existing: bool) -> List[Dict]:
    rendered = []

    for row in tqdm(rows, desc=f"Rendering -> {out_dir.name}", ncols=100):
        sample_id = row["sample_id"]
        out_video = out_dir / f"{sample_id}.mp4"

        if skip_existing and out_video.exists():
            rendered.append(
                {
                    "sample_id": sample_id,
                    "mode": row["mode"],
                    "video_path": str(out_video.resolve()),
                    "source_video_ids": row["source_video_ids"],
                    "captions": row["captions"],
                    "target_text": row["target_text"],
                    "prompt_text": row.get("prompt_text", "Describe what is happening in the video."),
                    "num_frames": len(row["frame_paths"]),
                    "status": "skipped_existing",
                }
            )
            continue

        render_one_video(row["frame_paths"], out_video, fps)

        rendered.append(
            {
                "sample_id": sample_id,
                "mode": row["mode"],
                "video_path": str(out_video.resolve()),
                "source_video_ids": row["source_video_ids"],
                "captions": row["captions"],
                "target_text": row["target_text"],
                "prompt_text": row.get("prompt_text", "Describe what is happening in the video."),
                "num_frames": len(row["frame_paths"]),
                "status": "ok",
            }
        )

    return rendered


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    single_rows = load_jsonl(Path(args.single_manifest))
    vw_rows = load_jsonl(Path(args.videoweave_manifest))

    if args.max_items_single is not None:
        single_rows = single_rows[: args.max_items_single]
    if args.max_items_vw is not None:
        vw_rows = vw_rows[: args.max_items_vw]

    out_root = Path(args.out_root)
    single_video_dir = out_root / "single_video_16f" / "videos"
    vw_video_dir = out_root / "videoweave_random_l2_f8" / "videos"

    single_rendered = process_manifest(
        single_rows, single_video_dir, args.fps, args.skip_existing
    )
    vw_rendered = process_manifest(
        vw_rows, vw_video_dir, args.fps, args.skip_existing
    )

    single_manifest_out = out_root / "single_video_16f" / "rendered_manifest.jsonl"
    vw_manifest_out = out_root / "videoweave_random_l2_f8" / "rendered_manifest.jsonl"
    summary_out = out_root / "render_summary.json"

    write_jsonl(single_manifest_out, single_rendered)
    write_jsonl(vw_manifest_out, vw_rendered)

    summary = {
        "single_input": args.single_manifest,
        "videoweave_input": args.videoweave_manifest,
        "out_root": str(out_root),
        "single_rendered": len(single_rendered),
        "videoweave_rendered": len(vw_rendered),
        "fps": args.fps,
        "single_manifest_out": str(single_manifest_out),
        "videoweave_manifest_out": str(vw_manifest_out),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()