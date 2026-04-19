#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build single-video and VideoWeave-style random-splicing manifests."
    )
    parser.add_argument(
        "--local-manifest",
        type=str,
        required=True,
        help="CSV produced by 03_build_local_success_manifest.py",
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        required=True,
        help="Root dir of extracted frames, e.g. videorope_exp/datasets/webvid/frames/webvid_10k_seed42_url_16f",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for generated manifests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and pairing",
    )
    parser.add_argument(
        "--single-frames",
        type=int,
        default=16,
        help="Frames per sample for the single-video setting",
    )
    parser.add_argument(
        "--videos-per-sample",
        type=int,
        default=2,
        help="L in VideoWeave; currently recommended to use 2",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Frames taken from each video for the VideoWeave setting",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Extracted frame extension",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional debug limit on number of source videos used",
    )
    return parser.parse_args()


def even_indices(n: int, k: int) -> List[int]:
    if k > n:
        raise ValueError(f"k={k} cannot be larger than n={n}")
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_frame_paths(frames_root: Path, video_id: str, ext: str, expected_n: int) -> List[Path]:
    frame_dir = frames_root / video_id
    if not frame_dir.exists():
        return []
    frames = sorted(frame_dir.glob(f"*.{ext}"))
    if len(frames) < expected_n:
        return []
    return frames


def build_single_manifest(
    rows: List[Dict[str, str]],
    frames_root: Path,
    ext: str,
    single_frames: int,
) -> List[Dict]:
    manifest = []
    for idx, row in enumerate(rows):
        video_id = row["id"]
        caption = row["caption"].strip()
        frame_paths = get_frame_paths(frames_root, video_id, ext, single_frames)
        if len(frame_paths) < single_frames:
            continue

        sample = {
            "sample_id": f"single_{idx:07d}",
            "mode": "single_video_16f",
            "source_video_ids": [video_id],
            "frame_paths": [str(p.resolve()) for p in frame_paths[:single_frames]],
            "captions": [caption],
            "target_text": caption,
            "prompt_text": "Describe what is happening in the video.",
        }
        manifest.append(sample)
    return manifest


def build_videoweave_manifest(
    rows: List[Dict[str, str]],
    frames_root: Path,
    ext: str,
    videos_per_sample: int,
    frames_per_video: int,
    seed: int,
) -> List[Dict]:
    if videos_per_sample != 2:
        raise NotImplementedError("This first implementation supports videos_per_sample=2 only.")

    eligible = []
    need_n = max(16, frames_per_video)
    for row in rows:
        video_id = row["id"]
        caption = row["caption"].strip()
        frame_paths = get_frame_paths(frames_root, video_id, ext, need_n)
        if len(frame_paths) < need_n:
            continue
        eligible.append(
            {
                "id": video_id,
                "caption": caption,
                "frame_paths": frame_paths,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(eligible)

    # 两两配对；如果是奇数个，最后一个丢弃
    pair_count = len(eligible) // 2
    selected_indices = even_indices(16, frames_per_video)

    manifest = []
    for i in range(pair_count):
        a = eligible[2 * i]
        b = eligible[2 * i + 1]

        a_frames = [a["frame_paths"][j] for j in selected_indices]
        b_frames = [b["frame_paths"][j] for j in selected_indices]

        # VideoWeave 核心：固定总帧数下，把两个短视频拼起来；caption 直接拼接
        target_text = f'{a["caption"]} {b["caption"]}'.strip()

        sample = {
            "sample_id": f"vw_l2f8_{i:07d}",
            "mode": "videoweave_random_l2_f8",
            "source_video_ids": [a["id"], b["id"]],
            "frame_paths": [str(p.resolve()) for p in (a_frames + b_frames)],
            "captions": [a["caption"], b["caption"]],
            "target_text": target_text,
            "prompt_text": "Describe what is happening in the video.",
        }
        manifest.append(sample)

    return manifest


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    local_manifest = Path(args.local_manifest)
    frames_root = Path(args.frames_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(local_manifest)
    if args.max_videos is not None:
        rows = rows[: args.max_videos]

    single_manifest = build_single_manifest(
        rows=rows,
        frames_root=frames_root,
        ext=args.image_ext,
        single_frames=args.single_frames,
    )

    videoweave_manifest = build_videoweave_manifest(
        rows=rows,
        frames_root=frames_root,
        ext=args.image_ext,
        videos_per_sample=args.videos_per_sample,
        frames_per_video=args.frames_per_video,
        seed=args.seed,
    )

    single_path = outdir / "single_video_16f_manifest.jsonl"
    videoweave_path = outdir / "videoweave_random_l2_f8_manifest.jsonl"
    summary_path = outdir / "build_summary.json"

    write_jsonl(single_path, single_manifest)
    write_jsonl(videoweave_path, videoweave_manifest)

    summary = {
        "local_manifest": str(local_manifest),
        "frames_root": str(frames_root),
        "source_videos_seen": len(rows),
        "single_video_samples": len(single_manifest),
        "videoweave_random_l2_f8_samples": len(videoweave_manifest),
        "single_output": str(single_path),
        "videoweave_output": str(videoweave_path),
        "seed": args.seed,
        "single_frames": args.single_frames,
        "videos_per_sample": args.videos_per_sample,
        "frames_per_video": args.frames_per_video,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()