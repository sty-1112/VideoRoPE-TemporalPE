#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VideoWeave manifests to LLaMA-Factory ShareGPT multimodal image datasets."
    )
    parser.add_argument(
        "--single-manifest",
        type=str,
        required=True,
        help="Path to single_video_16f_manifest.jsonl",
    )
    parser.add_argument(
        "--videoweave-manifest",
        type=str,
        required=True,
        help="Path to videoweave_random_l2_f8_manifest.jsonl",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Output dataset_dir for LLaMA-Factory, e.g. videorope_exp/datasets/llamafactory_webvid",
    )
    parser.add_argument(
        "--single-name",
        type=str,
        default="webvid_single_16f_sharegpt",
        help="Dataset name for single-video setting",
    )
    parser.add_argument(
        "--videoweave-name",
        type=str,
        default="webvid_videoweave_l2_f8_sharegpt",
        help="Dataset name for VideoWeave setting",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=512,
        help="Create pilot subsets of this size for each dataset; set <=0 to disable",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pilot subset sampling",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_human_message(num_images: int, prompt_text: str) -> str:
    # LLaMA-Factory要求 images 数量与 <image> 标记数量一致
    prefix = "\n".join(["<image>"] * num_images)
    if prompt_text:
        return f"{prefix}\n{prompt_text}"
    return prefix


def convert_manifest_rows(rows: List[Dict]) -> List[Dict]:
    dataset = []
    for row in rows:
        frame_paths = row["frame_paths"]
        target_text = row["target_text"].strip()
        prompt_text = row.get("prompt_text", "Describe what is happening in the video.").strip()

        if not frame_paths or not target_text:
            continue

        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": build_human_message(len(frame_paths), prompt_text),
                },
                {
                    "from": "gpt",
                    "value": target_text,
                },
            ],
            "images": frame_paths,
            # 下面这些字段只是为了你自己后续排查方便，LLaMA-Factory会忽略未映射列
            "sample_id": row.get("sample_id", ""),
            "mode": row.get("mode", ""),
            "source_video_ids": row.get("source_video_ids", []),
            "captions": row.get("captions", []),
            "target_text": target_text,
        }
        dataset.append(item)
    return dataset


def sample_pilot(rows: List[Dict], pilot_size: int, seed: int) -> List[Dict]:
    if pilot_size <= 0 or len(rows) <= pilot_size:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = sorted(idx[:pilot_size])
    return [rows[i] for i in idx]


def write_json(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    single_manifest = Path(args.single_manifest)
    videoweave_manifest = Path(args.videoweave_manifest)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    single_rows = load_jsonl(single_manifest)
    videoweave_rows = load_jsonl(videoweave_manifest)

    single_dataset = convert_manifest_rows(single_rows)
    videoweave_dataset = convert_manifest_rows(videoweave_rows)

    single_file = f"{args.single_name}.json"
    videoweave_file = f"{args.videoweave_name}.json"

    write_json(dataset_dir / single_file, single_dataset)
    write_json(dataset_dir / videoweave_file, videoweave_dataset)

    dataset_info = {
        args.single_name: {
            "file_name": single_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
        args.videoweave_name: {
            "file_name": videoweave_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
    }

    if args.pilot_size > 0:
        single_pilot = sample_pilot(single_dataset, args.pilot_size, args.seed)
        videoweave_pilot = sample_pilot(videoweave_dataset, args.pilot_size, args.seed)

        single_pilot_name = f"{args.single_name}_pilot{args.pilot_size}"
        videoweave_pilot_name = f"{args.videoweave_name}_pilot{args.pilot_size}"

        single_pilot_file = f"{single_pilot_name}.json"
        videoweave_pilot_file = f"{videoweave_pilot_name}.json"

        write_json(dataset_dir / single_pilot_file, single_pilot)
        write_json(dataset_dir / videoweave_pilot_file, videoweave_pilot)

        dataset_info[single_pilot_name] = {
            "file_name": single_pilot_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        }
        dataset_info[videoweave_pilot_name] = {
            "file_name": videoweave_pilot_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        }

    dataset_info_path = dataset_dir / "dataset_info.json"
    write_json(dataset_info_path, dataset_info)

    summary = {
        "single_manifest": str(single_manifest),
        "videoweave_manifest": str(videoweave_manifest),
        "dataset_dir": str(dataset_dir),
        "single_dataset_name": args.single_name,
        "videoweave_dataset_name": args.videoweave_name,
        "single_samples": len(single_dataset),
        "videoweave_samples": len(videoweave_dataset),
        "pilot_size": args.pilot_size,
        "dataset_info_path": str(dataset_info_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()