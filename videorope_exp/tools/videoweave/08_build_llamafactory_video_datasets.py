#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Build LLaMA-Factory ShareGPT video datasets.")
    parser.add_argument("--single-rendered", type=str, required=True)
    parser.add_argument("--videoweave-rendered", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--single-name", type=str, default="webvid_single_16f_video_sharegpt")
    parser.add_argument("--videoweave-name", type=str, default="webvid_videoweave_l2_f8_video_sharegpt")
    parser.add_argument("--pilot-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_rows(rows: List[Dict]) -> List[Dict]:
    data = []
    for row in rows:
        video_path = row["video_path"]
        target_text = row["target_text"].strip()
        prompt_text = row.get("prompt_text", "Describe what is happening in the video.").strip()

        item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{prompt_text}",
                },
                {
                    "from": "gpt",
                    "value": target_text,
                },
            ],
            "videos": [video_path],
            "sample_id": row.get("sample_id", ""),
            "mode": row.get("mode", ""),
            "source_video_ids": row.get("source_video_ids", []),
            "captions": row.get("captions", []),
            "target_text": target_text,
        }
        data.append(item)
    return data


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

    single_rows = load_jsonl(Path(args.single_rendered))
    vw_rows = load_jsonl(Path(args.videoweave_rendered))

    single_data = convert_rows(single_rows)
    vw_data = convert_rows(vw_rows)

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    single_file = f"{args.single_name}.json"
    vw_file = f"{args.videoweave_name}.json"

    write_json(dataset_dir / single_file, single_data)
    write_json(dataset_dir / vw_file, vw_data)

    dataset_info = {
        args.single_name: {
            "file_name": single_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "videos": "videos",
            },
        },
        args.videoweave_name: {
            "file_name": vw_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "videos": "videos",
            },
        },
    }

    if args.pilot_size > 0:
        single_pilot = sample_pilot(single_data, args.pilot_size, args.seed)
        vw_pilot = sample_pilot(vw_data, args.pilot_size, args.seed)

        single_pilot_name = f"{args.single_name}_pilot{args.pilot_size}"
        vw_pilot_name = f"{args.videoweave_name}_pilot{args.pilot_size}"

        write_json(dataset_dir / f"{single_pilot_name}.json", single_pilot)
        write_json(dataset_dir / f"{vw_pilot_name}.json", vw_pilot)

        dataset_info[single_pilot_name] = {
            "file_name": f"{single_pilot_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "videos": "videos",
            },
        }
        dataset_info[vw_pilot_name] = {
            "file_name": f"{vw_pilot_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "videos": "videos",
            },
        }

    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset_dir": str(dataset_dir),
        "single_samples": len(single_data),
        "videoweave_samples": len(vw_data),
        "pilot_size": args.pilot_size,
        "dataset_info_path": str(dataset_info_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()