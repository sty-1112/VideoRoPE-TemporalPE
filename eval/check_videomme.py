import json
import os
import re
import sys
from typing import Dict, List

import numpy as np


def read_jsonl(file_path: str) -> List[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _resolve_item_uid(item: dict, fallback_idx: int) -> str:
    """
    为每条 Video-MME 结果生成稳定唯一标识。
    优先级：
      1) __sample_uid__
      2) question_id
      3) index / id / qid
      4) videoID / video_id + question
      5) fallback
    """
    for key in ["__sample_uid__", "question_id", "index", "id", "qid", "videoID", "video_id"]:
        value = item.get(key, None)
        if value is not None and str(value) != "":
            return str(value)

    video_key = str(item.get("videoID", item.get("video_id", "")))
    question = str(item.get("question", ""))
    if video_key or question:
        return f"{video_key}||{question}"

    return f"fallback_{fallback_idx}"


def extract_answer_letter(text: str) -> str:
    """
    从模型输出中抽取 A/B/C/D。
    """
    if text is None:
        return ""

    s = str(text).strip()

    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    # 如果太长且完全没有 A/B/C/D，直接判空
    if len(s.split()) > 10 and not re.search(r"[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def safe_mean(xs: List[float]) -> str:
    xs = [x for x in xs if x >= 0]
    if len(xs) == 0:
        return "nan"
    return f"{np.mean(xs):.2f}"


def main(eval_dir: str):
    path_root = eval_dir
    json_data = []

    for p in os.listdir(path_root):
        if re.match(r"^\d+\.json$", p):
            json_data += read_jsonl(os.path.join(path_root, p))

    dedup_data = []
    seen = set()
    for i, data in enumerate(json_data):
        uid = _resolve_item_uid(data, i)
        if uid in seen:
            continue
        seen.add(uid)
        dedup_data.append(data)

    # Video-MME 全量一般是 2700；如果不够，给 warning 但不直接崩
    if len(dedup_data) != 2700:
        print(f"[WARN] expected 2700 unique samples, but got {len(dedup_data)}")

    for data in dedup_data:
        pred_letter = extract_answer_letter(data.get("prediction", ""))
        gt_letter = str(data.get("answer", "")).strip()
        data["score"] = 1 if pred_letter == gt_letter else 0

    DURATIONS = [
        "short",
        "medium",
        "long",
    ]

    DOMAINS = [
        "Knowledge",
        "Film & Television",
        "Sports Competition",
        "Artistic Performance",
        "Life Record",
        "Multilingual",
    ]

    SUB_CATEGORIES = [
        "Humanity & History",
        "Literature & Art",
        "Biology & Medicine",
        "Finance & Commerce",
        "Astronomy",
        "Geography",
        "Law",
        "Life Tip",
        "Technology",
        "Animation",
        "Movie & TV Show",
        "Documentary",
        "News Report",
        "Esports",
        "Basketball",
        "Football",
        "Athletics",
        "Other Sports",
        "Stage Play",
        "Magic Show",
        "Variety Show",
        "Acrobatics",
        "Handicraft",
        "Food",
        "Fashion",
        "Daily Life",
        "Travel",
        "Pet & Animal",
        "Exercise",
        "Multilingual",
    ]

    TASK_CATEGORIES = [
        "Temporal Perception",
        "Spatial Perception",
        "Attribute Perception",
        "Action Recognition",
        "Object Recognition",
        "OCR Problems",
        "Counting Problem",
        "Temporal Reasoning",
        "Spatial Reasoning",
        "Action Reasoning",
        "Object Reasoning",
        "Information Synopsis",
    ]

    duration_rating = {}
    for duration in DURATIONS + ["overall"]:
        duration_rating[duration] = {
            "overall": "",
            "domain": {k: [] for k in DOMAINS},
            "sub_category": {k: [] for k in SUB_CATEGORIES},
            "task_type": {k: [] for k in TASK_CATEGORIES},
        }

    for data in dedup_data:
        domain = data.get("domain", "")
        sub_ctg = data.get("sub_category", "")
        task_ctg = data.get("task_type", "")
        duration = data.get("duration", "")
        score = data.get("score", -1)

        if duration in duration_rating:
            if domain in duration_rating[duration]["domain"]:
                duration_rating[duration]["domain"][domain].append(score)
            if sub_ctg in duration_rating[duration]["sub_category"]:
                duration_rating[duration]["sub_category"][sub_ctg].append(score)
            if task_ctg in duration_rating[duration]["task_type"]:
                duration_rating[duration]["task_type"][task_ctg].append(score)

        if domain in duration_rating["overall"]["domain"]:
            duration_rating["overall"]["domain"][domain].append(score)
        if sub_ctg in duration_rating["overall"]["sub_category"]:
            duration_rating["overall"]["sub_category"][sub_ctg].append(score)
        if task_ctg in duration_rating["overall"]["task_type"]:
            duration_rating["overall"]["task_type"][task_ctg].append(score)

    for duration in DURATIONS + ["overall"]:
        all_scores = sum(duration_rating[duration]["domain"].values(), [])
        duration_rating[duration]["overall"] = safe_mean(all_scores)

        for domain in DOMAINS:
            duration_rating[duration]["domain"][domain] = safe_mean(
                duration_rating[duration]["domain"][domain]
            )

        for sub_ctg in SUB_CATEGORIES:
            duration_rating[duration]["sub_category"][sub_ctg] = safe_mean(
                duration_rating[duration]["sub_category"][sub_ctg]
            )

        for task_ctg in TASK_CATEGORIES:
            duration_rating[duration]["task_type"][task_ctg] = safe_mean(
                duration_rating[duration]["task_type"][task_ctg]
            )

    # 控制台输出
    print(json.dumps(duration_rating, indent=2, ensure_ascii=False))
    print(
        duration_rating["short"]["overall"],
        duration_rating["medium"]["overall"],
        duration_rating["long"]["overall"],
        duration_rating["overall"]["overall"],
    )

    # 同时保存结果文件，方便脚本后续读取
    save_path = os.path.join(path_root, "results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(duration_rating, f, indent=2, ensure_ascii=False)
    print(f"[INFO] saved results to {save_path}")


if __name__ == "__main__":
    main(sys.argv[1])