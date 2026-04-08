import importlib.util
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Dynamically override transformers.models.qwen2_vl.modeling_qwen2_vl
# with VideoRoPE's local implementation, without editing site-packages.
# -----------------------------------------------------------------------------
videorope_model_path = (
    Path(__file__).resolve().parents[1]
    / "videorope-transformer"
    / "modeling_videorope.py"
)

target_name = "transformers.models.qwen2_vl.modeling_qwen2_vl"
spec = importlib.util.spec_from_file_location(target_name, videorope_model_path)
module = importlib.util.module_from_spec(spec)
sys.modules[target_name] = module
spec.loader.exec_module(module)

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import random
import signal
import time
import traceback
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# -----------------------------------------------------------------------------
# Multiprocessing start method
# -----------------------------------------------------------------------------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# -----------------------------------------------------------------------------
# Timeout utilities
# -----------------------------------------------------------------------------
class SampleTimeoutError(TimeoutError):
    pass


def _alarm_handler(signum, frame):
    raise SampleTimeoutError("sample processing timed out")


HAS_SIGALRM = hasattr(signal, "SIGALRM")


def start_timeout(timeout_sec: int):
    if HAS_SIGALRM and timeout_sec and timeout_sec > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_sec)


def cancel_timeout():
    if HAS_SIGALRM:
        signal.alarm(0)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def save_jsonl_line(file_path: str, data: dict):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str) -> List[dict]:
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def read_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tsv_to_json(file_path: str):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append(row)
    return eval(json.dumps(data, indent=4))


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    random.seed(233)
    lst = copy.deepcopy(lst)
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def qa_template(data):
    question = f"Question: {data['question']}\n"
    question += "Options:\n"

    answer = data["candidates"][data["correct_choice"]]
    answer_idx = -1

    for idx, c in enumerate(data["candidates"]):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx

    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, answer


def get_item_key(item: dict) -> str:
    if "id" in item:
        return str(item["id"])
    return item["video_id"] + item["question"]


def parse_skip_video_ids(skip_video_ids: str) -> Set[str]:
    if not skip_video_ids:
        return set()
    return {x.strip() for x in skip_video_ids.split(",") if x.strip()}


def get_video_shape_str(video_inputs) -> str:
    try:
        if video_inputs is None:
            return "None"
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) > 0:
            x = video_inputs[0]
            if hasattr(x, "shape"):
                return str(tuple(x.shape))
        if hasattr(video_inputs, "shape"):
            return str(tuple(video_inputs.shape))
    except Exception:
        pass
    return "unknown"


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def eval_dataset(args):
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels
    context_length = args.context_length
    model_path = os.path.expanduser(args.model_path)

    eval_dataset_json = os.path.join(
        args.chat_conversation_output_folder, f"{str(args.chunk_idx)}.json"
    )
    error_json = os.path.join(
        args.chat_conversation_output_folder, f"{str(args.chunk_idx)}_errors.jsonl"
    )
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)

    # 已完成样本
    done_keys = set()
    if os.path.exists(eval_dataset_json):
        for item in read_jsonl(eval_dataset_json):
            done_keys.add(get_item_key(item))

    # 已失败样本
    failed_keys = set()
    if args.skip_failed_in_errors and os.path.exists(error_json):
        for item in read_jsonl(error_json):
            failed_keys.add(get_item_key(item))

    # 额外跳过视频
    skip_video_ids = parse_skip_video_ids(args.skip_video_ids)

    # -------------------------------------------------------------------------
    # model / processor
    # -------------------------------------------------------------------------
    llm = None
    sampling_params = None
    model = None

    total_pixels = int((context_length - 512) * 28 * 28)

    if context_length >= 48000:
        # lazy import vllm only when needed
        from vllm import LLM, SamplingParams

        llm = LLM(
            model_path,
            max_model_len=int(context_length) + 1536,
            limit_mm_per_prompt={"video": 10},
            gpu_memory_utilization=0.8,
            disable_log_stats=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            presence_penalty=0,
            frequency_penalty=0,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.to("cuda")
        model = model.eval()

    if args.nframes is None:
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            total_pixels=total_pixels,
            use_fast=False,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=min_pixels,
            nframes=args.nframes,
            use_fast=False,
        )

    # -------------------------------------------------------------------------
    # data
    # -------------------------------------------------------------------------
    qa_json = args.Eval_QA_root
    data = read_json_file(qa_json)

    if args.clean_subtitles:
        filtered = []
        for item in data:
            if "subtitle" in item["question"]:
                continue
            filtered.append(item)
        data = copy.deepcopy(filtered)

    keys = get_chunk(data, args.num_chunks, args.chunk_idx)

    log(f"[INFO] total_items_in_chunk={len(keys)}")
    log(f"[INFO] done_keys={len(done_keys)}")
    log(f"[INFO] failed_keys={len(failed_keys)}")
    log(f"[INFO] skip_video_ids={len(skip_video_ids)}")
    log(f"[INFO] output_json={eval_dataset_json}")
    log(f"[INFO] error_json={error_json}")

    answer_prompt = "\nAnswer with the option's letter from the given choices directly."

    for v_id, item in tqdm(
        list(enumerate(keys)),
        total=len(keys),
        dynamic_ncols=True,
    ):
        item_key = get_item_key(item)
        video_id = item.get("video_id", "")
        video_path = os.path.join(args.Eval_Video_root, item["video_path"])

        if item_key in done_keys:
            continue
        if item_key in failed_keys:
            continue
        if video_id in skip_video_ids:
            log(f"[SKIP_VIDEO_ID] idx={v_id} id={item.get('id')} video_id={video_id}")
            continue

        stage = "item_start"
        t0 = time.time()

        # 提前打印，便于定位卡住的是哪条
        log(
            f"[ITEM_START] idx={v_id}/{len(keys)-1} "
            f"id={item.get('id')} video_id={video_id} video_path={item.get('video_path')}"
        )

        # 显存清理
        torch.cuda.empty_cache()

        try:
            start_timeout(args.sample_timeout_sec)

            # -----------------------------------------------------------------
            # build prompt
            # -----------------------------------------------------------------
            stage = "qa_template"
            question, answer = qa_template(item)
            if args.add_answer_prompt:
                question += answer_prompt

            if args.nframes is None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "min_pixels": min_pixels,
                                "total_pixels": total_pixels,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "min_pixels": min_pixels,
                                "nframes": args.nframes,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]

            stage = "apply_chat_template"
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # -----------------------------------------------------------------
            # vision processing
            # -----------------------------------------------------------------
            stage = "process_vision_info"
            log(
                f"[BEFORE_PROCESS_VISION] idx={v_id} id={item.get('id')} "
                f"video_id={video_id}"
            )
            image_inputs, video_inputs = process_vision_info(messages)
            log(
                f"[AFTER_PROCESS_VISION] idx={v_id} id={item.get('id')} "
                f"video_shape={get_video_shape_str(video_inputs)}"
            )

            # -----------------------------------------------------------------
            # transformer path
            # -----------------------------------------------------------------
            if context_length < 48000:
                stage = "processor"
                log(f"[BEFORE_PROCESSOR] idx={v_id} id={item.get('id')}")
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                input_len = int(inputs.input_ids.shape[1])
                log(
                    f"[AFTER_PROCESSOR] idx={v_id} id={item.get('id')} "
                    f"input_len={input_len}"
                )

                max_pos = getattr(model.config, "max_position_embeddings", None)
                if max_pos is not None and input_len > int(max_pos):
                    raise RuntimeError(
                        f"input_len={input_len} exceeds model.config.max_position_embeddings={max_pos}"
                    )

                inputs = inputs.to(model.device)

                stage = "generate"
                log(f"[BEFORE_GENERATE] idx={v_id} id={item.get('id')}")
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        which_rope=args.which_rope,
                        scale_factor=args.scale_factor,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_text = output_text[0].strip()
                log(
                    f"[AFTER_GENERATE] idx={v_id} id={item.get('id')} "
                    f"prediction={generated_text!r}"
                )

            # -----------------------------------------------------------------
            # vllm path
            # -----------------------------------------------------------------
            else:
                stage = "vllm_prepare"
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                mm_data["which_rope"] = args.which_rope
                mm_data["scale_factor"] = args.scale_factor

                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }

                stage = "vllm_generate"
                log(f"[BEFORE_VLLM_GENERATE] idx={v_id} id={item.get('id')}")
                with torch.no_grad():
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                generated_text = outputs[0].outputs[0].text.strip()
                log(
                    f"[AFTER_VLLM_GENERATE] idx={v_id} id={item.get('id')} "
                    f"prediction={generated_text!r}"
                )

                del mm_data, llm_inputs, outputs

            # -----------------------------------------------------------------
            # save prediction
            # -----------------------------------------------------------------
            stage = "save_result"
            pred = copy.deepcopy(item)
            pred.update(
                {
                    "prediction": generated_text,
                    "answer": answer,
                    "which_rope": args.which_rope,
                    "scale_factor": args.scale_factor,
                    "context_length": args.context_length,
                    "nframes": args.nframes,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
            save_jsonl_line(eval_dataset_json, pred)
            done_keys.add(item_key)

            log(
                f"[ITEM_DONE] idx={v_id} id={item.get('id')} "
                f"elapsed_sec={round(time.time() - t0, 3)}"
            )

        except SampleTimeoutError as e:
            err = copy.deepcopy(item)
            err.update(
                {
                    "error_type": "SampleTimeoutError",
                    "error": str(e),
                    "stage": stage,
                    "which_rope": args.which_rope,
                    "scale_factor": args.scale_factor,
                    "context_length": args.context_length,
                    "nframes": args.nframes,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
            save_jsonl_line(error_json, err)
            failed_keys.add(item_key)
            log(
                f"[ITEM_TIMEOUT] idx={v_id} id={item.get('id')} stage={stage} "
                f"elapsed_sec={round(time.time() - t0, 3)}"
            )

        except Exception as e:
            err = copy.deepcopy(item)
            err.update(
                {
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "stage": stage,
                    "which_rope": args.which_rope,
                    "scale_factor": args.scale_factor,
                    "context_length": args.context_length,
                    "nframes": args.nframes,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
            save_jsonl_line(error_json, err)
            failed_keys.add(item_key)

            log(
                f"[ITEM_ERROR] idx={v_id} id={item.get('id')} "
                f"stage={stage} error={type(e).__name__}: {e}"
            )
            log(traceback.format_exc())

        finally:
            cancel_timeout()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/hwfile/mllm/weixilin/cache/Qwen2-VL-7B-Instruct",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument(
        "--Eval_QA_root",
        type=str,
        default="/fs-computility/mllm/weixilin/videorope_rebuttal/playground/data/longvideobench/json/lvb_val.json",
        help="folder containing QA JSON files",
    )
    parser.add_argument(
        "--Eval_Video_root",
        type=str,
        default="/fs-computility/mllm/weixilin/videorope_rebuttal/playground/data/longvideobench/videos/",
        help="folder containing video data",
    )
    parser.add_argument(
        "--chat_conversation_output_folder",
        type=str,
        default="/mnt/petrelfs/weixilin/projects/MLLM/Qwen2-VL/playground/results/longvideobench/",
        help="",
    )

    parser.add_argument("--context_length", type=float, default=64000)
    parser.add_argument("--min_pixels", type=float, default=144 * 28 * 28)
    parser.add_argument("--max_pixels", type=float, default=256 * 28 * 28)
    parser.add_argument("--which_rope", type=str, default="m_rope")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--clean_subtitles", action="store_true")

    # 新增参数
    parser.add_argument(
        "--sample_timeout_sec",
        type=int,
        default=180,
        help="per-sample timeout in seconds; 0 means disable timeout",
    )
    parser.add_argument(
        "--skip_video_ids",
        type=str,
        default="",
        help="comma-separated video_id list to skip manually",
    )
    parser.add_argument(
        "--skip_failed_in_errors",
        action="store_true",
        default=True,
        help="skip samples already recorded in *_errors.jsonl",
    )
    parser.add_argument(
        "--add_answer_prompt",
        action="store_true",
        default=False,
        help="append standard answer prompt to question",
    )

    args = parser.parse_args()
    eval_dataset(args)