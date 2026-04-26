import argparse
import copy
import csv
import json
import math
import os
import glob
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

def _inject_local_qwen2vl_modeling():
    target_name = "transformers.models.qwen2_vl.modeling_qwen2_vl"
    modeling_path = Path(__file__).resolve().parents[1] / "videorope-transformer" / "modeling_videorope.py"

    if not modeling_path.is_file():
        raise FileNotFoundError(f"Local VideoRoPE modeling file not found: {modeling_path}")

    cached_module = sys.modules.get(target_name)
    if cached_module is not None and getattr(cached_module, "__file__", None) == str(modeling_path):
        return

    spec = importlib.util.spec_from_file_location(target_name, modeling_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {target_name} from {modeling_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "transformers.models.qwen2_vl"
    sys.modules[target_name] = module
    spec.loader.exec_module(module)

_inject_local_qwen2vl_modeling()

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def save_jsonl_line(file_path, data):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def proxy_off():
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""


def read_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    import random
    random.seed(233)
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]


def tsv_to_json(file_path):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            data.append(row)
    return data


def _is_lora_adapter_dir(path: str) -> bool:
    if path is None:
        return False
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_qwen2vl_model(model_path: str, model_base: Optional[str]):
    is_lora = _is_lora_adapter_dir(model_path)

    if is_lora:
        if model_base is None:
            raise ValueError(
                "Detected a LoRA adapter directory, but --model-base was not provided."
            )

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_base,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        processor_path = model_base
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor_path = model_path

    model = model.to("cuda")
    model = model.eval()
    return model, processor_path


def _resolve_item_uid(item: dict, fallback_idx: int) -> str:
    for key in ["question_id", "index", "id", "qid", "videoID", "video_id"]:
        value = item.get(key, None)
        if value is not None and str(value) != "":
            return str(value)

    video_key = str(item.get("videoID", item.get("video_id", "")))
    question = str(item.get("question", ""))
    if video_key or question:
        return f"{video_key}||{question}"

    return f"fallback_{fallback_idx}"


def _resolve_video_path(video_root: str, item: dict) -> str:
    video_id_raw = str(item.get("videoID", "")).strip()
    if video_id_raw:
        patterns = [
            os.path.join(video_root, "**", f"{video_id_raw}.mp4"),
            os.path.join(video_root, "**", f"{video_id_raw}.mkv"),
            os.path.join(video_root, "**", f"{video_id_raw}.webm"),
            os.path.join(video_root, "**", f"{video_id_raw}.avi"),
            os.path.join(video_root, "**", f"{video_id_raw}.mov"),
            os.path.join(video_root, "**", video_id_raw),
        ]
        for pat in patterns:
            matches = glob.glob(pat, recursive=True)
            if matches:
                return matches[0]

    url = str(item.get("url", "")).strip()
    if url and "v=" in url:
        youtube_id = url.split("v=")[-1].split("&")[0]
        patterns = [
            os.path.join(video_root, "**", f"{youtube_id}.mp4"),
            os.path.join(video_root, "**", f"{youtube_id}.mkv"),
            os.path.join(video_root, "**", f"{youtube_id}.webm"),
            os.path.join(video_root, "**", f"{youtube_id}.avi"),
            os.path.join(video_root, "**", f"{youtube_id}.mov"),
            os.path.join(video_root, "**", youtube_id),
        ]
        for pat in patterns:
            matches = glob.glob(pat, recursive=True)
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Cannot find local video file for question_id={item.get('question_id')} "
        f"videoID={item.get('videoID')} under {video_root}"
    )


def _normalize_question_text(question: str, options_raw: str) -> str:
    options_text = str(options_raw).strip()
    options_text = options_text.strip("[]")
    options_text = options_text.replace("\\n", "\n")
    options_text = options_text.replace("' '", "\n")
    options_text = options_text.replace("'", "")
    options_text = options_text.replace("A. ", "(A) ")
    options_text = options_text.replace("B. ", "(B) ")
    options_text = options_text.replace("C. ", "(C) ")
    options_text = options_text.replace("D. ", "(D) ")

    question = str(question).strip()
    suffix = "\nAnswer with the option's letter from the given choices directly."
    return f"{question}\nOptions:\n{options_text}{suffix}"


def eval_dataset(args):
    proxy_off()

    min_pixels = args.min_pixels
    context_length = int(args.context_length)

    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base) if args.model_base is not None else None

    llm = None
    sampling_params = None
    model = None

    if context_length >= 48000:
        if _is_lora_adapter_dir(model_path):
            raise ValueError(
                "LoRA adapter directory cannot be evaluated through the vLLM path directly. "
                "Please merge the adapter first or keep context_length < 48000."
            )

        from vllm import LLM, SamplingParams

        llm = LLM(
            model_path,
            max_model_len=context_length + 1536,
            limit_mm_per_prompt={"video": 10},
            gpu_memory_utilization=0.8,
        )
        sampling_params = SamplingParams(
            best_of=1,
            temperature=0.0,
            top_p=1,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            presence_penalty=0,
            frequency_penalty=0,
        )
        processor_path = model_path
        total_pixels = (context_length - 512) * 28 * 28
    else:
        model, processor_path = _load_qwen2vl_model(
            model_path=model_path,
            model_base=model_base,
        )
        total_pixels = (context_length - 512) * 28 * 28

    if args.nframes is None:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=min_pixels,
            total_pixels=total_pixels,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=min_pixels,
            max_pixels=min_pixels,
            nframes=args.nframes,
        )

    data = tsv_to_json(args.Eval_QA_root)
    keys = get_chunk(data, args.num_chunks, args.chunk_idx)

    eval_dataset_json = os.path.join(
        args.chat_conversation_output_folder, f"{str(args.chunk_idx)}.json"
    )
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)

    st = set()
    success_count = 0
    error_count = 0
    shown_error = 0

    for v_id, _ in tqdm(enumerate(keys), total=len(keys)):
        torch.cuda.empty_cache()

        item = keys[v_id]
        item_uid = _resolve_item_uid(item, v_id)
        if item_uid in st:
            continue

        try:
            question = _normalize_question_text(
                item.get("question", ""),
                item.get("options", ""),
            )

            video_path = _resolve_video_path(args.Eval_Video_root, item)
            if shown_error < 3:
                print(f"[DEBUG] question_id={item.get('question_id')} video_path={video_path}")

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
                    },
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
                    },
                ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs is not None and shown_error < 3:
                print(f"[DEBUG] video tensor shape={video_inputs[0].shape}")

            if context_length < 48000:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        which_rope=args.which_rope,
                        scale_factor=args.scale_factor,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_text = output_text[0]
            else:
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

                with torch.no_grad():
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                generated_text = outputs[0].outputs[0].text

            print(generated_text)

            pred = copy.deepcopy(item)
            pred.update(
                {
                    "__sample_uid__": item_uid,
                    "prediction": generated_text,
                }
            )
            save_jsonl_line(eval_dataset_json, pred)
            success_count += 1
            st.add(item_uid)

        except Exception as e:
            error_count += 1
            err_pred = copy.deepcopy(item)
            err_pred.update(
                {
                    "__sample_uid__": item_uid,
                    "prediction": "",
                    "__error__": repr(e),
                }
            )
            save_jsonl_line(eval_dataset_json, err_pred)

            if shown_error < 20:
                print(f"[ERROR] uid={item_uid} question_id={item.get('question_id')} err={repr(e)}")
                shown_error += 1

            if args.fail_fast:
                raise

        torch.cuda.empty_cache()

    print(f"[SUMMARY] success_count={success_count}, error_count={error_count}, total={len(keys)}")

    if success_count == 0:
        raise RuntimeError(
            "No sample was successfully evaluated. "
            "Please inspect the printed [ERROR] messages and the generated json file."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--Eval_QA_root", type=str, required=True)
    parser.add_argument("--Eval_Video_root", type=str, required=True)
    parser.add_argument("--chat_conversation_output_folder", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--min_pixels", type=float, default=224 * 224)
    parser.add_argument("--max_pixels", type=float, default=256 * 28 * 28)
    parser.add_argument("--which_rope", type=str, default="m_rope")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    eval_dataset(args)