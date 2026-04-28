#!/usr/bin/env bash

set -euo pipefail
set -x

# ============================================================
# Conservative FT + VideoMME evaluation
#
# 功能：
#   1. 先训练 temporalpe_videorope
#   2. 再训练 videorope
#   3. 分别跑 VideoMME evaluation
#
# 关键修复：
#   LLaMA-Factory 第一次遇到新的 tokenized_path 时，可能只构建 tokenizer/cache 后退出。
#   本脚本会检查输出目录里是否真的出现 adapter_config.json。
#   如果没有，就自动对同一个 ROPE_MODE 再跑一次训练。
#
# 单 GPU 运行：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_conservative_ft_and_videomme.sh
#
# 常用可选参数：
#   RESET_TOKENIZED_CACHE=1          # 只在第一次 attempt 删除 cache，之后不会重复删
#   CONTEXT_LENGTHS="8192"           # 默认只跑 8192
#   NFRAMES=16                       # 默认固定 16 帧评估
#   FULL_BATCH_SIZE=4
#   PER_DEVICE_BATCH_SIZE=1
#   NUM_TRAIN_EPOCHS=0.3
#   LEARNING_RATE=2e-6
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/run_temporalpe_videorope_ft_conservative.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train script: ${TRAIN_SCRIPT}"
  exit 1
fi

mkdir -p log
mkdir -p playground/results/video_mme
mkdir -p videorope_exp/logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${REPO_ROOT}/videorope_exp/logs/conservative_ft_videomme_${TIMESTAMP}.log"
: > "${MAIN_LOG}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

# =========================
# 路径
# =========================
EXP_ROOT="${REPO_ROOT}/videorope_exp"
CKPT_DIR="${EXP_ROOT}/checkpoints"
DATA_ROOT="${EXP_ROOT}/datasets/Video-MME"

# 自动寻找 Video-MME 标注文件
if [[ -f "${DATA_ROOT}/Video-MME.tsv" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/Video-MME.tsv"
elif [[ -f "${DATA_ROOT}/data/Video-MME.tsv" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/data/Video-MME.tsv"
else
  echo "[ERROR] 找不到 Video-MME.tsv"
  echo "请确认它在以下路径之一："
  echo "  ${DATA_ROOT}/Video-MME.tsv"
  echo "  ${DATA_ROOT}/data/Video-MME.tsv"
  exit 1
fi

# 自动寻找 Video-MME 视频目录
if [[ -d "${DATA_ROOT}/videos" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/videos"
else
  echo "[WARN] 未发现 videos/ 子目录，默认把 ${DATA_ROOT} 当作视频根目录"
  EVAL_VIDEO_ROOT="${DATA_ROOT}"
fi

# =========================
# 基本环境：默认单 GPU
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
CHUNKS="${#GPULIST[@]}"

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export USE_LOCAL_VIDEOROPE_QWEN2VL=1

# =========================
# 训练数据与 base model
# =========================
export DATASET_NAME="${DATASET_NAME:-webvid_videoweave_l2_f8_video_sharegpt}"
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/videorope_exp/datasets/llamafactory_webvid_video_full}"

LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}"
elif [[ -d "${LOCAL_BASE_MODEL_DIR}" ]]; then
  MODEL_NAME_OR_PATH="${LOCAL_BASE_MODEL_DIR}"
else
  MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
fi

# =========================
# Conservative train defaults
# =========================
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-4}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.3}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

export LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export VIDEO_MAXLEN="${VIDEO_MAXLEN:-16}"
export TOTAL_PIXELS="${TOTAL_PIXELS:-1806336}"
export VAL_SIZE="${VAL_SIZE:-128}"

# 每次总控运行使用独立 RUN_TAG，避免覆盖旧实验
RUN_TAG="${RUN_TAG:-conservative-lr${LEARNING_RATE}-ep${NUM_TRAIN_EPOCHS}-r${LORA_RANK}-${TIMESTAMP}}"

# =========================
# VideoMME eval defaults
# =========================
MIN_PIXELS_FACTOR="${MIN_PIXELS_FACTOR:-144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# 默认只评估 8192，LoRA adapter 可以走 transformers path。
# context >= 48000 会走 vLLM path，不适合直接评估 LoRA adapter。
CONTEXT_LENGTHS_STR="${CONTEXT_LENGTHS:-8192}"
read -ra CONTEXT_ARRAY <<< "${CONTEXT_LENGTHS_STR}"

# 固定 16 帧，更接近 VideoWeave 训练设置，也更省显存
NFRAMES="${NFRAMES:-16}"

# 默认清理同名 eval 输出，避免旧 json 被复用
OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"

echo "=================================================="
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] MAIN_LOG=${MAIN_LOG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CHUNKS=${CHUNKS}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] DATASET_DIR=${DATASET_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] EVAL_QA_ROOT=${EVAL_QA_ROOT}"
echo "[INFO] EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT}"
echo "[INFO] CONTEXT_LENGTHS=${CONTEXT_LENGTHS_STR}"
echo "[INFO] NFRAMES=${NFRAMES}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "=================================================="

# =========================
# Helper functions
# =========================
has_lora_adapter() {
  local d="$1"

  if [[ -f "${d}/adapter_config.json" ]]; then
    return 0
  fi

  if [[ -d "${d}" ]]; then
    find "${d}" -mindepth 2 -maxdepth 3 -name adapter_config.json | grep -q .
    return $?
  fi

  return 1
}

resolve_adapter_dir() {
  local d="$1"

  if [[ -f "${d}/adapter_config.json" ]]; then
    echo "${d}"
    return 0
  fi

  local f
  f="$(find "${d}" -mindepth 2 -maxdepth 3 -name adapter_config.json | sort -V | tail -n 1 || true)"

  if [[ -z "${f}" ]]; then
    return 1
  fi

  dirname "${f}"
}

# RESET_TOKENIZED_CACHE 只允许在整个总控脚本中生效一次。
# 否则可能出现：第一次删 cache -> tokenizer 后退出；重试又删 cache -> 永远只 tokenizer。
CACHE_RESET_REMAINING="${RESET_TOKENIZED_CACHE:-0}"
TRAINED_ADAPTER_DIR=""

train_one_rope() {
  local rope_mode="$1"
  local scale_factor="$2"
  local out_dir="$3"

  local max_attempts="${MAX_TRAIN_ATTEMPTS:-3}"

  for attempt in $(seq 1 "${max_attempts}"); do
    echo "=================================================="
    echo "[TRAIN] ROPE_MODE=${rope_mode}"
    echo "[TRAIN] SCALE_FACTOR=${scale_factor}"
    echo "[TRAIN] OUT_DIR=${out_dir}"
    echo "[TRAIN] attempt=${attempt}/${max_attempts}"
    echo "[TRAIN] RESET_TOKENIZED_CACHE for this attempt=${CACHE_RESET_REMAINING}"
    echo "=================================================="

    ROPE_MODE="${rope_mode}" \
    BASE_MODEL_PATH="${MODEL_NAME_OR_PATH}" \
    DATASET_DIR="${DATASET_DIR}" \
    DATASET_NAME="${DATASET_NAME}" \
    OUTPUT_DIR="${out_dir}" \
    RUN_TAG="${RUN_TAG}" \
    RESET_TOKENIZED_CACHE="${CACHE_RESET_REMAINING}" \
    bash "${TRAIN_SCRIPT}"

    # 从第二次 attempt 开始绝对不能再删 cache
    CACHE_RESET_REMAINING=0

    if has_lora_adapter "${out_dir}"; then
      TRAINED_ADAPTER_DIR="$(resolve_adapter_dir "${out_dir}")"
      echo "[TRAIN] Found LoRA adapter: ${TRAINED_ADAPTER_DIR}"
      return 0
    fi

    echo "[WARN] No adapter_config.json found after attempt ${attempt}."
    echo "[WARN] This attempt may have only built tokenized cache."

    if [[ "${attempt}" -lt "${max_attempts}" ]]; then
      echo "[INFO] Removing incomplete output dir before retry: ${out_dir}"
      rm -rf "${out_dir}"
    fi
  done

  echo "[ERROR] Training did not produce a LoRA adapter after ${max_attempts} attempts: ${out_dir}"
  exit 1
}

# =========================
# 要跑的两个模式
# 格式：ROPE_MODE SCALE_FACTOR
# =========================
ROPE_MODES=(
  "temporalpe_videorope 2.0"
  "videorope 2.0"
)

declare -A OUTPUT_DIR_BY_ROPE
declare -A SCALE_BY_ROPE

# =========================
# Phase 1: 依次训练 TemporalPE 和 VideoRoPE
# =========================
for ENTRY in "${ROPE_MODES[@]}"; do
  ROPE_MODE="$(echo "${ENTRY}" | awk '{print $1}')"
  SCALE_FACTOR="$(echo "${ENTRY}" | awk '{print $2}')"

  OUT_DIR="${CKPT_DIR}/Qwen2-VL-${ROPE_MODE}-${DATASET_NAME}-${RUN_TAG}"

  SCALE_BY_ROPE["${ROPE_MODE}"]="${SCALE_FACTOR}"

  TRAINED_ADAPTER_DIR=""
  train_one_rope "${ROPE_MODE}" "${SCALE_FACTOR}" "${OUT_DIR}"

  OUTPUT_DIR_BY_ROPE["${ROPE_MODE}"]="${TRAINED_ADAPTER_DIR}"
  echo "[INFO] Registered adapter for ${ROPE_MODE}: ${TRAINED_ADAPTER_DIR}"
done

# =========================
# Phase 2: 分别评估两个微调结果
# =========================
for ENTRY in "${ROPE_MODES[@]}"; do
  ROPE_MODE="$(echo "${ENTRY}" | awk '{print $1}')"
  SCALE_FACTOR="$(echo "${ENTRY}" | awk '{print $2}')"
  MODEL_PATH="${OUTPUT_DIR_BY_ROPE[${ROPE_MODE}]}"

  MODEL_BASENAME="$(basename "${MODEL_PATH}")"

  echo "=================================================="
  echo "[EVAL] ROPE_MODE=${ROPE_MODE}"
  echo "[EVAL] MODEL_PATH=${MODEL_PATH}"
  echo "[EVAL] MODEL_BASE=${MODEL_NAME_OR_PATH}"
  echo "[EVAL] SCALE_FACTOR=${SCALE_FACTOR}"
  echo "=================================================="

  for CONTEXT_LENGTH in "${CONTEXT_ARRAY[@]}"; do
    if [[ -f "${MODEL_PATH}/adapter_config.json" && "${CONTEXT_LENGTH}" -ge 48000 ]]; then
      echo "[SKIP] MODEL_PATH is a LoRA adapter and context_length=${CONTEXT_LENGTH} uses vLLM path."
      echo "[SKIP] Please merge LoRA first for >=48000 context evaluation."
      continue
    fi

    OUTPUT_FOLDER="${REPO_ROOT}/playground/results/video_mme/${MODEL_BASENAME}-${CONTEXT_LENGTH}-${MIN_PIXELS_FACTOR}tokens-nframes${NFRAMES}"

    if [[ "${OVERWRITE_EVAL}" == "1" ]]; then
      rm -rf "${OUTPUT_FOLDER}"
    fi
    mkdir -p "${OUTPUT_FOLDER}"

    echo "--------------------------------------------------"
    echo "[EVAL] ROPE_MODE=${ROPE_MODE}"
    echo "[EVAL] context_length=${CONTEXT_LENGTH}"
    echo "[EVAL] output=${OUTPUT_FOLDER}"
    echo "--------------------------------------------------"

    for IDX in $(seq 0 $((CHUNKS - 1))); do
      CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m eval.model_videomme_qwen2_vl \
        --model-path "${MODEL_PATH}" \
        --model-base "${MODEL_NAME_OR_PATH}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --Eval_QA_root "${EVAL_QA_ROOT}" \
        --Eval_Video_root "${EVAL_VIDEO_ROOT}" \
        --chat_conversation_output_folder "${OUTPUT_FOLDER}" \
        --context_length "${CONTEXT_LENGTH}" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx "${IDX}" \
        --which_rope "${ROPE_MODE}" \
        --scale_factor "${SCALE_FACTOR}" \
        --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) \
        --nframes "${NFRAMES}" &
    done

    wait

    echo "[EVAL] Checking VideoMME scores for ${OUTPUT_FOLDER}"
    python eval/check_videomme.py "${OUTPUT_FOLDER}" | tee "${OUTPUT_FOLDER}/score.log"

    if [[ -f "${OUTPUT_FOLDER}/results.json" ]]; then
      echo "[EVAL] Saved results: ${OUTPUT_FOLDER}/results.json"
    else
      echo "[WARN] results.json not found in ${OUTPUT_FOLDER}"
    fi
  done
done

echo "=================================================="
echo "[DONE] Conservative FT + VideoMME evaluation finished."
echo "[LOG] ${MAIN_LOG}"
echo "=================================================="