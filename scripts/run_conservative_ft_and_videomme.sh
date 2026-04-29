#!/usr/bin/env bash

set -euo pipefail
set -x

# ============================================================
# Multi-experiment FT + VideoMME evaluation
#
# This script runs 3 experiment groups:
#   1) larger LR + unchanged LoRA target
#   2) current LR + more LoRA target modules
#   3) larger LR + more LoRA target modules
#
# For each group, it trains and evaluates:
#   - temporalpe_videorope
#   - videorope
#
# Total: 3 groups × 2 rope modes = 6 trained/evaluated models.
#
# Required scripts:
#   scripts/run_temporalpe_videorope_ft_conservative.sh
#   scripts/run_videomme_eval_rope.sh
#
# Single-GPU run:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_conservative_ft_and_videomme.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/run_temporalpe_videorope_ft_conservative.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/run_videomme_eval_rope.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train script: ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] Missing eval script: ${EVAL_SCRIPT}"
  echo "[ERROR] Please create scripts/run_videomme_eval_rope.sh first."
  exit 1
fi

mkdir -p log
mkdir -p playground/results/video_mme
mkdir -p videorope_exp/logs

# ============================================================
# Global experiment defaults
# 之后主要改这里。
# ============================================================

# ---------- runtime ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXP_TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# ---------- data ----------
export DATASET_NAME="${DATASET_NAME:-webvid_videoweave_l2_f8_video_sharegpt}"
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/videorope_exp/datasets/llamafactory_webvid_video_full}"

# ---------- base model ----------
LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}"
elif [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
elif [[ -d "${LOCAL_BASE_MODEL_DIR}" ]]; then
  MODEL_NAME_OR_PATH="${LOCAL_BASE_MODEL_DIR}"
else
  MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
fi
export MODEL_NAME_OR_PATH

# ---------- train: common settings ----------
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"

# 如需固定 max_steps，可设置 MAX_STEPS；为空则按 epoch 训练
export MAX_STEPS="${MAX_STEPS:-}"

# ---------- train: optimizer / scheduler ----------
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"

# ---------- train: LoRA common settings ----------
export FINETUNING_TYPE="${FINETUNING_TYPE:-lora}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ---------- train: video/input ----------
export VIDEO_FPS="${VIDEO_FPS:-4.0}"
export VIDEO_MAXLEN="${VIDEO_MAXLEN:-16}"
export TOTAL_PIXELS="${TOTAL_PIXELS:-1806336}"
export CUTOFF_LEN="${CUTOFF_LEN:-4096}"
export LLAMAFACTORY_FIXED_VIDEO_FRAMES="${LLAMAFACTORY_FIXED_VIDEO_FRAMES:-16}"

# ---------- train: validation / saving / logging ----------
export VAL_SIZE="${VAL_SIZE:-128}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"

export LOGGING_STEPS="${LOGGING_STEPS:-50}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# ---------- train: precision / memory ----------
export BF16="${BF16:-true}"
export FP16="${FP16:-false}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-180000000}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# ---------- train: cache/output ----------
export RESET_TOKENIZED_CACHE="${RESET_TOKENIZED_CACHE:-0}"
export OVERWRITE_CACHE="${OVERWRITE_CACHE:-true}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-true}"
export PLOT_LOSS="${PLOT_LOSS:-true}"
export REPORT_TO="${REPORT_TO:-none}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# ---------- eval ----------
export CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-8192}"
export NFRAMES="${NFRAMES:-16}"
export MIN_PIXELS_FACTOR="${MIN_PIXELS_FACTOR:-144}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"
export POST_COMPLETE_GRACE="${POST_COMPLETE_GRACE:-60}"
export MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"

# 若只想训练不评估，设 SKIP_EVAL=1
export SKIP_EVAL="${SKIP_EVAL:-0}"

# 若只想评估已有 adapter，不训练，设 SKIP_TRAIN=1 并手动传入：
#   TEMPORALPE_MODEL_PATH=...
#   VIDEOROPE_MODEL_PATH=...
export SKIP_TRAIN="${SKIP_TRAIN:-0}"
export TEMPORALPE_MODEL_PATH="${TEMPORALPE_MODEL_PATH:-}"
export VIDEOROPE_MODEL_PATH="${VIDEOROPE_MODEL_PATH:-}"

# ============================================================
# Sweep settings
# 这里定义三组实验。
# ============================================================

CURRENT_LR="${CURRENT_LR:-1e-6}"
BIGGER_LR="${BIGGER_LR:-2e-6}"

BASE_LORA_TARGET="${BASE_LORA_TARGET:-q_proj,v_proj}"
MORE_LORA_TARGET="${MORE_LORA_TARGET:-q_proj,k_proj}"

# 格式：
#   EXP_ID|LEARNING_RATE|LORA_TARGET|LORA_RANK|LORA_ALPHA
#
# 默认三组：
#   1. 更大学习率 + 不变 LoRA 设置
#   2. 当前学习率 + 更多 LoRA 更新参数
#   3. 更大学习率 + 更多 LoRA 更新参数
EXP_SPECS=(
  "lr2e-6-qv|${BIGGER_LR}|${BASE_LORA_TARGET}|${LORA_RANK}|${LORA_ALPHA}"
  "lr1e-6-qkvo|${CURRENT_LR}|${MORE_LORA_TARGET}|${LORA_RANK}|${LORA_ALPHA}"
  "lr2e-6-qkvo|${BIGGER_LR}|${MORE_LORA_TARGET}|${LORA_RANK}|${LORA_ALPHA}"
)

# 如果你想只跑其中一组，可以设置 EXP_FILTER：
#   EXP_FILTER=lr2e-6-qv CUDA_VISIBLE_DEVICES=0 bash scripts/run_conservative_ft_and_videomme.sh
EXP_FILTER="${EXP_FILTER:-}"

# ============================================================
# Internal setup
# ============================================================

IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPULIST[@]}"

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export USE_LOCAL_VIDEOROPE_QWEN2VL="${USE_LOCAL_VIDEOROPE_QWEN2VL:-1}"

EXP_ROOT="${REPO_ROOT}/videorope_exp"
CKPT_DIR="${EXP_ROOT}/checkpoints"
DATA_ROOT="${EXP_ROOT}/datasets/Video-MME"

MAIN_LOG="${REPO_ROOT}/videorope_exp/logs/ft_videomme_sweep_${EXP_TIMESTAMP}.log"
: > "${MAIN_LOG}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

rope_short_name() {
  local rope="$1"
  case "${rope}" in
    temporalpe|temporalpe_videorope)
      echo "temporalpe"
      ;;
    videorope)
      echo "videorope"
      ;;
    *)
      echo "${rope}" | tr '_' '-' | tr '/' '-'
      ;;
  esac
}

lora_target_short_name() {
  local target="$1"
  case "${target}" in
    q_proj,v_proj|q_proj.v_proj)
      echo "qv"
      ;;
    q_proj,k_proj|q_proj.k_proj)
      echo "qk"
      ;;
    q_proj,k_proj,v_proj,o_proj|q_proj.k_proj.v_proj.o_proj)
      echo "qkvo"
      ;;
    all)
      echo "all"
      ;;
    *)
      echo "${target}" | sed 's/_proj//g' | tr ',' '-' | tr '_' '-'
      ;;
  esac
}

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

echo "=================================================="
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] MAIN_LOG=${MAIN_LOG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] DATASET_DIR=${DATASET_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] EXP_TIMESTAMP=${EXP_TIMESTAMP}"
echo "[INFO] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[INFO] FULL_BATCH_SIZE=${FULL_BATCH_SIZE}"
echo "[INFO] PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "[INFO] BASE_LORA_TARGET=${BASE_LORA_TARGET}"
echo "[INFO] MORE_LORA_TARGET=${MORE_LORA_TARGET}"
echo "[INFO] LORA_RANK=${LORA_RANK}"
echo "[INFO] LORA_ALPHA=${LORA_ALPHA}"
echo "[INFO] LORA_DROPOUT=${LORA_DROPOUT}"
echo "[INFO] CONTEXT_LENGTHS=${CONTEXT_LENGTHS}"
echo "[INFO] NFRAMES=${NFRAMES}"
echo "[INFO] SKIP_TRAIN=${SKIP_TRAIN}"
echo "[INFO] SKIP_EVAL=${SKIP_EVAL}"
echo "[INFO] EXP_FILTER=${EXP_FILTER}"
echo "[INFO] EXP_SPECS:"
printf '  %s\n' "${EXP_SPECS[@]}"
echo "=================================================="

# RESET_TOKENIZED_CACHE 只允许在整个总控脚本中生效一次。
# 否则可能出现每个实验第一轮都删除 cache，导致不断只做 tokenizer。
CACHE_RESET_REMAINING="${RESET_TOKENIZED_CACHE}"
TRAINED_ADAPTER_DIR=""

train_one_rope() {
  local exp_id="$1"
  local rope_mode="$2"
  local scale_factor="$3"
  local out_dir="$4"
  local exp_name="$5"
  local exp_lr="$6"
  local exp_lora_target="$7"
  local exp_lora_rank="$8"
  local exp_lora_alpha="$9"

  local max_attempts="${MAX_TRAIN_ATTEMPTS:-3}"

  for attempt in $(seq 1 "${max_attempts}"); do
    echo "=================================================="
    echo "[TRAIN] EXP_ID=${exp_id}"
    echo "[TRAIN] ROPE_MODE=${rope_mode}"
    echo "[TRAIN] SCALE_FACTOR=${scale_factor}"
    echo "[TRAIN] LEARNING_RATE=${exp_lr}"
    echo "[TRAIN] LORA_TARGET=${exp_lora_target}"
    echo "[TRAIN] LORA_RANK=${exp_lora_rank}"
    echo "[TRAIN] LORA_ALPHA=${exp_lora_alpha}"
    echo "[TRAIN] EXP_NAME=${exp_name}"
    echo "[TRAIN] OUT_DIR=${out_dir}"
    echo "[TRAIN] attempt=${attempt}/${max_attempts}"
    echo "[TRAIN] RESET_TOKENIZED_CACHE for this attempt=${CACHE_RESET_REMAINING}"
    echo "=================================================="

    ROPE_MODE="${rope_mode}" \
    SCALE_FACTOR="${scale_factor}" \
    BASE_MODEL_PATH="${MODEL_NAME_OR_PATH}" \
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
    DATASET_DIR="${DATASET_DIR}" \
    DATASET_NAME="${DATASET_NAME}" \
    OUTPUT_DIR="${out_dir}" \
    EXP_NAME="${exp_name}" \
    EXP_TIMESTAMP="${EXP_TIMESTAMP}" \
    TRAIN_PARAM_TAG="${TRAIN_PARAM_TAG}" \
    RUN_TAG="${RUN_TAG}" \
    RESET_TOKENIZED_CACHE="${CACHE_RESET_REMAINING}" \
    FULL_BATCH_SIZE="${FULL_BATCH_SIZE}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    MAX_STEPS="${MAX_STEPS}" \
    LEARNING_RATE="${exp_lr}" \
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}" \
    WARMUP_RATIO="${WARMUP_RATIO}" \
    MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
    FINETUNING_TYPE="${FINETUNING_TYPE}" \
    LORA_TARGET="${exp_lora_target}" \
    LORA_RANK="${exp_lora_rank}" \
    LORA_ALPHA="${exp_lora_alpha}" \
    LORA_DROPOUT="${LORA_DROPOUT}" \
    VIDEO_FPS="${VIDEO_FPS}" \
    VIDEO_MAXLEN="${VIDEO_MAXLEN}" \
    TOTAL_PIXELS="${TOTAL_PIXELS}" \
    CUTOFF_LEN="${CUTOFF_LEN}" \
    LLAMAFACTORY_FIXED_VIDEO_FRAMES="${LLAMAFACTORY_FIXED_VIDEO_FRAMES}" \
    VAL_SIZE="${VAL_SIZE}" \
    EVAL_STRATEGY="${EVAL_STRATEGY}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE}" \
    LOGGING_STEPS="${LOGGING_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    BF16="${BF16}" \
    FP16="${FP16}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
    PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS}" \
    DDP_TIMEOUT="${DDP_TIMEOUT}" \
    MASTER_PORT="${MASTER_PORT}" \
    OVERWRITE_CACHE="${OVERWRITE_CACHE}" \
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR}" \
    PLOT_LOSS="${PLOT_LOSS}" \
    REPORT_TO="${REPORT_TO}" \
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS}" \
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
# Rope modes
# =========================
ROPE_MODES=(
  "temporalpe_videorope 2.0"
  "videorope 2.0"
)

# =========================
# Main loop: 3 experiment groups × 2 rope modes
# =========================
for SPEC in "${EXP_SPECS[@]}"; do
  IFS='|' read -r EXP_ID EXP_LR EXP_LORA_TARGET EXP_LORA_RANK EXP_LORA_ALPHA <<< "${SPEC}"

  if [[ -n "${EXP_FILTER}" && "${EXP_ID}" != "${EXP_FILTER}" ]]; then
    echo "[INFO] Skip EXP_ID=${EXP_ID} because EXP_FILTER=${EXP_FILTER}"
    continue
  fi

  EXP_LORA_TARGET_SHORT="$(lora_target_short_name "${EXP_LORA_TARGET}")"

  # 短参数名，例如：
  #   lr2e-6-ep1.0-bs1-f16-qv-r16-a32
  #   lr1e-6-ep1.0-bs1-f16-qkvo-r16-a32
  TRAIN_PARAM_TAG="lr${EXP_LR}-ep${NUM_TRAIN_EPOCHS}-bs${FULL_BATCH_SIZE}-f${VIDEO_MAXLEN}-${EXP_LORA_TARGET_SHORT}-r${EXP_LORA_RANK}-a${EXP_LORA_ALPHA}"
  RUN_TAG="${TRAIN_PARAM_TAG}-${EXP_TIMESTAMP}"

  echo "=================================================="
  echo "[EXP] EXP_ID=${EXP_ID}"
  echo "[EXP] LEARNING_RATE=${EXP_LR}"
  echo "[EXP] LORA_TARGET=${EXP_LORA_TARGET}"
  echo "[EXP] LORA_TARGET_SHORT=${EXP_LORA_TARGET_SHORT}"
  echo "[EXP] LORA_RANK=${EXP_LORA_RANK}"
  echo "[EXP] LORA_ALPHA=${EXP_LORA_ALPHA}"
  echo "[EXP] TRAIN_PARAM_TAG=${TRAIN_PARAM_TAG}"
  echo "=================================================="

  declare -A OUTPUT_DIR_BY_ROPE=()
  declare -A SCALE_BY_ROPE=()

  # -------------------------
  # Phase 1: train TemporalPE and VideoRoPE for this EXP_ID
  # -------------------------
  for ENTRY in "${ROPE_MODES[@]}"; do
    ROPE_MODE="$(echo "${ENTRY}" | awk '{print $1}')"
    SCALE_FACTOR="$(echo "${ENTRY}" | awk '{print $2}')"
    ROPE_SHORT="$(rope_short_name "${ROPE_MODE}")"

    EXP_NAME="Qwen2-VL-${ROPE_SHORT}-videoweave-${TRAIN_PARAM_TAG}-${EXP_TIMESTAMP}"
    OUT_DIR="${CKPT_DIR}/${EXP_NAME}"

    SCALE_BY_ROPE["${ROPE_MODE}"]="${SCALE_FACTOR}"

    if [[ "${SKIP_TRAIN}" == "1" ]]; then
      if [[ "${ROPE_SHORT}" == "temporalpe" ]]; then
        if [[ -z "${TEMPORALPE_MODEL_PATH}" ]]; then
          echo "[ERROR] SKIP_TRAIN=1 but TEMPORALPE_MODEL_PATH is empty."
          exit 1
        fi
        OUTPUT_DIR_BY_ROPE["${ROPE_MODE}"]="${TEMPORALPE_MODEL_PATH}"
      elif [[ "${ROPE_SHORT}" == "videorope" ]]; then
        if [[ -z "${VIDEOROPE_MODEL_PATH}" ]]; then
          echo "[ERROR] SKIP_TRAIN=1 but VIDEOROPE_MODEL_PATH is empty."
          exit 1
        fi
        OUTPUT_DIR_BY_ROPE["${ROPE_MODE}"]="${VIDEOROPE_MODEL_PATH}"
      fi
      echo "[INFO] SKIP_TRAIN=1, registered ${ROPE_MODE}: ${OUTPUT_DIR_BY_ROPE[${ROPE_MODE}]}"
      continue
    fi

    TRAINED_ADAPTER_DIR=""
    train_one_rope \
      "${EXP_ID}" \
      "${ROPE_MODE}" \
      "${SCALE_FACTOR}" \
      "${OUT_DIR}" \
      "${EXP_NAME}" \
      "${EXP_LR}" \
      "${EXP_LORA_TARGET}" \
      "${EXP_LORA_RANK}" \
      "${EXP_LORA_ALPHA}"

    OUTPUT_DIR_BY_ROPE["${ROPE_MODE}"]="${TRAINED_ADAPTER_DIR}"
    echo "[INFO] Registered adapter for ${ROPE_MODE}: ${TRAINED_ADAPTER_DIR}"
  done

  # -------------------------
  # Phase 2: eval TemporalPE and VideoRoPE for this EXP_ID
  # -------------------------
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "[INFO] SKIP_EVAL=1 for EXP_ID=${EXP_ID}, skip evaluation."
    continue
  fi

  CONTEXT_LENGTHS_STR="${CONTEXT_LENGTHS}"
  read -ra CONTEXT_ARRAY <<< "${CONTEXT_LENGTHS_STR}"

  for ENTRY in "${ROPE_MODES[@]}"; do
    ROPE_MODE="$(echo "${ENTRY}" | awk '{print $1}')"
    SCALE_FACTOR="$(echo "${ENTRY}" | awk '{print $2}')"
    MODEL_PATH="${OUTPUT_DIR_BY_ROPE[${ROPE_MODE}]}"
    MODEL_BASENAME="$(basename "${MODEL_PATH}")"

    echo "=================================================="
    echo "[EVAL] EXP_ID=${EXP_ID}"
    echo "[EVAL] ROPE_MODE=${ROPE_MODE}"
    echo "[EVAL] MODEL_PATH=${MODEL_PATH}"
    echo "[EVAL] MODEL_BASE=${MODEL_NAME_OR_PATH}"
    echo "[EVAL] SCALE_FACTOR=${SCALE_FACTOR}"
    echo "=================================================="

    for CONTEXT_LENGTH in "${CONTEXT_ARRAY[@]}"; do
      if [[ -f "${MODEL_PATH}/adapter_config.json" && "${CONTEXT_LENGTH}" -ge 48000 ]]; then
        echo "[SKIP] MODEL_PATH is a LoRA adapter and context_length=${CONTEXT_LENGTH} may use vLLM path."
        echo "[SKIP] Please merge LoRA first for >=48000 context evaluation."
        continue
      fi

      OUTPUT_FOLDER="${REPO_ROOT}/playground/results/video_mme/${MODEL_BASENAME}-ctx${CONTEXT_LENGTH}-nf${NFRAMES}"

      echo "--------------------------------------------------"
      echo "[EVAL] Calling run_videomme_eval_rope.sh"
      echo "[EVAL] EXP_ID=${EXP_ID}"
      echo "[EVAL] ROPE_MODE=${ROPE_MODE}"
      echo "[EVAL] context_length=${CONTEXT_LENGTH}"
      echo "[EVAL] output=${OUTPUT_FOLDER}"
      echo "--------------------------------------------------"

      bash "${EVAL_SCRIPT}" \
        --rope "${ROPE_MODE}" \
        --scale-factor "${SCALE_FACTOR}" \
        --model-path "${MODEL_PATH}" \
        --model-base "${MODEL_NAME_OR_PATH}" \
        --num-gpus "${NUM_GPUS}" \
        --gpu-ids "${CUDA_VISIBLE_DEVICES}" \
        --context-length "${CONTEXT_LENGTH}" \
        --nframes "${NFRAMES}" \
        --min-pixels-factor "${MIN_PIXELS_FACTOR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_FOLDER}" \
        --overwrite "${OVERWRITE_EVAL}" \
        --monitor-interval "${MONITOR_INTERVAL}" \
        --post-complete-grace "${POST_COMPLETE_GRACE}"
    done
  done

  echo "=================================================="
  echo "[EXP DONE] EXP_ID=${EXP_ID}"
  echo "=================================================="
done

echo "=================================================="
echo "[DONE] Multi-experiment FT + VideoMME evaluation finished."
echo "[LOG] ${MAIN_LOG}"
echo "=================================================="