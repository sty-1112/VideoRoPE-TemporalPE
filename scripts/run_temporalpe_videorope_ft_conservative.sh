#!/usr/bin/env bash

set -euo pipefail
set -x

# ============================================================
# Parameterized LoRA SFT for TemporalPE / VideoRoPE on VideoWeave
#
# 设计目标：
#   - FT 脚本只负责执行一次训练。
#   - 大部分训练参数都可以从外部环境变量传入。
#   - 总控脚本负责设置默认实验参数、命名、训练顺序和评估。
#
# 单独运行示例：
#   CUDA_VISIBLE_DEVICES=0 \
#   ROPE_MODE=temporalpe_videorope \
#   NUM_TRAIN_EPOCHS=1.0 \
#   LEARNING_RATE=5e-7 \
#   bash scripts/run_temporalpe_videorope_ft_conservative.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LF_ROOT="${REPO_ROOT}/LLaMA-Factory"

cd "${LF_ROOT}"

# =========================
# 基本环境
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPULIST[@]}"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTHONPATH="${REPO_ROOT}:${LF_ROOT}/src:${PYTHONPATH:-}"

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# 强制 LLaMA-Factory 加载本地 videorope-transformer/modeling_videorope.py
export USE_LOCAL_VIDEOROPE_QWEN2VL="${USE_LOCAL_VIDEOROPE_QWEN2VL:-1}"

# 对已经渲染好的视频，训练时固定读取帧数
export LLAMAFACTORY_FIXED_VIDEO_FRAMES="${LLAMAFACTORY_FIXED_VIDEO_FRAMES:-16}"

# =========================
# 数据参数
# =========================
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/videorope_exp/datasets/llamafactory_webvid_video_full}"
export DATASET_NAME="${DATASET_NAME:-webvid_videoweave_l2_f8_video_sharegpt}"

# =========================
# 模型参数
# =========================
LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  export MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}"
elif [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
  export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
elif [[ -d "${LOCAL_BASE_MODEL_DIR}" ]]; then
  export MODEL_NAME_OR_PATH="${LOCAL_BASE_MODEL_DIR}"
else
  export MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
fi

# =========================
# RoPE 参数
# =========================
export ROPE_MODE="${ROPE_MODE:-videorope}"
export SCALE_FACTOR="${SCALE_FACTOR:-2.0}"

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

ROPE_SHORT="$(rope_short_name "${ROPE_MODE}")"

# =========================
# 训练 batch / step 参数
# =========================
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"

TOTAL_MICRO_BATCH=$(( PER_DEVICE_BATCH_SIZE * NUM_GPUS ))

if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS}"
else
  GRAD_ACC=$(( (FULL_BATCH_SIZE + TOTAL_MICRO_BATCH - 1) / TOTAL_MICRO_BATCH ))
  if [[ "${GRAD_ACC}" -lt 1 ]]; then
    GRAD_ACC=1
  fi
fi

EFFECTIVE_BATCH=$(( TOTAL_MICRO_BATCH * GRAD_ACC ))

export MASTER_PORT="${MASTER_PORT:-29501}"

# =========================
# 视频与输入长度参数
# =========================
export VIDEO_FPS="${VIDEO_FPS:-4.0}"
export VIDEO_MAXLEN="${VIDEO_MAXLEN:-16}"
export TOTAL_PIXELS="${TOTAL_PIXELS:-1806336}"
export CUTOFF_LEN="${CUTOFF_LEN:-4096}"

# =========================
# 核心训练超参
# =========================
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"

# 如需使用 max_steps 覆盖 epoch，可外部传入 MAX_STEPS
export MAX_STEPS="${MAX_STEPS:-}"

# =========================
# LoRA 参数
# =========================
export FINETUNING_TYPE="${FINETUNING_TYPE:-lora}"
export LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

LORA_TARGET_SHORT="$(lora_target_short_name "${LORA_TARGET}")"

# =========================
# LLaMA-Factory 训练行为参数
# =========================
export STAGE="${STAGE:-sft}"
export DO_TRAIN="${DO_TRAIN:-true}"
export TEMPLATE="${TEMPLATE:-qwen2_vl}"
export BF16="${BF16:-true}"
export FP16="${FP16:-false}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-180000000}"

export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
export OVERWRITE_CACHE="${OVERWRITE_CACHE:-true}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-true}"

export VAL_SIZE="${VAL_SIZE:-128}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
export EVAL_STEPS="${EVAL_STEPS:-500}"

export LOGGING_STEPS="${LOGGING_STEPS:-50}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export PLOT_LOSS="${PLOT_LOSS:-true}"
export REPORT_TO="${REPORT_TO:-none}"

# 额外透传参数，例如：
#   EXTRA_TRAIN_ARGS="--flash_attn fa2 --packing false"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# =========================
# 命名与输出
# =========================
export EXP_TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# 默认参数短名：
#   lr5e-7-ep1.0-bs1-f16-qv-r16-a32
export TRAIN_PARAM_TAG="${TRAIN_PARAM_TAG:-lr${LEARNING_RATE}-ep${NUM_TRAIN_EPOCHS}-bs${FULL_BATCH_SIZE}-f${VIDEO_MAXLEN}-${LORA_TARGET_SHORT}-r${LORA_RANK}-a${LORA_ALPHA}}"

if [[ -n "${EXP_NAME:-}" ]]; then
  export EXP_NAME="${EXP_NAME}"
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  export EXP_NAME="$(basename "${OUTPUT_DIR}")"
else
  export EXP_NAME="Qwen2-VL-${ROPE_SHORT}-videoweave-${TRAIN_PARAM_TAG}-${EXP_TIMESTAMP}"
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR}"
else
  export OUTPUT_DIR="${REPO_ROOT}/videorope_exp/checkpoints/${EXP_NAME}"
fi

export RUN_TAG="${RUN_TAG:-${TRAIN_PARAM_TAG}-${EXP_TIMESTAMP}}"
export JOB_NAME="${EXP_NAME}"

# =========================
# Tokenized cache
# =========================
export TOKENIZED_PATH="${TOKENIZED_PATH:-${LF_ROOT}/cache/tokenized-${DATASET_NAME}-frames${LLAMAFACTORY_FIXED_VIDEO_FRAMES}-val${VAL_SIZE}}"

if [[ "${RESET_TOKENIZED_CACHE:-0}" == "1" ]]; then
  echo "[INFO] Removing TOKENIZED_PATH=${TOKENIZED_PATH}"
  rm -rf "${TOKENIZED_PATH}"
fi

# =========================
# Log
# =========================
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/videorope_exp/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}_${LOG_TIMESTAMP}.log}"

echo "${MODEL_NAME_OR_PATH}" > "${OUTPUT_DIR}/base_model_path.txt"

echo "==================================================" | tee -a "${LOG_FILE}"
echo "EXP_NAME=${EXP_NAME}" | tee -a "${LOG_FILE}"
echo "JOB_NAME=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "ROPE_MODE=${ROPE_MODE}" | tee -a "${LOG_FILE}"
echo "ROPE_SHORT=${ROPE_SHORT}" | tee -a "${LOG_FILE}"
echo "SCALE_FACTOR=${SCALE_FACTOR}" | tee -a "${LOG_FILE}"
echo "TRAIN_PARAM_TAG=${TRAIN_PARAM_TAG}" | tee -a "${LOG_FILE}"
echo "EXP_TIMESTAMP=${EXP_TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "DATASET_DIR=${DATASET_DIR}" | tee -a "${LOG_FILE}"
echo "DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}" | tee -a "${LOG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "TOKENIZED_PATH=${TOKENIZED_PATH}" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${LOG_FILE}"
echo "FULL_BATCH_SIZE=${FULL_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "GRAD_ACC=${GRAD_ACC}" | tee -a "${LOG_FILE}"
echo "EFFECTIVE_BATCH=${EFFECTIVE_BATCH}" | tee -a "${LOG_FILE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}" | tee -a "${LOG_FILE}"
echo "MAX_STEPS=${MAX_STEPS}" | tee -a "${LOG_FILE}"
echo "LEARNING_RATE=${LEARNING_RATE}" | tee -a "${LOG_FILE}"
echo "LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}" | tee -a "${LOG_FILE}"
echo "WARMUP_RATIO=${WARMUP_RATIO}" | tee -a "${LOG_FILE}"
echo "MAX_GRAD_NORM=${MAX_GRAD_NORM}" | tee -a "${LOG_FILE}"
echo "LORA_TARGET=${LORA_TARGET}" | tee -a "${LOG_FILE}"
echo "LORA_TARGET_SHORT=${LORA_TARGET_SHORT}" | tee -a "${LOG_FILE}"
echo "LORA_RANK=${LORA_RANK}" | tee -a "${LOG_FILE}"
echo "LORA_ALPHA=${LORA_ALPHA}" | tee -a "${LOG_FILE}"
echo "LORA_DROPOUT=${LORA_DROPOUT}" | tee -a "${LOG_FILE}"
echo "VIDEO_FPS=${VIDEO_FPS}" | tee -a "${LOG_FILE}"
echo "VIDEO_MAXLEN=${VIDEO_MAXLEN}" | tee -a "${LOG_FILE}"
echo "TOTAL_PIXELS=${TOTAL_PIXELS}" | tee -a "${LOG_FILE}"
echo "CUTOFF_LEN=${CUTOFF_LEN}" | tee -a "${LOG_FILE}"
echo "VAL_SIZE=${VAL_SIZE}" | tee -a "${LOG_FILE}"
echo "EVAL_STRATEGY=${EVAL_STRATEGY}" | tee -a "${LOG_FILE}"
echo "EVAL_STEPS=${EVAL_STEPS}" | tee -a "${LOG_FILE}"
echo "SAVE_STEPS=${SAVE_STEPS}" | tee -a "${LOG_FILE}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS}" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

# =========================
# Build train args
# =========================
TRAIN_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --stage "${STAGE}"
  --do_train "${DO_TRAIN}"
  --finetuning_type "${FINETUNING_TYPE}"
  --lora_target "${LORA_TARGET}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --dataset_dir "${DATASET_DIR}"
  --dataset "${DATASET_NAME}"
  --template "${TEMPLATE}"
  --video_fps "${VIDEO_FPS}"
  --video_maxlen "${VIDEO_MAXLEN}"
  --total_pixels "${TOTAL_PIXELS}"
  --cutoff_len "${CUTOFF_LEN}"
  --overwrite_cache "${OVERWRITE_CACHE}"
  --tokenized_path "${TOKENIZED_PATH}"
  --preprocessing_num_workers "${PREPROCESSING_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --plot_loss "${PLOT_LOSS}"
  --overwrite_output_dir "${OVERWRITE_OUTPUT_DIR}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --ddp_timeout "${DDP_TIMEOUT}"
  --val_size "${VAL_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --eval_strategy "${EVAL_STRATEGY}"
  --eval_steps "${EVAL_STEPS}"
  --which_rope "${ROPE_MODE}"
  --report_to "${REPORT_TO}"
)

if [[ "${BF16}" == "true" ]]; then
  TRAIN_ARGS+=(--bf16 true)
fi

if [[ "${FP16}" == "true" ]]; then
  TRAIN_ARGS+=(--fp16 true)
fi

if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${MAX_GRAD_NORM}" ]]; then
  TRAIN_ARGS+=(--max_grad_norm "${MAX_GRAD_NORM}")
fi

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # 支持简单追加参数，例如：
  #   EXTRA_TRAIN_ARGS="--flash_attn fa2 --packing false"
  # 注意不要在这里传入带复杂嵌套引号的内容。
  read -r -a EXTRA_TRAIN_ARGS_ARRAY <<< "${EXTRA_TRAIN_ARGS}"
  TRAIN_ARGS+=("${EXTRA_TRAIN_ARGS_ARRAY[@]}")
fi

torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  src/train.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[INFO] Training command finished. OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"