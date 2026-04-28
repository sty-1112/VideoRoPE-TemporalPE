#!/usr/bin/env bash

set -euo pipefail
set -x

# ============================================================
# Conservative LoRA SFT for TemporalPE / VideoRoPE on VideoWeave
#
# 单独运行示例：
#   CUDA_VISIBLE_DEVICES=0 ROPE_MODE=temporalpe_videorope bash scripts/run_temporalpe_videorope_ft_conservative.sh
#   CUDA_VISIBLE_DEVICES=0 ROPE_MODE=videorope bash scripts/run_temporalpe_videorope_ft_conservative.sh
#
# 说明：
#   这个脚本只负责一次 train.py 调用。
#   如果 LLaMA-Factory 第一次只构建 tokenized cache 后退出，
#   外层总控脚本会自动重新运行同一个 ROPE_MODE。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LF_ROOT="${REPO_ROOT}/LLaMA-Factory"

cd "${LF_ROOT}"

# =========================
# 基本环境：默认单 GPU
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPULIST[@]}"

export WANDB_DISABLED=true
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTHONPATH="${REPO_ROOT}:${LF_ROOT}/src:${PYTHONPATH:-}"

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 强制 LLaMA-Factory 加载本地 videorope-transformer/modeling_videorope.py
export USE_LOCAL_VIDEOROPE_QWEN2VL=1

# 对已经渲染好的 16 帧视频，训练时固定取 16 帧
export LLAMAFACTORY_FIXED_VIDEO_FRAMES="${LLAMAFACTORY_FIXED_VIDEO_FRAMES:-16}"

# =========================
# 数据
# =========================
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/videorope_exp/datasets/llamafactory_webvid_video_full}"
export DATASET_NAME="${DATASET_NAME:-webvid_videoweave_l2_f8_video_sharegpt}"

# =========================
# 模型
# =========================
LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"

if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  export MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}"
elif [[ -d "${LOCAL_BASE_MODEL_DIR}" ]]; then
  export MODEL_NAME_OR_PATH="${LOCAL_BASE_MODEL_DIR}"
else
  export MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
fi

# =========================
# RoPE 模式
# =========================
export ROPE_MODE="${ROPE_MODE:-videorope}"

# =========================
# 更保守训练超参
# =========================
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-4}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"

TOTAL_MICRO_BATCH=$(( PER_DEVICE_BATCH_SIZE * NUM_GPUS ))
GRAD_ACC=$(( (FULL_BATCH_SIZE + TOTAL_MICRO_BATCH - 1) / TOTAL_MICRO_BATCH ))
if [[ "${GRAD_ACC}" -lt 1 ]]; then
  GRAD_ACC=1
fi

export MASTER_PORT="${MASTER_PORT:-29501}"

unset OMP_NUM_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export VIDEO_FPS="${VIDEO_FPS:-4.0}"
export VIDEO_MAXLEN="${VIDEO_MAXLEN:-16}"
export TOTAL_PIXELS="${TOTAL_PIXELS:-1806336}"

# 核心保守设置
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.3}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

export LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
export VAL_SIZE="${VAL_SIZE:-128}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# =========================
# 输出与缓存
# =========================
export RUN_TAG="${RUN_TAG:-conservative-lr${LEARNING_RATE}-ep${NUM_TRAIN_EPOCHS}-r${LORA_RANK}}"
export JOB_NAME="${ROPE_MODE}-${DATASET_NAME}-${RUN_TAG}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-${JOB_NAME}}"

# 注意：
# tokenized cache 不区分 ROPE_MODE。
# 第一次可能只构建 cache 后退出，第二次才真正训练。
export TOKENIZED_PATH="${TOKENIZED_PATH:-${LF_ROOT}/cache/tokenized-${DATASET_NAME}-frames${LLAMAFACTORY_FIXED_VIDEO_FRAMES}-val${VAL_SIZE}}"

if [[ "${RESET_TOKENIZED_CACHE:-0}" == "1" ]]; then
  echo "[INFO] Removing TOKENIZED_PATH=${TOKENIZED_PATH}"
  rm -rf "${TOKENIZED_PATH}"
fi

export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/videorope_exp/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log}"

# 记录 base model，后续评估 LoRA adapter 时使用
echo "${MODEL_NAME_OR_PATH}" > "${OUTPUT_DIR}/base_model_path.txt"

echo "==================================================" | tee -a "${LOG_FILE}"
echo "JOB_NAME=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "ROPE_MODE=${ROPE_MODE}" | tee -a "${LOG_FILE}"
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
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}" | tee -a "${LOG_FILE}"
echo "LEARNING_RATE=${LEARNING_RATE}" | tee -a "${LOG_FILE}"
echo "LORA_TARGET=${LORA_TARGET}" | tee -a "${LOG_FILE}"
echo "LORA_RANK=${LORA_RANK}" | tee -a "${LOG_FILE}"
echo "LORA_ALPHA=${LORA_ALPHA}" | tee -a "${LOG_FILE}"
echo "VIDEO_MAXLEN=${VIDEO_MAXLEN}" | tee -a "${LOG_FILE}"
echo "TOTAL_PIXELS=${TOTAL_PIXELS}" | tee -a "${LOG_FILE}"
echo "VAL_SIZE=${VAL_SIZE}" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  src/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_target "${LORA_TARGET}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset "${DATASET_NAME}" \
  --template qwen2_vl \
  --video_fps "${VIDEO_FPS}" \
  --video_maxlen "${VIDEO_MAXLEN}" \
  --total_pixels "${TOTAL_PIXELS}" \
  --cutoff_len 4096 \
  --overwrite_cache true \
  --tokenized_path "${TOKENIZED_PATH}" \
  --preprocessing_num_workers "${PREPROCESSING_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO}" \
  --gradient_checkpointing true \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size "${VAL_SIZE}" \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --which_rope "${ROPE_MODE}" \
  --report_to none \
  2>&1 | tee -a "${LOG_FILE}"

echo "[INFO] Training command finished. OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"