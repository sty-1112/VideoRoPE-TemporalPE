#!/usr/bin/env bash
set -euo pipefail
set -x

# 用法：
#   ROPE_MODE=videorope bash scripts/run_temporalpe_videorope_ft.sh
#   ROPE_MODE=temporalpe_videorope bash scripts/run_temporalpe_videorope_ft.sh
#
# 可选：
#   DATASET_NAME=webvid_videoweave_l2_f8_video_sharegpt
#   BASE_MODEL_PATH=/absolute/path/to/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone
#   CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LF_ROOT="${REPO_ROOT}/LLaMA-Factory"

cd "${LF_ROOT}"

# ===== 基本环境 =====
# 单 GPU 默认值
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPULIST[@]}"

export WANDB_DISABLED=true
export NCCL_DEBUG=INFO
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTHONPATH="${REPO_ROOT}:${LF_ROOT}/src:${PYTHONPATH:-}"

# 调试相关；稳定后可按需关闭
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"

# 强制 LLaMA-Factory 加载本地 videorope-transformer/modeling_videorope.py
export USE_LOCAL_VIDEOROPE_QWEN2VL=1

# 对已经渲染好的 16 帧视频，训练时固定取 16 帧
export LLAMAFACTORY_FIXED_VIDEO_FRAMES="${LLAMAFACTORY_FIXED_VIDEO_FRAMES:-16}"

# ===== 数据 =====
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/videorope_exp/datasets/llamafactory_webvid_video_full}"
export DATASET_NAME="${DATASET_NAME:-webvid_videoweave_l2_f8_video_sharegpt}"

# ===== 模型 =====
LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  export MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}"
elif [[ -d "${LOCAL_BASE_MODEL_DIR}" ]]; then
  export MODEL_NAME_OR_PATH="${LOCAL_BASE_MODEL_DIR}"
else
  export MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
fi

# ===== Rope 模式 =====
export ROPE_MODE="${ROPE_MODE:-videorope}"

# ===== 输出 =====
export JOB_NAME="${ROPE_MODE}-${DATASET_NAME}-1gpu-qk-lora"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-${JOB_NAME}}"

# 单卡重新构建 cache，避免复用旧验证划分
export TOKENIZED_PATH="${TOKENIZED_PATH:-${LF_ROOT}/cache/tokenized-${DATASET_NAME}-frames${LLAMAFACTORY_FIXED_VIDEO_FRAMES}-val128-1gpu-qk}"

# ===== 日志 =====
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/videorope_exp/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log}"

# ===== 训练超参 =====
# 单卡上建议更保守，避免把 base model 拉偏
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"

GRAD_ACC=$(( FULL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS) ))
if [[ "${GRAD_ACC}" -lt 1 ]]; then
  GRAD_ACC=1
fi

export MASTER_PORT="${MASTER_PORT:-29501}"

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export VIDEO_FPS="${VIDEO_FPS:-4.0}"
export VIDEO_MAXLEN="${VIDEO_MAXLEN:-16}"
export TOTAL_PIXELS="${TOTAL_PIXELS:-1806336}"

# 更保守的训练配方
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.5}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-4}"

# 单卡减少验证和保存频率
export VAL_SIZE="${VAL_SIZE:-128}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"

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
echo "VIDEO_MAXLEN=${VIDEO_MAXLEN}" | tee -a "${LOG_FILE}"
echo "TOTAL_PIXELS=${TOTAL_PIXELS}" | tee -a "${LOG_FILE}"
echo "VAL_SIZE=${VAL_SIZE}" | tee -a "${LOG_FILE}"
echo "FULL_BATCH_SIZE=${FULL_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "GRAD_ACC=${GRAD_ACC}" | tee -a "${LOG_FILE}"
echo "LEARNING_RATE=${LEARNING_RATE}" | tee -a "${LOG_FILE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}" | tee -a "${LOG_FILE}"
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
  --lora_target q_proj,k_proj \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
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
  --save_total_limit 3 \
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