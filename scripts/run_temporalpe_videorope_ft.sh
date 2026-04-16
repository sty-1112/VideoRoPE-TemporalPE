#!/usr/bin/env bash
set -euo pipefail
set -x

# ===== 路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_ROOT="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${LF_ROOT}/.." && pwd)"

cd "${LF_ROOT}"

# ===== 基本环境 =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPULIST[@]}"

export WANDB_DISABLED=true
export NCCL_DEBUG=INFO
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTHONPATH="${REPO_ROOT}:${LF_ROOT}/src:${PYTHONPATH:-}"

# ===== 训练配置 =====
export JOB_NAME="temporalpe_videorope_ft"
export MODEL_NAME_OR_PATH="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-videorope-128frames-8k-context-330k-llava-video"
export OUTPUT_DIR="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-${JOB_NAME}"

# 按作者脚本默认的数据集名字
export DATASETS="llava_videos_330k_split0,llava_videos_330k_split1,llava_videos_330k_split2"
export TOKENIZED_PATH="${LF_ROOT}/cache/training_qwen2vl_pretokenized_data-llava_videos_330k_other_times300k_2_3mins30k-128frames"

# ===== 批大小 =====
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-16}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACC=$(( FULL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS) ))
if [[ "${GRAD_ACC}" -lt 1 ]]; then
  GRAD_ACC=1
fi

# ===== 端口 =====
export MASTER_PORT="${MASTER_PORT:-29501}"

# ===== 启动 =====
torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  src/train.py \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset "${DATASETS}" \
  --template qwen2_vl \
  --total_pixels 6272000 \
  --video_maxlen 128 \
  --cutoff_len 8200 \
  --overwrite_cache true \
  --tokenized_path "${TOKENIZED_PATH}" \
  --preprocessing_num_workers 16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 0.2 \
  --logging_steps 1 \
  --save_steps 200 \
  --save_total_limit 2 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 100 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 200 \
  --flash_attn fa2 \
  --which_rope temporalpe_videorope \
  --report_to none