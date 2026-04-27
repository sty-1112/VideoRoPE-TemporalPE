#!/usr/bin/env bash
set -euo pipefail
set -x

# CUDA_VISIBLE_DEVICES=0 bash scripts/run_videomme_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p log
mkdir -p playground/results/video_mme

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/videomme_${timestamp}.log"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=20480
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 单 GPU
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
CHUNKS=${#GPULIST[@]}

# 基础模型目录
BASE_MODEL_PATH="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"

# checkpoint 根目录
CKPT_DIR="${REPO_ROOT}/videorope_exp/checkpoints"

# Video-MME 数据目录
DATA_ROOT="${REPO_ROOT}/videorope_exp/datasets/Video-MME"

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

if [[ -d "${DATA_ROOT}/videos" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/videos"
else
  echo "[WARN] 未发现 videos/ 子目录，默认把 ${DATA_ROOT} 当作视频根目录"
  EVAL_VIDEO_ROOT="${DATA_ROOT}"
fi

MIN_PIXELS_FACTOR=144

# 先做单卡 sanity check，只测 8192
CONTEXT_LENGTHS=(8192)

# 格式：
# "MODEL_PATH WHICH_ROPE SCALE_FACTOR MODEL_BASE"
#
# MODEL_BASE:
#   __NONE__ 表示完整模型目录
#   否则表示 LoRA adapter 需要加载的 base model
MODEL_LIST=(
  # "Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone m_rope 1.0 __NONE__"
  # "Qwen2-VL-videorope-webvid_videoweave_l2_f8_video_sharegpt videorope 2.0 ${BASE_MODEL_PATH}"
  "Qwen2-VL-temporalpe_videorope-webvid_videoweave_l2_f8_video_sharegpt temporalpe_videorope 2.0 ${BASE_MODEL_PATH}"
  # "Qwen2-VL-videorope-128frames-8k-context-330k-llava-video videorope 2.0 __NONE__"
)

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[INFO] EVAL_QA_ROOT=${EVAL_QA_ROOT}"
echo "[INFO] EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT}"
echo "[INFO] GPU_LIST=${GPU_LIST}"
echo "[INFO] CHUNKS=${CHUNKS}"

for MODEL_ENTRY in "${MODEL_LIST[@]}"; do
  MODEL_PATH_REL=$(echo "${MODEL_ENTRY}" | awk '{print $1}')
  WHICH_ROPE=$(echo "${MODEL_ENTRY}" | awk '{print $2}')
  SCALE_FACTOR=$(echo "${MODEL_ENTRY}" | awk '{print $3}')
  MODEL_BASE=$(echo "${MODEL_ENTRY}" | awk '{print $4}')

  MODEL_PATH="${CKPT_DIR}/${MODEL_PATH_REL}"
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] 模型目录不存在: ${MODEL_PATH}"
    exit 1
  fi

  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    OUTPUT_FOLDER="playground/results/video_mme/${MODEL_PATH_REL}-${WHICH_ROPE}-${context_length}-${MIN_PIXELS_FACTOR}tokens"
    mkdir -p "${OUTPUT_FOLDER}"

    echo "[INFO] Running model=${MODEL_PATH_REL} rope=${WHICH_ROPE} scale=${SCALE_FACTOR} ctx=${context_length}"

    if [[ "${MODEL_BASE}" == "__NONE__" ]]; then
      CUDA_VISIBLE_DEVICES="${GPULIST[0]}" python -u -m eval.model_videomme_qwen2_vl \
        --model-path "${MODEL_PATH}" \
        --max_new_tokens 128 \
        --Eval_QA_root "${EVAL_QA_ROOT}" \
        --Eval_Video_root "${EVAL_VIDEO_ROOT}" \
        --chat_conversation_output_folder "${OUTPUT_FOLDER}" \
        --context_length "${context_length}" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --which_rope "${WHICH_ROPE}" \
        --scale_factor "${SCALE_FACTOR}" \
        --nframes 48 \
        --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 ))
    else
      CUDA_VISIBLE_DEVICES="${GPULIST[0]}" python -u -m eval.model_videomme_qwen2_vl \
        --model-path "${MODEL_PATH}" \
        --model-base "${MODEL_BASE}" \
        --max_new_tokens 128 \
        --Eval_QA_root "${EVAL_QA_ROOT}" \
        --Eval_Video_root "${EVAL_VIDEO_ROOT}" \
        --chat_conversation_output_folder "${OUTPUT_FOLDER}" \
        --context_length "${context_length}" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --which_rope "${WHICH_ROPE}" \
        --scale_factor "${SCALE_FACTOR}" \
        --nframes 48 \
        --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 ))
    fi

    echo "[INFO] running check_videomme.py ..."
    python eval/check_videomme.py "${OUTPUT_FOLDER}"

    echo "[INFO] done model=${MODEL_PATH_REL} ctx=${context_length}"
  done
done