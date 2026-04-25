#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p log
mkdir -p playground/results/longvideobench

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/longvideobench_${timestamp}.log"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=20480

BASE_MODEL_PATH="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"
EXP_ROOT="${REPO_ROOT}/videorope_exp"
CKPT_DIR="${EXP_ROOT}/checkpoints"
DATA_ROOT="${EXP_ROOT}/datasets/LongVideoBench"

if [[ -f "${DATA_ROOT}/json/lvb_val.json" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/json/lvb_val.json"
elif [[ -f "${DATA_ROOT}/lvb_val.json" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/lvb_val.json"
else
  echo "[ERROR] 找不到 lvb_val.json"
  exit 1
fi

if [[ -d "${DATA_ROOT}/videos" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/videos"
elif [[ -d "${DATA_ROOT}/video" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/video"
else
  EVAL_VIDEO_ROOT="${DATA_ROOT}"
fi

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
CHUNKS=${#GPULIST[@]}

MIN_PIXELS_FACTOR=144

# LoRA adapter 评测先只跑 transformers 路径，不跑 >=48000 的 vLLM
CONTEXT_LENGTHS=(8192 16384 32768)

MODEL_LIST=(
  # "Qwen2-VL-videorope-webvid_videoweave_l2_f8_video_sharegpt videorope 2.0"
  "Qwen2-VL-temporalpe_videorope-webvid_videoweave_l2_f8_video_sharegpt temporalpe_videorope 2.0"
)

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] EVAL_QA_ROOT=${EVAL_QA_ROOT}"
echo "[INFO] EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT}"
echo "[INFO] GPU_LIST=${GPU_LIST}"
echo "[INFO] CHUNKS=${CHUNKS}"

for MODEL_ENTRY in "${MODEL_LIST[@]}"; do
  CKPT=$(echo "${MODEL_ENTRY}" | awk '{print $1}')
  WHICH_ROPE=$(echo "${MODEL_ENTRY}" | awk '{print $2}')
  SCALE_FACTOR=$(echo "${MODEL_ENTRY}" | awk '{print $3}')

  ADAPTER_PATH="${CKPT_DIR}/${CKPT}"
  if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "[ERROR] checkpoint / adapter 目录不存在: ${ADAPTER_PATH}"
    exit 1
  fi

  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    OUTPUT_FOLDER="playground/results/longvideobench/${CKPT}-${context_length}-${MIN_PIXELS_FACTOR}tokens-clean_subtitles"
    mkdir -p "${OUTPUT_FOLDER}"

    echo "[INFO] Running ${CKPT} | rope=${WHICH_ROPE} | scale=${SCALE_FACTOR} | ctx=${context_length}"

    for IDX in $(seq 0 $((CHUNKS - 1))); do
      CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u -m eval.model_longvideobench_qwen2_vl \
        --model-path "${ADAPTER_PATH}" \
        --model-base "${BASE_MODEL_PATH}" \
        --max_new_tokens 128 \
        --Eval_QA_root "${EVAL_QA_ROOT}" \
        --Eval_Video_root "${EVAL_VIDEO_ROOT}" \
        --chat_conversation_output_folder "${OUTPUT_FOLDER}" \
        --context_length "${context_length}" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx "${IDX}" \
        --which_rope "${WHICH_ROPE}" \
        --scale_factor "${SCALE_FACTOR}" \
        --nframes 48 \
        --clean_subtitles \
        --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) &
    done

    wait
    echo "[INFO] all chunks finished for ${CKPT}, ctx=${context_length}"

    echo "[INFO] output files:"
    find "${OUTPUT_FOLDER}" -maxdepth 1 -type f | sort

    echo "[INFO] running check_long_video_bench.py ..."
    python eval/check_long_video_bench.py "${OUTPUT_FOLDER}"

    if [[ -f "${OUTPUT_FOLDER}/upload_board.json" ]]; then
      echo "[INFO] upload_board.json:"
      cat "${OUTPUT_FOLDER}/upload_board.json"
    fi

    if [[ -f "${OUTPUT_FOLDER}/results.json" ]]; then
      echo "[INFO] results.json:"
      cat "${OUTPUT_FOLDER}/results.json"
    fi

    echo "[INFO] done ${CKPT} ctx=${context_length}"
  done
done