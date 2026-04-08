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
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

EXP_ROOT="${REPO_ROOT}/videorope_exp"
CKPT_DIR="${EXP_ROOT}/checkpoints"
DATA_ROOT="${EXP_ROOT}/datasets/LongVideoBench"

if [[ -f "${DATA_ROOT}/json/lvb_val.json" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/json/lvb_val.json"
elif [[ -f "${DATA_ROOT}/lvb_val.json" ]]; then
  EVAL_QA_ROOT="${DATA_ROOT}/lvb_val.json"
else
  echo "[ERROR] 找不到 lvb_val.json"
  echo "请确认它在以下路径之一："
  echo "  ${DATA_ROOT}/json/lvb_val.json"
  echo "  ${DATA_ROOT}/lvb_val.json"
  exit 1
fi

if [[ -d "${DATA_ROOT}/videos" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/videos"
elif [[ -d "${DATA_ROOT}/video" ]]; then
  EVAL_VIDEO_ROOT="${DATA_ROOT}/video"
else
  echo "[WARN] 未发现 videos/ 或 video/，默认把 ${DATA_ROOT} 当作视频根目录"
  EVAL_VIDEO_ROOT="${DATA_ROOT}"
fi

# -----------------------------
# 运行环境
# 用法示例：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_longvideobench_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_longvideobench_all.sh
# -----------------------------

MIN_PIXELS_FACTOR=144
CONTEXT_LENGTHS=(8192)

MODEL_LIST=(
  "Qwen2-VL-vanilla_rope-128frames-8k-context-330k-llava-video vanilla_rope 1.0"
  "Qwen2-VL-tad_rope-128frames-8k-context-330k-llava-video tad_rope 1.0"
  "Qwen2-VL-m_rope-128frames-8k-context-330k-llava-video m_rope 1.0"
  "Qwen2-VL-videorope-128frames-8k-context-330k-llava-video videorope 2.0"
)

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] EVAL_QA_ROOT=${EVAL_QA_ROOT}"
echo "[INFO] EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT}"

for MODEL_ENTRY in "${MODEL_LIST[@]}"; do
  CKPT=$(echo "${MODEL_ENTRY}" | awk '{print $1}')
  WHICH_ROPE=$(echo "${MODEL_ENTRY}" | awk '{print $2}')
  SCALE_FACTOR=$(echo "${MODEL_ENTRY}" | awk '{print $3}')

  if [[ ! -d "${CKPT_DIR}/${CKPT}" ]]; then
    echo "[ERROR] checkpoint 目录不存在: ${CKPT_DIR}/${CKPT}"
    exit 1
  fi

  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    OUTPUT_FOLDER="playground/results/longvideobench/${CKPT}-${context_length}-${MIN_PIXELS_FACTOR}tokens-clean_subtitles"
    mkdir -p "${OUTPUT_FOLDER}"

    echo "[INFO] Running ${CKPT} | rope=${WHICH_ROPE} | scale=${SCALE_FACTOR} | ctx=${context_length}"

    CUDA_VISIBLE_DEVICES=0 python -u -m eval.model_longvideobench_qwen2_vl \
      --model-path "${CKPT_DIR}/${CKPT}" \
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
      --clean_subtitles \
      --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 ))

    RESULT_JSON="${OUTPUT_FOLDER}/0.json"
    if [[ ! -f "${RESULT_JSON}" ]]; then
      echo "[ERROR] 结果文件不存在: ${RESULT_JSON}"
      exit 1
    fi

    LINE_COUNT=$(wc -l < "${RESULT_JSON}")
    echo "[INFO] ${RESULT_JSON} line_count=${LINE_COUNT}"

    if [[ "${LINE_COUNT}" -ne 964 ]]; then
      echo "[ERROR] LongVideoBench clean_subtitles 结果数不是 964，而是 ${LINE_COUNT}"
      exit 1
    fi

    python eval/check_long_video_bench.py "${OUTPUT_FOLDER}"

    echo "[INFO] upload_board.json for ${CKPT}:"
    cat "${OUTPUT_FOLDER}/upload_board.json"
    echo
  done
done