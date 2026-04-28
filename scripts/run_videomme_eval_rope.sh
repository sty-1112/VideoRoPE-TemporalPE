#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# VideoMME inference + scoring for one rope method.
#
# 例子：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_videomme_eval_rope.sh \
#     --rope videorope \
#     --num-gpus 1 \
#     --context-length 8192 \
#     --nframes 16 \
#     --overwrite 1
#
# 如果自动找错 adapter，手动指定：
#   --model-path /path/to/adapter_or_full_model
#
# 只对已有 0.json 打分：
#   bash scripts/run_videomme_eval_rope.sh \
#     --score-only 1 \
#     --output-dir playground/results/video_mme/xxx
#
# 关键功能：
#   1. 命令行指定 rope 方法
#   2. 命令行指定 GPU 数量
#   3. 自动按 GPU 数切分 VideoMME
#   4. 推断完成后自动运行 check_videomme.py
#   5. 如果推断文件行数已经完整但进程不退出，会自动 kill，避免卡住
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------
# Defaults
# -------------------------
ROPE="videorope"
SCALE_FACTOR="2.0"

MODEL_PATH=""
MODEL_BASE=""

NUM_GPUS=""
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
GPU_IDS_EXPLICIT=0

CONTEXT_LENGTH="8192"
NFRAMES="16"

MIN_PIXELS_FACTOR="144"
MAX_NEW_TOKENS="128"

DATA_ROOT="${REPO_ROOT}/videorope_exp/datasets/Video-MME"
EVAL_QA_ROOT=""
EVAL_VIDEO_ROOT=""

OUTPUT_DIR=""
OVERWRITE="1"
SCORE_ONLY="0"

MONITOR_INTERVAL="10"
POST_COMPLETE_GRACE="60"

# 0 表示不启用最长运行时间限制。
# 如果你想防止某个 chunk 长时间不动，可以设为秒数，比如 21600。
CHUNK_TIMEOUT_SEC="0"

FAIL_FAST="0"

# -------------------------
# Parse args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rope)
      ROPE="$2"
      shift 2
      ;;
    --scale-factor)
      SCALE_FACTOR="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-base)
      MODEL_BASE="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      GPU_IDS_EXPLICIT=1
      shift 2
      ;;
    --context-length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --nframes)
      NFRAMES="$2"
      shift 2
      ;;
    --min-pixels-factor)
      MIN_PIXELS_FACTOR="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --qa-root)
      EVAL_QA_ROOT="$2"
      shift 2
      ;;
    --video-root)
      EVAL_VIDEO_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="$2"
      shift 2
      ;;
    --score-only)
      SCORE_ONLY="$2"
      shift 2
      ;;
    --monitor-interval)
      MONITOR_INTERVAL="$2"
      shift 2
      ;;
    --post-complete-grace)
      POST_COMPLETE_GRACE="$2"
      shift 2
      ;;
    --chunk-timeout-sec)
      CHUNK_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST="1"
      shift 1
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

# -------------------------
# Resolve GPU ids / num gpus
# -------------------------
if [[ -n "${NUM_GPUS}" && "${GPU_IDS_EXPLICIT}" == "0" ]]; then
  GPU_IDS="$(python - "${NUM_GPUS}" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
)"
fi

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="${#GPU_ARRAY[@]}"
fi

if [[ "${#GPU_ARRAY[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "[ERROR] --num-gpus=${NUM_GPUS}, but --gpu-ids/CUDA_VISIBLE_DEVICES gives ${#GPU_ARRAY[@]} ids: ${GPU_IDS}"
  exit 1
fi

CHUNKS="${NUM_GPUS}"

# -------------------------
# Resolve Video-MME paths
# -------------------------
if [[ -z "${EVAL_QA_ROOT}" ]]; then
  if [[ -f "${DATA_ROOT}/Video-MME.tsv" ]]; then
    EVAL_QA_ROOT="${DATA_ROOT}/Video-MME.tsv"
  elif [[ -f "${DATA_ROOT}/data/Video-MME.tsv" ]]; then
    EVAL_QA_ROOT="${DATA_ROOT}/data/Video-MME.tsv"
  else
    echo "[ERROR] Cannot find Video-MME.tsv under ${DATA_ROOT}"
    echo "Try passing --qa-root /path/to/Video-MME.tsv"
    exit 1
  fi
fi

if [[ -z "${EVAL_VIDEO_ROOT}" ]]; then
  if [[ -d "${DATA_ROOT}/videos" ]]; then
    EVAL_VIDEO_ROOT="${DATA_ROOT}/videos"
  else
    EVAL_VIDEO_ROOT="${DATA_ROOT}"
  fi
fi

# -------------------------
# Resolve base model
# -------------------------
if [[ -z "${MODEL_BASE}" ]]; then
  LOCAL_BASE="${REPO_ROOT}/videorope_exp/checkpoints/Qwen2-VL-7B-Instruct-with-Qwen2-Language-Backbone"
  if [[ -d "${LOCAL_BASE}" ]]; then
    MODEL_BASE="${LOCAL_BASE}"
  else
    MODEL_BASE="Qwen/Qwen2-VL-7B-Instruct"
  fi
fi

# -------------------------
# Helper functions
# -------------------------
find_latest_adapter_for_rope() {
  local rope="$1"

  find "${REPO_ROOT}/videorope_exp/checkpoints" \
    -path "*Qwen2-VL-${rope}-*" \
    -name adapter_config.json \
    -printf '%T@ %h\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

count_lines() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    wc -l < "${f}" | tr -d ' '
  else
    echo "0"
  fi
}

expected_lines_for_chunk() {
  local qa="$1"
  local chunks="$2"
  local idx="$3"

  python - "${qa}" "${chunks}" "${idx}" <<'PY'
import csv
import math
import sys

qa = sys.argv[1]
chunks = int(sys.argv[2])
idx = int(sys.argv[3])

with open(qa, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

n = len(rows)
chunk_size = math.ceil(n / chunks)
start = idx * chunk_size
end = min((idx + 1) * chunk_size, n)

print(max(0, end - start))
PY
}

score_output_dir() {
  local out="$1"

  echo "=================================================="
  echo "[SCORE] ${out}"
  echo "=================================================="

  python eval/check_videomme.py "${out}" | tee "${out}/score.log"

  if [[ -f "${out}/results.json" ]]; then
    echo "[INFO] results saved to: ${out}/results.json"
  else
    echo "[ERROR] score script finished but results.json not found."
    exit 1
  fi
}

# -------------------------
# Score only mode
# -------------------------
if [[ "${SCORE_ONLY}" == "1" ]]; then
  if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] --score-only 1 requires --output-dir"
    exit 1
  fi
  score_output_dir "${OUTPUT_DIR}"
  exit 0
fi

# -------------------------
# Resolve model path
# -------------------------
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(find_latest_adapter_for_rope "${ROPE}")"
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Cannot auto find adapter for rope=${ROPE}"
    echo "Please pass --model-path /path/to/adapter_or_full_model"
    exit 1
  fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH is not a directory: ${MODEL_PATH}"
  exit 1
fi

# -------------------------
# Resolve output dir
# -------------------------
if [[ -z "${OUTPUT_DIR}" ]]; then
  MODEL_TAG="$(basename "${MODEL_PATH}")"
  PARENT_TAG="$(basename "$(dirname "${MODEL_PATH}")")"

  if [[ "${MODEL_TAG}" == checkpoint-* ]]; then
    MODEL_TAG="${PARENT_TAG}-${MODEL_TAG}"
  fi

  OUTPUT_DIR="${REPO_ROOT}/playground/results/video_mme/${MODEL_TAG}-${ROPE}-${CONTEXT_LENGTH}-${MIN_PIXELS_FACTOR}tokens-nframes${NFRAMES}"
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  echo "[INFO] Removing old output dir: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# -------------------------
# Environment
# -------------------------
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LLaMA-Factory/src:${PYTHONPATH:-}"
export USE_LOCAL_VIDEOROPE_QWEN2VL=1
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MIN_PIXELS=$(( MIN_PIXELS_FACTOR * 28 * 28 ))

echo "=================================================="
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] ROPE=${ROPE}"
echo "[INFO] SCALE_FACTOR=${SCALE_FACTOR}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] MODEL_BASE=${MODEL_BASE}"
echo "[INFO] GPU_IDS=${GPU_IDS}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}"
echo "[INFO] CHUNKS=${CHUNKS}"
echo "[INFO] CONTEXT_LENGTH=${CONTEXT_LENGTH}"
echo "[INFO] NFRAMES=${NFRAMES}"
echo "[INFO] EVAL_QA_ROOT=${EVAL_QA_ROOT}"
echo "[INFO] EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MIN_PIXELS=${MIN_PIXELS}"
echo "=================================================="

# -------------------------
# Launch chunks
# -------------------------
declare -A PIDS
declare -A EXPECTED
declare -A JSONS
declare -A START_AT
declare -A DONE
declare -A COMPLETE_AT

FAIL=0

for IDX in $(seq 0 $((CHUNKS - 1))); do
  GPU="${GPU_ARRAY[$IDX]}"
  JSON_FILE="${OUTPUT_DIR}/${IDX}.json"
  LOG_FILE="${LOG_DIR}/chunk_${IDX}.log"
  EXPECTED_LINES="$(expected_lines_for_chunk "${EVAL_QA_ROOT}" "${CHUNKS}" "${IDX}")"

  EXPECTED["${IDX}"]="${EXPECTED_LINES}"
  JSONS["${IDX}"]="${JSON_FILE}"
  START_AT["${IDX}"]="$(date +%s)"

  NFRAME_ARGS=()
  if [[ "${NFRAMES}" != "none" && -n "${NFRAMES}" ]]; then
    NFRAME_ARGS=(--nframes "${NFRAMES}")
  fi

  FAIL_FAST_ARGS=()
  if [[ "${FAIL_FAST}" == "1" ]]; then
    FAIL_FAST_ARGS=(--fail-fast)
  fi

  echo "--------------------------------------------------"
  echo "[LAUNCH] chunk=${IDX}/${CHUNKS}, gpu=${GPU}, expected_lines=${EXPECTED_LINES}"
  echo "[LAUNCH] json=${JSON_FILE}"
  echo "[LAUNCH] log=${LOG_FILE}"
  echo "--------------------------------------------------"

  CUDA_VISIBLE_DEVICES="${GPU}" python -m eval.model_videomme_qwen2_vl \
    --model-path "${MODEL_PATH}" \
    --model-base "${MODEL_BASE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --Eval_QA_root "${EVAL_QA_ROOT}" \
    --Eval_Video_root "${EVAL_VIDEO_ROOT}" \
    --chat_conversation_output_folder "${OUTPUT_DIR}" \
    --context_length "${CONTEXT_LENGTH}" \
    --num-chunks "${CHUNKS}" \
    --chunk-idx "${IDX}" \
    --which_rope "${ROPE}" \
    --scale_factor "${SCALE_FACTOR}" \
    --min_pixels "${MIN_PIXELS}" \
    "${NFRAME_ARGS[@]}" \
    "${FAIL_FAST_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &

  PIDS["${IDX}"]="$!"
done

# -------------------------
# Monitor chunks
# -------------------------
REMAINING="${CHUNKS}"

while [[ "${REMAINING}" -gt 0 ]]; do
  NOW="$(date +%s)"

  for IDX in $(seq 0 $((CHUNKS - 1))); do
    if [[ "${DONE[$IDX]:-0}" == "1" ]]; then
      continue
    fi

    PID="${PIDS[$IDX]}"
    JSON_FILE="${JSONS[$IDX]}"
    EXPECTED_LINES="${EXPECTED[$IDX]}"
    LINES="$(count_lines "${JSON_FILE}")"

    if kill -0 "${PID}" 2>/dev/null; then
      echo "[MONITOR] chunk=${IDX}, pid=${PID}, lines=${LINES}/${EXPECTED_LINES}"

      # 情况 1：json 行数已经完整，但进程仍不退出。
      # 等 POST_COMPLETE_GRACE 秒后强制结束，避免推断后卡住。
      if [[ "${EXPECTED_LINES}" -gt 0 && "${LINES}" -ge "${EXPECTED_LINES}" ]]; then
        if [[ -z "${COMPLETE_AT[$IDX]:-}" ]]; then
          COMPLETE_AT["${IDX}"]="${NOW}"
          echo "[INFO] chunk=${IDX} output is complete. Waiting ${POST_COMPLETE_GRACE}s for clean exit."
        else
          ELAPSED_AFTER_COMPLETE=$(( NOW - COMPLETE_AT[$IDX] ))
          if [[ "${ELAPSED_AFTER_COMPLETE}" -ge "${POST_COMPLETE_GRACE}" ]]; then
            echo "[WARN] chunk=${IDX} still alive after complete output. Killing pid=${PID} to avoid hang."
            kill "${PID}" 2>/dev/null || true
            sleep 5
            kill -9 "${PID}" 2>/dev/null || true
            wait "${PID}" 2>/dev/null || true

            LINES_AFTER="$(count_lines "${JSON_FILE}")"
            echo "[INFO] chunk=${IDX} killed after complete output, lines=${LINES_AFTER}/${EXPECTED_LINES}"

            if [[ "${LINES_AFTER}" -lt "${EXPECTED_LINES}" ]]; then
              echo "[ERROR] chunk=${IDX} incomplete after kill."
              FAIL=1
            fi

            DONE["${IDX}"]="1"
            REMAINING=$((REMAINING - 1))
          fi
        fi
      fi

      # 情况 2：推断长时间没有结束，也没有完整输出。
      if [[ "${CHUNK_TIMEOUT_SEC}" != "0" ]]; then
        RUNTIME=$(( NOW - START_AT[$IDX] ))
        if [[ "${RUNTIME}" -ge "${CHUNK_TIMEOUT_SEC}" ]]; then
          echo "[ERROR] chunk=${IDX} exceeded CHUNK_TIMEOUT_SEC=${CHUNK_TIMEOUT_SEC}. Killing pid=${PID}."
          kill "${PID}" 2>/dev/null || true
          sleep 5
          kill -9 "${PID}" 2>/dev/null || true
          wait "${PID}" 2>/dev/null || true

          LINES_AFTER="$(count_lines "${JSON_FILE}")"
          echo "[ERROR] chunk=${IDX} timeout, lines=${LINES_AFTER}/${EXPECTED_LINES}"
          FAIL=1
          DONE["${IDX}"]="1"
          REMAINING=$((REMAINING - 1))
        fi
      fi

    else
      # 进程已经自然退出
      RC=0
      wait "${PID}" || RC=$?
      LINES_AFTER="$(count_lines "${JSON_FILE}")"

      echo "[DONE] chunk=${IDX}, rc=${RC}, lines=${LINES_AFTER}/${EXPECTED_LINES}"

      if [[ "${LINES_AFTER}" -lt "${EXPECTED_LINES}" ]]; then
        echo "[ERROR] chunk=${IDX} incomplete. See log: ${LOG_DIR}/chunk_${IDX}.log"
        FAIL=1
      fi

      DONE["${IDX}"]="1"
      REMAINING=$((REMAINING - 1))
    fi
  done

  if [[ "${REMAINING}" -gt 0 ]]; then
    sleep "${MONITOR_INTERVAL}"
  fi
done

if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] At least one chunk failed or incomplete."
  echo "[INFO] Output dir: ${OUTPUT_DIR}"
  echo "[INFO] Logs: ${LOG_DIR}"
  exit 1
fi

# -------------------------
# Score
# -------------------------
score_output_dir "${OUTPUT_DIR}"

echo "=================================================="
echo "[DONE] VideoMME eval finished."
echo "[OUTPUT_DIR] ${OUTPUT_DIR}"
echo "[RESULTS] ${OUTPUT_DIR}/results.json"
echo "[SCORE_LOG] ${OUTPUT_DIR}/score.log"
echo "=================================================="