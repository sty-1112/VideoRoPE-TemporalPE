#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/VideoRoPE

MANIFEST="videorope_exp/datasets/webvid/subsets/webvid_10k_seed42_url/download_manifest.csv"
OUTDIR="videorope_exp/datasets/webvid/videos_seq/webvid_10k_seed42_url"
SCRIPT="videorope_exp/tools/videoweave/02_download_videos_sequential.py"

BATCH_SIZE=1000
TOTAL=10000
TIMEOUT=30
RETRIES=2
SLEEP_SEC=0.5

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/batch_logs"

for START in $(seq 0 ${BATCH_SIZE} $((TOTAL - BATCH_SIZE))); do
  echo "============================================================"
  echo "Starting batch: start_index=${START}, max_items=${BATCH_SIZE}"
  echo "============================================================"

  python "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --outdir "${OUTDIR}" \
    --start-index "${START}" \
    --max-items "${BATCH_SIZE}" \
    --timeout "${TIMEOUT}" \
    --retries "${RETRIES}" \
    --sleep "${SLEEP_SEC}" \
    --skip-existing

  cp "${OUTDIR}/logs/success.csv" "${OUTDIR}/batch_logs/success_${START}.csv"
  cp "${OUTDIR}/logs/failed.csv" "${OUTDIR}/batch_logs/failed_${START}.csv"
  cp "${OUTDIR}/logs/summary.json" "${OUTDIR}/batch_logs/summary_${START}.json"

  echo "Batch ${START} finished."
done

echo "============================================================"
echo "All batches finished."
echo "Videos dir: ${OUTDIR}/videos"
echo "Meta dir:   ${OUTDIR}/meta"
echo "Caps dir:   ${OUTDIR}/captions"
echo "Logs dir:   ${OUTDIR}/batch_logs"
echo "============================================================"