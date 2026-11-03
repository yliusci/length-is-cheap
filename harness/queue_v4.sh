#!/bin/zsh
# Filler v4: one model at a time, cells in priority order, 24 h in total. Start a cell only if memory >= 25 %.
cd "${0:A:h}/.."
export CTXROT_DTYPE=float16
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.42 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.38
Q15=Qwen/Qwen2.5-1.5B-Instruct Q05=Qwen/Qwen2.5-0.5B-Instruct LL=meta-llama/Llama-3.2-3B SM=HuggingFaceTB/SmolLM2-1.7B-Instruct OL=allenai/OLMo-2-0425-1B-Instruct
END=$(( $(date +%s) + ${V4_HOURS:-24} * 3600 ))
run() {  # model L seconds
  while (( $(date +%s) < END )); do
    while (( $(sysctl -n kern.memorystatus_level) < 25 )); do sleep 60; (( $(date +%s) < END )) || return; done
    left=$(( END - $(date +%s) )); s=$(( $3 < left ? $3 : left ))
    CTXROT_ONLY=$1 python harness/measure_v4.py $2 100 $s; rc=$?
    (( rc == 3 )) || { echo "case $1 $2 rc=$rc"; return; }
  done
}
for c in "$Q15 2048 5400" "$Q05 2048 5400" "$LL 2048 5400" "$SM 2048 5400" "$OL 2048 5400" "$SM 4096 3600" \
         "$Q15 8192 7200" "$Q05 8192 7200" "$LL 8192 7200" "$Q15 16384 9000" "$Q05 16384 9000" "$LL 16384 9000" \
         "$Q15 32768 9000" "$Q05 32768 9000"; do
  run ${=c}
done
echo QUEUE DONE
