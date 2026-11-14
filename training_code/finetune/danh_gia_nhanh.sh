#!/bin/bash
# Chạy model đã fine-tune trên bộ 300 câu (data/eval300.jsonl) và so với model gốc.
# Mẫu: chay_model_300.command. Kết quả: outputs/eval300_ft (model ft), baseline: outputs/eval300_model (qwen3:8b gốc).
# Dùng:
#   bash training_code/finetune/danh_gia_nhanh.sh                 # LLM_MODEL=datamate-qwen3:8b-ft
#   LLM_MODEL=datamate-qwen3:4b-ft OUT=outputs/eval300_ft4b bash training_code/finetune/danh_gia_nhanh.sh
#   LIMIT=30 bash ...   chạy thử 30 câu đầu (kiểm tra nhanh trước khi chạy đủ 300)
# Chạy lại sẽ làm tiếp câu còn thiếu (--resume). Muốn chạy lại từ đầu: đổi tên/xoá thư mục OUT.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export LLM_MODEL=${LLM_MODEL:-datamate-qwen3:8b-ft}
export PYTHONUNBUFFERED=1 AGENT_VOTE=0 AGENT_MODE=${AGENT_MODE:-dieu_phoi}
OUT=${OUT:-outputs/eval300_ft}
BASELINE=${BASELINE:-outputs/eval300_model/submission.jsonl}
IDS=data/eval300.jsonl

if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
if ! ollama list | awk 'NR>1{print $1}' | grep -qx "$LLM_MODEL"; then
  echo "Chưa có model '$LLM_MODEL' trong Ollama. Chạy fuse_va_gguf.sh trước (hoặc xem: ollama list)."; exit 1
fi
if [ -n "${LIMIT:-}" ]; then head -n "$LIMIT" "$IDS" > /tmp/ids_limit.jsonl; IDS=/tmp/ids_limit.jsonl; fi

mkdir -p "$OUT"
if [ ! -d "$OUT/code" ]; then
  mkdir -p "$OUT/code" && cp src/*.py src/*.txt src/*.json src/*.jsonl "$OUT/code/" 2>/dev/null
  echo "chép src/ lúc $(date)" > "$OUT/code/CHEP_LUC.txt"
fi
{
echo "=== bắt đầu $(date) | LLM_MODEL=$LLM_MODEL | AGENT_MODE=$AGENT_MODE | $(wc -l < "$IDS" | tr -d ' ') câu"
caffeinate -i $PY "$OUT/code/main.py" --data data/train --out "$OUT" --ids "$IDS" --resume
if [ -f "$BASELINE" ]; then
  echo "--- So sánh: [1] model ft ($OUT)  với  [2] model gốc ($BASELINE) ---"
  $PY scripts/cham_theo_muc.py "$IDS" data/train/answers.jsonl "$OUT/submission.jsonl" "$BASELINE"
else
  echo "(Chưa có baseline $BASELINE -> chạy ./chay_model_300.command với qwen3:8b để so sánh)"
  $PY scripts/cham_theo_muc.py "$IDS" data/train/answers.jsonl "$OUT/submission.jsonl"
fi
echo "=== xong $(date)"
} 2>&1 | tee -a "$OUT/run.txt"
