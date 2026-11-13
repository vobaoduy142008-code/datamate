#!/bin/zsh -l
# v3.7 – chạy 1000 câu PUBLIC TEST. Dừng lúc nào cũng được (Ctrl+C); chạy lại lệnh này sẽ làm tiếp câu còn thiếu.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/public_v37
mkdir -p $OUT
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY versions/v3.7/main.py --data data/public_test --out $OUT --resume
$PY scripts/tao_file_nop.py data/public_test/questions.jsonl $OUT/submission.jsonl $OUT/nop_public_v37.jsonl
echo "=== xong $(date)  -> file nộp: $OUT/nop_public_v37.jsonl"
} 2>&1 | tee -a $OUT/run.txt
