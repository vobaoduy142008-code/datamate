#!/bin/zsh -l
# v3.2: đo 1 lần giải trên 100 câu train MỚI chọn ngẫu nhiên (seed 610). Kết quả: outputs/seed610_v32_100
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/seed610_v32_100
mkdir -p $OUT
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 100 --seed 610 --resume
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt
