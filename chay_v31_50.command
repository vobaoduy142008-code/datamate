#!/bin/zsh -l
# v3.1: đo 1 lần giải trên 50 câu train MỚI chọn ngẫu nhiên (seed 233). Kết quả: outputs/seed233_v31_50
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/seed233_v31_50
mkdir -p $OUT
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 50 --seed 233 --resume
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt
