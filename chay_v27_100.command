#!/bin/zsh -l
# v2.7: đo 1 lần giải trên 100 câu train MỚI chọn ngẫu nhiên (seed 21). Kết quả: outputs/seed21_v27_100
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/seed21_v27_100
mkdir -p $OUT
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 100 --seed 21
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
echo "=== xong $(date)"
} 2>&1 | tee $OUT/run.txt
