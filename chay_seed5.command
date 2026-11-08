#!/bin/zsh -l
# Đo trung thực: 50 câu train MỚI (seed 5), giải 2 lần mỗi câu (AGENT_VOTE=1). Kết quả: outputs/seed5_vote
cd "$(dirname "$0")"
OUT=outputs/seed5_vote
mkdir -p $OUT
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=1
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 50 --seed 5
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
echo "=== xong $(date)"
} 2>&1 | tee $OUT/run.txt
