#!/bin/zsh -l
# v3.7 – KIỂM TRA HỒI QUY trên train: ĐÚNG 100 câu seed 2026 mà v3.5 đã chạy (+0.87) để so từng câu.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/seed2026_v37_100
mkdir -p $OUT
{
echo "=== bắt đầu $(date)"
caffeinate -i $PY versions/v3.7/main.py --data data/train --out $OUT --sample 100 --seed 2026 --resume
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt
