#!/bin/zsh -l
# Chạy đo v1.7 trên 50 câu train (tạo bởi Claude). Nhấp đúp file này để chạy.
cd "$(dirname "$0")"
OUT=outputs/v19_50
mkdir -p $OUT
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
{
echo "=== bắt đầu $(date)"
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "Ollama chưa chạy -> mở Ollama"; open -a Ollama; sleep 15
fi
$PY src/tu_kiem_tra.py --data data/train --so-shop 1 | tail -3
caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 50 --seed 2
$PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset --show
echo "=== xong $(date)"
} 2>&1 | tee $OUT/run.txt
