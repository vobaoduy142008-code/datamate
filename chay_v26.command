#!/bin/zsh -l
# v2.6: đo 1 lần giải. Phần 1: 50 câu train MỚI (seed 13). Phần 2: lại bộ seed 9 để so với v2.5.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
for S in 13 9; do
  OUT=outputs/seed${S}_v26
  mkdir -p $OUT
  {
  echo "=== seed $S bắt đầu $(date)"
  caffeinate -i $PY src/main.py --data data/train --out $OUT --sample 50 --seed $S
  $PY src/scorer.py data/train/answers.jsonl $OUT/submission.jsonl --subset
  echo "=== seed $S xong $(date)"
  } 2>&1 | tee $OUT/run.txt
done
