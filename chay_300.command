#!/bin/zsh -l
# Chạy BỘ ĐÁNH GIÁ 300 câu train (100 dễ / 100 trung bình / 100 khó) bằng CODE CHÍNH (src/), rồi chấm theo từng mức
# và so từng câu với lần chạy v3.10 (+0,850): câu tốt lên / tệ đi.
# Cách dùng:  ./chay_300.command            -> kết quả: outputs/eval300_src
# Dừng lúc nào cũng được (Ctrl+C); chạy lại sẽ làm tiếp câu còn thiếu.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/eval300_src
mkdir -p $OUT
{
echo "=== bắt đầu src $(date)"
caffeinate -i $PY src/main.py --data data/train --out $OUT --ids data/eval300.jsonl --resume
$PY scripts/cham_theo_muc.py data/eval300.jsonl data/train/answers.jsonl $OUT/submission.jsonl outputs/eval300_v3.10/submission.jsonl
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt
