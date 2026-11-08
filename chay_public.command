#!/bin/zsh -l
# Chạy bài trên PUBLIC TEST (1000 câu) trên Mac. Dừng giữa chừng thì chạy lại file này: tự chạy tiếp câu còn thiếu.
# Lúc nào cũng tạo được file nộp đủ 1000 câu (câu chưa chạy = abstain): outputs/public/nop_public.jsonl
cd "$(dirname "$0")"
OUT=outputs/public
mkdir -p $OUT
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "Ollama chưa chạy -> mở Ollama"; open -a Ollama; sleep 15
fi
caffeinate -i $PY src/main.py --data data/public_test --out $OUT --resume 2>&1 | tee -a $OUT/run.txt
$PY scripts/tao_file_nop.py data/public_test/questions.jsonl $OUT/submission.jsonl $OUT/nop_public.jsonl
