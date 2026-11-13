#!/bin/zsh -l
# Chạy CODE CHÍNH (src/) trên public test, chỉ những câu cần chạy lại, rồi ghép với đáp án v3.7 thành file nộp 1000 câu.
#   Danh sách câu: outputs/public_v311/ids.txt (244 câu: 210 câu 4 shop dev_011..014, câu luật riêng,
#   câu "không đủ dữ liệu", câu công nợ, + 12 câu KIỂM CHỨNG phải ra giống hệt v3.7).
# Dừng lúc nào cũng được (Ctrl+C); chạy lại lệnh này sẽ làm tiếp câu còn thiếu.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/public_v311
mkdir -p $OUT
{
echo "=== bắt đầu src $(date) ($(wc -l < $OUT/ids.txt) câu)"
caffeinate -i $PY src/main.py --data data/public_test --out $OUT --ids $OUT/ids.txt --resume
$PY scripts/ghep_v310.py data/public_test/questions.jsonl outputs/public_v37/nop_public_v37.jsonl $OUT outputs/nop_public.jsonl
echo "=== xong $(date)  -> file nộp: outputs/nop_public.jsonl"
} 2>&1 | tee -a $OUT/run.txt
