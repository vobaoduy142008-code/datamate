#!/bin/zsh -l
# v3.10 – chỉ chạy 236 câu public cần chạy lại (~2 tiếng), rồi ghép với v3.7 thành file nộp 1000 câu.
#   - 210 câu của 4 shop sổ tay dev_011..014 (bảng chuẩn mới của v3.8)
#   - 7 câu hỏi "luật riêng" (khách trung thành, giờ cao điểm...) + 7 câu "không đủ dữ liệu"
#   - 12 câu KIỂM CHỨNG: prompt y hệt v3.7 -> phải ra giống hệt v3.7 (chứng minh phần ghép đúng)
# Dừng lúc nào cũng được (Ctrl+C); chạy lại lệnh này sẽ làm tiếp câu còn thiếu.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/public_v310
mkdir -p $OUT
{
echo "=== bắt đầu v3.10 $(date) ($(wc -l < $OUT/ids.txt) câu)"
caffeinate -i $PY versions/v3.10/main.py --data data/public_test --out $OUT --ids $OUT/ids.txt --resume
$PY scripts/ghep_v310.py data/public_test/questions.jsonl outputs/public_v37/nop_public_v37.jsonl $OUT outputs/nop_public_v310.jsonl
echo "=== xong $(date)  -> file nộp: outputs/nop_public_v310.jsonl"
} 2>&1 | tee -a $OUT/run.txt
