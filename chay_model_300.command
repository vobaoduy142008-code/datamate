#!/bin/zsh -l
# Chạy chế độ ĐIỀU PHỐI (model Qwen3-8B chọn công cụ + chuẩn hoá câu + kiểm duyệt) trên BỘ 300 câu train (data/eval300.jsonl),
# rồi chấm theo từng mức (dễ / trung bình / khó) và so với lần chạy chế độ cũ outputs/eval300_src (nếu có).
# Cách dùng:  ./chay_model_300.command         -> kết quả: outputs/eval300_model
#   - Trong lúc chạy: caffeinate -i giữ máy KHÔNG ngủ. Ctrl+C để dừng; chạy lại sẽ làm tiếp câu còn thiếu.
#   - Chạy XONG đủ 300 câu: máy tự NGỦ sau 60 giây. Không muốn ngủ:  NGU=0 ./chay_model_300.command
# Lần chạy đầu chép src/ vào outputs/eval300_model/code/ rồi chạy bằng bản chép đó (sửa src/ trong lúc chạy không ảnh hưởng).
# Muốn chạy lại TỪ ĐẦU với src/ mới: đổi tên thư mục outputs/eval300_model.
# Log từng câu (công cụ model chọn, câu chuẩn, kết quả kiểm duyệt): outputs/eval300_model/logs/log.jsonl (trường code / output).
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0 AGENT_MODE=dieu_phoi
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/eval300_model
mkdir -p $OUT
if [ ! -d $OUT/code ]; then
  mkdir -p $OUT/code && cp src/*.py src/*.txt src/*.json src/*.jsonl $OUT/code/ 2>/dev/null
  echo "đã chép src/ vào $OUT/code lúc $(date)" > $OUT/code/CHEP_LUC.txt
fi
{
echo "=== bắt đầu (AGENT_MODE=dieu_phoi) $(date) – code: $(cat $OUT/code/CHEP_LUC.txt)"
caffeinate -i $PY $OUT/code/main.py --data data/train --out $OUT --ids data/eval300.jsonl --resume
if [ -f outputs/eval300_src/submission.jsonl ]; then
  $PY scripts/cham_theo_muc.py data/eval300.jsonl data/train/answers.jsonl $OUT/submission.jsonl outputs/eval300_src/submission.jsonl
else
  $PY scripts/cham_theo_muc.py data/eval300.jsonl data/train/answers.jsonl $OUT/submission.jsonl
fi
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt

# đủ 300 câu đã chạy -> cho máy ngủ
DA_CHAY=$(wc -l < $OUT/submission.jsonl 2>/dev/null | tr -d ' ')
if [ "${NGU:-1}" = "1" ] && [ "${DA_CHAY:-0}" -ge 300 ]; then
  echo "Đã chạy đủ $DA_CHAY câu. Máy sẽ ngủ sau 60 giây (bấm Ctrl+C để huỷ)..." | tee -a $OUT/run.txt
  sleep 60 && pmset sleepnow
else
  echo "Mới chạy $DA_CHAY/300 câu -> không cho máy ngủ."
fi
