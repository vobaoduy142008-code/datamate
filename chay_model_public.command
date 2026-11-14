#!/bin/zsh -l
# Chạy chế độ ĐIỀU PHỐI (model Qwen3-8B chọn công cụ + chuẩn hoá câu + kiểm duyệt) trên ĐỦ 1000 câu public test
# -> file nộp outputs/public_model/nop_public_model.jsonl
#   - Trong lúc chạy: caffeinate -i giữ máy KHÔNG ngủ (màn hình vẫn được tắt).
#   - Chạy XONG đủ 1000 câu: máy tự NGỦ. Không muốn ngủ:  NGU=0 ./chay_model_public.command
#   - Bấm Ctrl+C để dừng giữa chừng (máy KHÔNG ngủ); chạy lại lệnh này sẽ làm tiếp câu còn thiếu.
# Lần chạy đầu chép src/ vào outputs/public_model/code/ rồi chạy bằng bản chép đó, nên trong lúc chạy
# có sửa src/ cũng không ảnh hưởng. Muốn chạy lại TỪ ĐẦU với src/ mới: đổi tên thư mục outputs/public_model.
# Log từng câu (công cụ model chọn, câu chuẩn, kết quả kiểm duyệt): outputs/public_model/logs/log.jsonl (trường code / output).
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0 AGENT_MODE=dieu_phoi
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/public_model
mkdir -p $OUT
if [ ! -d $OUT/code ]; then
  mkdir -p $OUT/code && cp src/*.py src/*.txt src/*.json src/*.jsonl $OUT/code/ 2>/dev/null
  echo "đã chép src/ vào $OUT/code lúc $(date)" > $OUT/code/CHEP_LUC.txt
fi
{
echo "=== bắt đầu (AGENT_MODE=dieu_phoi) $(date) – code: $(cat $OUT/code/CHEP_LUC.txt)"
caffeinate -i $PY $OUT/code/main.py --data data/public_test --out $OUT --resume
$PY scripts/tao_file_nop.py data/public_test/questions.jsonl $OUT/submission.jsonl $OUT/nop_public_model.jsonl
echo "=== xong $(date)  -> file nộp: $OUT/nop_public_model.jsonl"
} 2>&1 | tee -a $OUT/run.txt

# đủ 1000 câu đã chạy -> cho máy ngủ
DA_CHAY=$(wc -l < $OUT/submission.jsonl 2>/dev/null | tr -d ' ')
if [ "${NGU:-1}" = "1" ] && [ "${DA_CHAY:-0}" -ge 1000 ]; then
  echo "Đã chạy đủ $DA_CHAY câu. Máy sẽ ngủ sau 60 giây (bấm Ctrl+C để huỷ)..." | tee -a $OUT/run.txt
  sleep 60 && pmset sleepnow
else
  echo "Mới chạy $DA_CHAY/1000 câu -> không cho máy ngủ."
fi
