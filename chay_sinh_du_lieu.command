#!/bin/zsh -l
# Sinh dữ liệu SFT có kiểm chứng cho v6 (model gốc qwen3:8b qua Ollama sinh DSL -> engine -> so đáp án train; xem training_code/dsl/sinh_du_lieu.py).
# Cách dùng:  ./chay_sinh_du_lieu.command                 -> ghi training_code/dsl/sft_raw.jsonl (+ that_bai.jsonl, vi_du_sinh.jsonl)
#             ./chay_sinh_du_lieu.command --limit 60      (thử nhanh: 60 câu chia đều các shop)
#             LLM_MODEL=qwen3:14b ./chay_sinh_du_lieu.command   (đổi model sinh)
#   - Trong lúc chạy: caffeinate -i giữ máy KHÔNG ngủ. Ctrl+C để dừng; chạy lại sẽ làm tiếp (bỏ câu đã xử lý).
#   - Chạy XONG: máy tự NGỦ sau 60 giây. Không muốn ngủ:  NGU=0 ./chay_sinh_du_lieu.command
# Lần chạy đầu chép src/ và training_code/dsl/*.py vào outputs/sinh_du_lieu/code/ rồi chạy bằng bản chép đó (sửa code trong lúc chạy không ảnh hưởng).
# Muốn dùng code mới: xoá hoặc đổi tên thư mục outputs/sinh_du_lieu/code (dữ liệu đã sinh vẫn giữ, chạy tiếp được).
# Nên chạy training_code/dsl/tao_giu_rieng.py TRƯỚC (tách tập giữ riêng khỏi dữ liệu huấn luyện).
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 LLM_MODEL=${LLM_MODEL:-qwen3:8b}
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/sinh_du_lieu
mkdir -p $OUT
if [ ! -d $OUT/code ]; then
  mkdir -p $OUT/code/src $OUT/code/dsl
  cp src/*.py src/*.txt src/*.json src/*.jsonl $OUT/code/src/ 2>/dev/null
  cp training_code/dsl/*.py $OUT/code/dsl/
  echo "đã chép src/ + training_code/dsl/ lúc $(date)" > $OUT/code/CHEP_LUC.txt
fi
echo "=== bắt đầu sinh dữ liệu (model $LLM_MODEL) $(date) – code: $(cat $OUT/code/CHEP_LUC.txt)" | tee -a $OUT/run.txt
caffeinate -i $PY $OUT/code/dsl/sinh_du_lieu.py --root "$PWD" --src "$PWD/$OUT/code/src" "$@" 2>&1 | tee -a $OUT/run.txt
RC=${pipestatus[1]}
echo "=== kết thúc mã $RC $(date)" | tee -a $OUT/run.txt
if [ "${NGU:-1}" = "1" ] && [ "$RC" = "0" ]; then
  echo "Xong. Máy sẽ ngủ sau 60 giây (bấm Ctrl+C để huỷ)..."
  sleep 60 && pmset sleepnow
else
  echo "Chưa xong hoặc lỗi (mã $RC) hoặc NGU=0 -> không cho máy ngủ."
fi
