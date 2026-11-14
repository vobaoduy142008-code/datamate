#!/bin/zsh -l
# Chạy agent v6 (model dẫn dắt: hiểu câu -> DSL -> engine -> model đọc kết quả -> đối chiếu công thức -> chính sách) trên 300 câu train giữ riêng
# (data/eval300.jsonl), rồi in bảng: đúng / bỏ / sai theo mức, theo đường (dsl / tu_giai), tỉ lệ câu model giải mà công thức cũ KHÔNG bắt, độ trễ.
# Cách dùng:  ./chay_v6_300.command                                 -> model mặc định qwen3:8b, kết quả outputs/eval300_v6_qwen3_8b
#             LLM_MODEL=qwen3:4b ./chay_v6_300.command              (so 8B / 4B)
#             LLM_MODEL=datamate-qwen3:4b-ft ./chay_v6_300.command  (4B đã fine-tune)
#   Biến tuỳ chọn: IDS=data/giu_rieng.jsonl (bộ câu khác) · OUT=<thư mục kết quả> · AGENT_MODE (mặc định v6; dùng cho ablation) ·
#                  V6_TAT_DOI_CHIEU=1 · V6_CHI_DSL=1 · V6_CHI_TU_GIAI=1 · V6_CHINH_SACH=<file> · V6_SO_VI_DU=<n>
#   - Trong lúc chạy: caffeinate -i giữ máy KHÔNG ngủ. Ctrl+C để dừng; chạy lại sẽ làm tiếp câu còn thiếu.
#   - Chạy XONG đủ số câu: máy tự NGỦ sau 60 giây. Không muốn ngủ:  NGU=0 ./chay_v6_300.command
# Lần chạy đầu chép src/ vào <OUT>/code/ rồi chạy bằng bản chép đó (gồm cả src/chinh_sach_v6.json và vi_du_v6.jsonl lúc đó).
# Muốn chạy lại TỪ ĐẦU với src/ mới (vd sau hieu_chinh.py): đổi tên thư mục kết quả.
# Log từng câu (hieu, dsl, kết quả engine, lần sửa, đối chiếu, quyết định): <OUT>/logs/log.jsonl, trường output đầu tiên "V6_JSON {...}".
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0 AGENT_MODE=${AGENT_MODE-v6} LLM_MODEL=${LLM_MODEL:-qwen3:8b}
IDS=${IDS:-data/eval300.jsonl}
N_CAU=$(wc -l < $IDS | tr -d ' ')
TEN=$(echo "$LLM_MODEL" | tr '/:' '__')
OUT=${OUT:-outputs/eval300_v6_$TEN}
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
mkdir -p $OUT
if [ ! -d $OUT/code ]; then
  mkdir -p $OUT/code && cp src/*.py src/*.txt src/*.json src/*.jsonl $OUT/code/ 2>/dev/null
  echo "đã chép src/ vào $OUT/code lúc $(date)" > $OUT/code/CHEP_LUC.txt
fi
{
echo "=== bắt đầu (AGENT_MODE='$AGENT_MODE' model $LLM_MODEL, bộ $IDS) $(date) – code: $(cat $OUT/code/CHEP_LUC.txt)"
caffeinate -i $PY $OUT/code/main.py --data data/train --out $OUT --ids $IDS --resume
$PY training_code/dsl/bang_v6.py $OUT --eval $IDS
echo "=== xong $(date)"
} 2>&1 | tee -a $OUT/run.txt

# đủ số câu đã chạy -> cho máy ngủ
DA_CHAY=$(wc -l < $OUT/submission.jsonl 2>/dev/null | tr -d ' ')
if [ "${NGU:-1}" = "1" ] && [ "${DA_CHAY:-0}" -ge "$N_CAU" ]; then
  echo "Đã chạy đủ $DA_CHAY câu. Máy sẽ ngủ sau 60 giây (bấm Ctrl+C để huỷ)..." | tee -a $OUT/run.txt
  sleep 60 && pmset sleepnow
else
  echo "Mới chạy $DA_CHAY/$N_CAU câu (hoặc NGU=0) -> không cho máy ngủ."
fi
