#!/bin/zsh -l
# Chạy BỘ ĐÁNH GIÁ 300 câu train (100 dễ / 100 trung bình / 100 khó) bằng một phiên bản, rồi chấm theo từng mức.
# Cách dùng:  ./chay_300.command v3.7        (hoặc v3.8 ...)   -> kết quả: outputs/eval300_<bản>
# Nếu đã có kết quả của bản v3.7 thì tự so từng câu với v3.7 (câu tốt lên / tệ đi).
cd "$(dirname "$0")"
V=${1:-v3.7}
PY=.venv/bin/python
export PYTHONUNBUFFERED=1 AGENT_VOTE=0
[ -f versions/$V/main.py ] || { echo "Không có thư mục versions/$V"; exit 1; }
if ! curl -s http://localhost:11434/api/tags > /dev/null; then open -a Ollama; sleep 15; fi
OUT=outputs/eval300_$V
mkdir -p $OUT
{
echo "=== bắt đầu $V $(date)"
caffeinate -i $PY versions/$V/main.py --data data/train --out $OUT --ids data/eval300.jsonl --resume
CMP=""
[ "$V" != "v3.7" ] && [ -f outputs/eval300_v3.7/submission.jsonl ] && CMP=outputs/eval300_v3.7/submission.jsonl
$PY scripts/cham_theo_muc.py data/eval300.jsonl data/train/answers.jsonl $OUT/submission.jsonl $CMP
echo "=== xong $V $(date)"
} 2>&1 | tee -a $OUT/run.txt
