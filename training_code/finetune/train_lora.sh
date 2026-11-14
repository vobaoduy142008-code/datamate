#!/bin/bash
# SFT LoRA (QLoRA 4-bit) cho Qwen3-8B bằng MLX - chỉnh cho MacBook M1 Pro 16 GB, ưu tiên AN TOÀN VỀ THỜI GIAN.
#   SMOKE=1 bash training_code/finetune/train_lora.sh     # chạy thử 10 bước/20 ví dụ: đo token/giây, RAM đỉnh, ước tính tổng thời gian
#   bash training_code/finetune/train_lora.sh             # train thật (~1 epoch, tối đa MAX_EXAMPLES=900 ví dụ)
#   RESUME=1 bash ...                                     # train tiếp từ adapters.safetensors
# Biến môi trường: SIZE(8b) MODEL MAX_EXAMPLES(900) EPOCHS(1) BATCH(1) NUM_LAYERS(8; thử 16) RANK(16) LR(2e-5) MAXLEN(1536) SAVE_EVERY(50) ADAPTER DATA TOKS_PER_SEC
set -euo pipefail
cd "$(dirname "$0")/../.."          # thư mục datamate
PY=.venv/bin/python

SIZE=${SIZE:-8b}                                 # 8b | 4b
SIZE_UP=$(echo "$SIZE" | tr "a-z" "A-Z")         # bash 3.2 của macOS không có ${var^^}
MODEL=${MODEL:-mlx-community/Qwen3-${SIZE_UP}-4bit}
DATA=${DATA:-data}
SMOKE=${SMOKE:-0}
BATCH=${BATCH:-1}
NUM_LAYERS=${NUM_LAYERS:-8}                      # 8 tầng cuối: nhẹ RAM; 16 cho chất lượng cao hơn một chút nếu RAM đủ
RANK=${RANK:-16}
MAXLEN=${MAXLEN:-1536}
if [ "$SMOKE" = "1" ]; then
  ADAPTER=${ADAPTER:-adapters/smoke_qwen3_${SIZE}}
  MAX_EXAMPLES=${MAX_EXAMPLES:-20}; EPOCHS=1; SAVE_EVERY=10; LR=${LR:-2e-4}   # lr lớn chỉ để adapter smoke có tác dụng đo được
  FORCE_ITERS=10; REPORT=2; VALB=2
else
  ADAPTER=${ADAPTER:-adapters/qwen3_${SIZE}_lora}
  MAX_EXAMPLES=${MAX_EXAMPLES:-900}; EPOCHS=${EPOCHS:-1}; SAVE_EVERY=${SAVE_EVERY:-50}; LR=${LR:-2e-5}
  FORCE_ITERS=""; REPORT=10; VALB=20
fi
TOKS_PER_SEC=${TOKS_PER_SEC:-90}                 # đoán cho M1 Pro 8B-4bit; SMOKE=1 sẽ đo số thật

# ---- kiểm tra ---------------------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || { echo "Cần Mac Apple Silicon (arm64)."; exit 1; }
[ -x "$PY" ] || { echo "Không thấy $PY (chưa tạo .venv)."; exit 1; }
"$PY" -c "import mlx_lm" 2>/dev/null || { echo "Chưa cài mlx-lm. Chạy: uv pip install --python .venv/bin/python mlx-lm"; exit 1; }
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/valid.jsonl" ] || { echo "Thiếu $DATA/train.jsonl hoặc valid.jsonl. Chạy chuan_bi_du_lieu.py trước."; exit 1; }

RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "RAM máy: ${RAM_GB} GB | model: $MODEL | batch $BATCH | num-layers $NUM_LAYERS | rank $RANK | max-seq $MAXLEN | SMOKE=$SMOKE"
if [ "$SIZE" = "8b" ] && [ "$RAM_GB" -le 16 ]; then
  echo "LƯU Ý (8B trên ${RAM_GB}GB): TẮT Ollama (pkill ollama; hoặc Quit Ollama ở thanh menu), đóng Chrome/VS Code/Docker Desktop, cắm sạc."
  pgrep -x ollama >/dev/null 2>&1 && echo ">>> Ollama đang chạy và có thể đang giữ model trong RAM: chạy 'pkill ollama' rồi chạy lại script này." || true
  echo "Nếu gặp 'Insufficient Memory / out of memory': NUM_LAYERS=4 MAXLEN=1024 ..., hoặc (macOS 14+) nâng giới hạn GPU: sudo sysctl iogpu.wired_limit_mb=13000 (tự reset khi khởi động lại)."
fi

# ---- tập con dữ liệu để giới hạn thời gian -------------------------------------------
mkdir -p "$ADAPTER"
USED="$ADAPTER/data_used"; mkdir -p "$USED"
head -n "$MAX_EXAMPLES" "$DATA/train.jsonl" > "$USED/train.jsonl"      # train.jsonl đã xáo trộn sẵn bởi chuan_bi_du_lieu.py
head -n 60 "$DATA/valid.jsonl" > "$USED/valid.jsonl"
N=$(wc -l < "$USED/train.jsonl" | tr -d ' ')
ITERS=${FORCE_ITERS:-$(( (N * EPOCHS + BATCH - 1) / BATCH ))}
AVG_TOK=$("$PY" - "$USED/train.jsonl" <<'PYEOF'
import json, sys
n = t = 0
for l in open(sys.argv[1], encoding="utf-8"):
    m = json.loads(l)["messages"]
    t += sum(len(x["content"]) for x in m) / 2.5 + 20 * len(m); n += 1
print(int(t / max(n, 1)))
PYEOF
)
SEC=$(( ITERS * BATCH * AVG_TOK / TOKS_PER_SEC ))
echo "Dùng $N ví dụ (~${AVG_TOK} token/ví dụ, ước lượng thô) -> $ITERS bước. Ước tính ~$(( SEC / 60 )) phút nếu ${TOKS_PER_SEC} token/giây."
[ "$SMOKE" = "1" ] || echo "(Chưa đo: chạy SMOKE=1 trước để có số thật.)"

cat > "$ADAPTER/lora_config.yaml" <<YAML
lora_parameters:
  rank: $RANK
  scale: 20.0
  dropout: 0.05
YAML

EXTRA=()
if [ "${RESUME:-0}" = "1" ] && [ -f "$ADAPTER/adapters.safetensors" ]; then
  EXTRA+=(--resume-adapter-file "$ADAPTER/adapters.safetensors"); echo "Train tiếp từ $ADAPTER/adapters.safetensors"
fi

LOG="$ADAPTER/train.log"; : > "$LOG"
echo "Bắt đầu $(date). Log: $LOG. Ctrl+C an toàn: checkpoint ${SAVE_EVERY} bước/lần (file <số>_adapters.safetensors)."
T0=$(date +%s)
caffeinate -i "$PY" -m mlx_lm.lora \
  --model "$MODEL" --train --data "$USED" -c "$ADAPTER/lora_config.yaml" \
  --fine-tune-type lora --iters "$ITERS" --batch-size "$BATCH" --num-layers "$NUM_LAYERS" \
  --learning-rate "$LR" --max-seq-length "$MAXLEN" --grad-checkpoint --mask-prompt \
  --steps-per-report "$REPORT" --steps-per-eval "$SAVE_EVERY" --val-batches "$VALB" \
  --save-every "$SAVE_EVERY" --adapter-path "$ADAPTER" --seed 0 \
  "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee -a "$LOG"
T1=$(date +%s)
echo "Xong $(date) sau $(( (T1 - T0) / 60 )) phút. Adapter: $ADAPTER/adapters.safetensors"

# ---- tổng kết tốc độ / RAM thật từ log ---------------------------------------------------
"$PY" - "$LOG" "$ITERS" "$BATCH" "$SMOKE" "$T1" "$T0" <<'PYEOF' | tee "$ADAPTER/tong_ket.txt"
import re, sys
log, iters, batch, smoke, t1, t0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
tps, peak, last_tok, last_it = [], 0.0, 0, 0
for l in open(log, encoding="utf-8", errors="ignore"):
    m = re.search(r"Iter (\d+): Train loss.*?Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB", l)
    if m:
        tps.append(float(m.group(2))); last_it = int(m.group(1)); last_tok = int(m.group(3)); peak = max(peak, float(m.group(4)))
if not tps:
    print("Không đọc được dòng 'Tokens/sec' trong log (xem log có lỗi không)."); sys.exit(0)
use = tps[1:] if len(tps) > 2 else tps          # bỏ lần báo cáo đầu (có thời gian khởi động/biên dịch)
t = sum(use) / len(use)
print(f"TỔNG KẾT: {t:.0f} token/giây | RAM đỉnh {peak:.1f} GB | {last_tok/last_it:.0f} token/bước (đo thật)")
if smoke == "1":
    print("Ước tính train thật theo số đo này (chạy lại không SMOKE):")
    for n in (300, 600, 900, 1200):
        print(f"   {n:5d} ví dụ x 1 epoch ~ {n * last_tok / last_it / t / 60:.0f} phút")
    print("Nếu RAM đỉnh > ~12 GB: giảm NUM_LAYERS/MAXLEN. Nếu máy dùng swap nhiều (Activity Monitor: Memory Pressure đỏ) thì tốc độ đo sẽ thấp hơn thực tế khi đóng ứng dụng.")
PYEOF
[ "$SMOKE" = "1" ] || echo "Val loss: grep 'Val loss' $LOG   (giảm rồi phẳng = tốt; tăng lại = dùng checkpoint sớm hơn, ADAPTER_FILE=$ADAPTER/0000300_adapters.safetensors)"
