#!/bin/bash
# ĐƯỜNG A (fuse, chắc chắn chạy được trong Ollama): gộp LoRA vào base 4-bit + de-quantize -> HF (bf16, ~16GB)
#   -> GGUF bf16 -> llama-quantize Q4_K_M -> Modelfile (template/params của qwen3:8b) -> ollama create.
# Trên Mac 16GB bước 1 sẽ dùng swap (chậm, bình thường). Đóng ứng dụng khác, tắt Ollama (pkill ollama) trước bước 1.
# Biến: SIZE(8b) ADAPTER ADAPTER_FILE OUT OLLAMA_NAME BASE_OLLAMA FUSE_BASE QUANT(Q4_K_M)
#   FUSE_BASE mặc định mlx-community/Qwen3-8B-4bit (đúng base đã train, de-quantize); đặt FUSE_BASE=Qwen/Qwen3-8B để gộp lên bf16 gốc (cần tải 16GB và nhiều RAM hơn)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SIZE=${SIZE:-8b}; SIZE_UP=$(echo "$SIZE" | tr "a-z" "A-Z")
ADAPTER=${ADAPTER:-adapters/qwen3_${SIZE}_lora}
FUSE_BASE=${FUSE_BASE:-mlx-community/Qwen3-${SIZE_UP}-4bit}
QUANT=${QUANT:-Q4_K_M}
OUT=${OUT:-models_ft/qwen3_${SIZE}_ft}            # NGOÀI training_code/ để Dockerfile (COPY training_code) không chép nhầm file nặng
OLLAMA_NAME=${OLLAMA_NAME:-datamate-qwen3:${SIZE}-ft}
BASE_OLLAMA=${BASE_OLLAMA:-qwen3:${SIZE}}
. "$(dirname "$0")/lib_cong_cu.sh"
step() { echo; echo "=== [A $1] $(date +%H:%M:%S) ==="; }

command -v ollama >/dev/null || { echo "Chưa có ollama."; exit 1; }
[ -f "$ADAPTER/adapters.safetensors" ] || { echo "Không thấy $ADAPTER/adapters.safetensors (chạy train_lora.sh trước)."; exit 1; }
ollama list | awk 'NR>1{print $1}' | grep -qx "$BASE_OLLAMA" || { echo "Chưa có $BASE_OLLAMA: ollama pull $BASE_OLLAMA"; exit 1; }
mkdir -p "$OUT"
if [ -n "${ADAPTER_FILE:-}" ]; then      # dùng checkpoint cụ thể
  TMPAD="$OUT/adapter_tmp"; rm -rf "$TMPAD"; mkdir -p "$TMPAD"
  cp "$ADAPTER/adapter_config.json" "$TMPAD/"; cp "$ADAPTER_FILE" "$TMPAD/adapters.safetensors"; ADAPTER="$TMPAD"
fi
chuan_bi_cong_cu can_quantize
FREE_GB=$(df -g . | awk 'NR==2{print $4}'); echo "Ổ đĩa trống: ${FREE_GB} GB (cần ~45GB cho đường A)"

step "1/5 mlx_lm.fuse lên $FUSE_BASE"
DEQ=(); case "$FUSE_BASE" in *bit*) DEQ=(--dequantize);; esac
rm -rf "$OUT/hf"
"$PY" -m mlx_lm.fuse --model "$FUSE_BASE" --adapter-path "$ADAPTER" --save-path "$OUT/hf" "${DEQ[@]+"${DEQ[@]}"}"
ls -la "$OUT/hf" | head -20
# luôn chép tokenizer + chat template gốc của Qwen để GGUF mang đúng thẻ đặc biệt
"$PY" - "$FUSE_BASE" "$OUT/hf" <<'PYEOF'
import re, sys
from transformers import AutoTokenizer
src, dst = sys.argv[1], sys.argv[2]
if "bit" in src:
    src = "Qwen/" + re.sub(r"-\d+bit$", "", src.split("/")[-1])
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
print("đã chép tokenizer từ", src)
PYEOF

step "2/5 convert sang GGUF bf16"
"$VENV_GGUF/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$OUT/hf" --outtype bf16 --outfile "$OUT/model-bf16.gguf"

step "3/5 lượng tử hoá $QUANT"
"$QUANTIZE" "$OUT/model-bf16.gguf" "$OUT/model.gguf" "$QUANT"
ls -lh "$OUT/model.gguf"
echo "(Xoá $OUT/hf và $OUT/model-bf16.gguf sau khi kiểm xong để lấy lại ~32GB.)"

step "4/5 Modelfile từ $BASE_OLLAMA"
ollama show "$BASE_OLLAMA" --modelfile > "$OUT/Modelfile.base"
{ echo "FROM ./model.gguf"; grep -v '^FROM ' "$OUT/Modelfile.base" | grep -v '^# Modelfile generated'; } > "$OUT/Modelfile"
head -c 1200 "$OUT/Modelfile"; echo

step "5/5 ollama create $OLLAMA_NAME"
curl -s http://127.0.0.1:11434/api/tags >/dev/null || { open -a Ollama 2>/dev/null || true; sleep 12; }
( cd "$OUT" && ollama create "$OLLAMA_NAME" -f Modelfile )
echo "Xong đường A: $OLLAMA_NAME . Kiểm: $PY training_code/finetune/kiem_model_ollama.py $OLLAMA_NAME --so-voi $BASE_OLLAMA"
echo "File cho Docker: $OUT/model.gguf + $OUT/Modelfile"
