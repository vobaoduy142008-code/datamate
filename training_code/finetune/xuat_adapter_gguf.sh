#!/bin/bash
# ĐƯỜNG B (không fuse): adapter MLX -> PEFT -> lora.gguf (convert_lora_to_gguf.py) -> Modelfile "FROM qwen3:8b + ADAPTER ./lora.gguf" -> ollama create.
# Nhanh (vài phút) và nhẹ (file lora ~ vài chục MB), NHƯNG việc Ollama nạp GGUF-LoRA cho qwen3 chưa chắc chạy -> kiểm bằng kiem_xuat_nhanh.sh.
# Biến: SIZE(8b) ADAPTER ADAPTER_FILE OUT OLLAMA_NAME BASE_OLLAMA(qwen3:8b) BASE_HF(Qwen/Qwen3-8B)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SIZE=${SIZE:-8b}; SIZE_UP=$(echo "$SIZE" | tr "a-z" "A-Z")
ADAPTER=${ADAPTER:-adapters/qwen3_${SIZE}_lora}
OUT=${OUT:-models_ft/qwen3_${SIZE}_ft_B}
OLLAMA_NAME=${OLLAMA_NAME:-datamate-qwen3:${SIZE}-ft}
BASE_OLLAMA=${BASE_OLLAMA:-qwen3:${SIZE}}
BASE_HF=${BASE_HF:-Qwen/Qwen3-${SIZE_UP}}
. "$(dirname "$0")/lib_cong_cu.sh"
step() { echo; echo "=== [B $1] $(date +%H:%M:%S) ==="; }

command -v ollama >/dev/null || { echo "Chưa có ollama."; exit 1; }
[ -f "$ADAPTER/adapters.safetensors" ] || { echo "Không thấy $ADAPTER/adapters.safetensors"; exit 1; }
ollama list | awk 'NR>1{print $1}' | grep -qx "$BASE_OLLAMA" || { echo "Chưa có $BASE_OLLAMA: ollama pull $BASE_OLLAMA"; exit 1; }
mkdir -p "$OUT"; chuan_bi_cong_cu

step "1/4 adapter MLX -> PEFT"
rm -rf "$OUT/peft"
FILEARG=(); [ -n "${ADAPTER_FILE:-}" ] && FILEARG=(--file "$ADAPTER_FILE")
"$PY" "$(dirname "$0")/adapter_sang_peft.py" "$ADAPTER" "$OUT/peft" --base-id "$BASE_HF" "${FILEARG[@]+"${FILEARG[@]}"}"

step "2/4 tải config của $BASE_HF (chỉ file nhỏ, không tải trọng số)"
hf_tai "$BASE_HF" --include "*.json" "merges.txt" "vocab.json" --local-dir "$OUT/base_cfg"

step "3/4 convert_lora_to_gguf.py -> lora.gguf"
"$VENV_GGUF/bin/python" "$LLAMA_CPP/convert_lora_to_gguf.py" --base "$OUT/base_cfg" --outfile "$OUT/lora.gguf" --outtype f16 "$OUT/peft"
ls -lh "$OUT/lora.gguf"

step "4/4 Modelfile + ollama create $OLLAMA_NAME"
printf 'FROM %s\nADAPTER ./lora.gguf\n' "$BASE_OLLAMA" > "$OUT/Modelfile"   # TEMPLATE/PARAMETER thừa hưởng từ $BASE_OLLAMA
cat "$OUT/Modelfile"
curl -s http://127.0.0.1:11434/api/tags >/dev/null || { open -a Ollama 2>/dev/null || true; sleep 12; }
( cd "$OUT" && ollama create "$OLLAMA_NAME" -f Modelfile )
echo "Xong đường B: $OLLAMA_NAME . Kiểm: $PY training_code/finetune/kiem_model_ollama.py $OLLAMA_NAME --so-voi $BASE_OLLAMA"
