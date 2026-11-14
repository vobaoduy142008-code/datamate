#!/bin/bash
# Xuất model THẬT sau khi train xong: tự chọn đường A (fuse+GGUF) hoặc B (adapter GGUF), kiểm bằng ollama, tự quay sang đường còn lại nếu hỏng.
#   bash training_code/finetune/xuat_model.sh            # MODE=auto: đọc models_ft/duong_xuat.txt (do kiem_xuat_nhanh.sh ghi); chưa có -> thử A rồi B
#   MODE=A|B bash ...                                     # ép một đường
#   ADAPTER_FILE=adapters/qwen3_8b_lora/0000300_adapters.safetensors bash ...   # dùng checkpoint cụ thể
# Kết quả: model Ollama datamate-qwen3:8b-ft + models_ft/duong_xuat_cuoi.txt (đường nào, file nào dùng cho Docker).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=training_code/finetune
SIZE=${SIZE:-8b}
NAME=${OLLAMA_NAME:-datamate-qwen3:${SIZE}-ft}
MODE=${MODE:-auto}
case "$MODE" in
  A) ORDER="A";; B) ORDER="B";;
  *) ORDER="A B"
     if [ -f models_ft/duong_xuat.txt ]; then
       case "$(head -n1 models_ft/duong_xuat.txt)" in B) ORDER="B A";; NONE) echo "kiem_xuat_nhanh.sh báo cả hai đường lỗi - vẫn thử A rồi B.";; esac
     fi;;
esac
echo "Thứ tự thử: $ORDER (model: $NAME)"
for P in $ORDER; do
  echo "################ ĐƯỜNG $P ################"
  if [ "$P" = "A" ]; then OLLAMA_NAME=$NAME bash $D/fuse_va_gguf.sh; else OLLAMA_NAME=$NAME bash $D/xuat_adapter_gguf.sh; fi
  RC=$?
  if [ $RC -eq 0 ] && .venv/bin/python $D/kiem_model_ollama.py "$NAME" --so-voi "qwen3:${SIZE}"; then
    if [ "$P" = "A" ]; then F="models_ft/qwen3_${SIZE}_ft (model.gguf + Modelfile)"; else F="models_ft/qwen3_${SIZE}_ft_B (lora.gguf + Modelfile; base qwen3:${SIZE})"; fi
    printf 'DUONG=%s\nOLLAMA_NAME=%s\nFILE=%s\n' "$P" "$NAME" "$F" | tee models_ft/duong_xuat_cuoi.txt
    echo "THÀNH CÔNG đường $P. Bước tiếp: bash $D/danh_gia_nhanh.sh"
    exit 0
  fi
  echo "Đường $P KHÔNG đạt (mã $RC)."
done
echo "Cả hai đường đều không đạt - xem README mục 7."
exit 1
