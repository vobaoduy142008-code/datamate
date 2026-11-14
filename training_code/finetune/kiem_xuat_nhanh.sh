#!/bin/bash
# Kiểm TRƯỚC khi train thật: dùng adapter SMOKE (10 bước) thử xuất theo đường B rồi A, gọi 1-3 câu qua ollama, báo đường nào chạy.
#   bash training_code/finetune/kiem_xuat_nhanh.sh              # tự chạy SMOKE=1 train nếu chưa có adapter smoke (~10-20 phút gồm tải model)
#   ONLY=B bash ...                                               # chỉ thử đường B (nhanh, ~5 phút); ONLY=A chỉ thử đường A (~30-40 phút)
# Ghi kết quả vào models_ft/duong_xuat.txt (dòng 1: A | B | NONE; xuat_model.sh đọc file này).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=training_code/finetune
PY=.venv/bin/python
SIZE=${SIZE:-8b}
SM=adapters/smoke_qwen3_${SIZE}
ONLY=${ONLY:-AB}
mkdir -p models_ft

if [ ! -f "$SM/adapters.safetensors" ]; then
  echo ">>> Chưa có adapter smoke -> chạy SMOKE=1 train_lora.sh"
  SMOKE=1 SIZE=$SIZE bash $D/train_lora.sh || { echo "SMOKE train lỗi - dừng."; exit 1; }
fi
curl -s http://127.0.0.1:11434/api/tags >/dev/null || { open -a Ollama 2>/dev/null; sleep 12; }
ollama list | awk 'NR>1{print $1}' | grep -qx "qwen3:${SIZE}" || ollama pull "qwen3:${SIZE}"

RES_B="chưa thử"; RES_A="chưa thử"
if [[ "$ONLY" == *B* ]]; then
  echo "################ THỬ ĐƯỜNG B ################"
  if ADAPTER=$SM OUT=models_ft/smoke_B OLLAMA_NAME=datamate-smoke-b bash $D/xuat_adapter_gguf.sh \
     && $PY $D/kiem_model_ollama.py datamate-smoke-b --so-voi "qwen3:${SIZE}"; then RES_B="CHẠY ĐƯỢC"; else RES_B="LỖI"; fi
fi
if [[ "$ONLY" == *A* ]]; then
  echo "################ THỬ ĐƯỜNG A ################"
  if ADAPTER=$SM OUT=models_ft/smoke_A OLLAMA_NAME=datamate-smoke-a bash $D/fuse_va_gguf.sh \
     && $PY $D/kiem_model_ollama.py datamate-smoke-a --so-voi "qwen3:${SIZE}"; then RES_A="CHẠY ĐƯỢC"; else RES_A="LỖI"; fi
fi

echo; echo "================ TỔNG KẾT ================"
echo "Đường B (ADAPTER GGUF trên qwen3:${SIZE}): $RES_B"
echo "Đường A (fuse + GGUF Q4_K_M)           : $RES_A"
if   [ "$RES_A" = "CHẠY ĐƯỢC" ]; then KQ=A
elif [ "$RES_B" = "CHẠY ĐƯỢC" ]; then KQ=B
else KQ=NONE; fi
echo "$KQ" > models_ft/duong_xuat.txt
echo "-> Ghi models_ft/duong_xuat.txt = $KQ (ưu tiên A khi cả hai chạy: là model GGUF thường, chắc chắn hơn trong Docker)."
[ "$KQ" = NONE ] && echo "Cả hai lỗi: xem log phía trên + README mục 7 trước khi train thật."
echo "Dọn model thử:  ollama rm datamate-smoke-a datamate-smoke-b ; rm -rf models_ft/smoke_A models_ft/smoke_B   (giải phóng ~35GB)"
