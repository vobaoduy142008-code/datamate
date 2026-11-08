#!/bin/bash
# Tải trọng số Qwen3-14B (bản gốc bf16, khoảng 30 GB) từ Hugging Face vào models/Qwen3-14B.
# Chạy trong thư mục datamate (Ubuntu/WSL hoặc Mac):   bash scripts/tai_model.sh
# Bị ngắt giữa chừng thì chạy lại đúng lệnh này: chỉ tải tiếp phần còn thiếu.
set -e
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-14B}
DEST=${DEST:-models/Qwen3-14B}
if ! command -v uv > /dev/null 2>&1; then
    echo "Chưa có uv -> cài uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run --no-project --with "huggingface_hub[hf_transfer]" python - "$MODEL_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
mid, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=mid, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "LICENSE*", "README*"])
print("Đã tải xong", mid, "->", dest)
PY
du -sh "$DEST"
ls "$DEST"
