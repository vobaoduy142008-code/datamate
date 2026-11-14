#!/bin/bash
# Hàm dùng chung (được `source` bởi fuse_va_gguf.sh, xuat_adapter_gguf.sh). Không chạy trực tiếp.
# chuan_bi_cong_cu [can_quantize]  -> đặt LLAMA_CPP, QUANTIZE, VENV_GGUF; tự clone llama.cpp / tạo venv nếu thiếu.
LLAMA_CPP=${LLAMA_CPP:-$PWD/llama.cpp}
VENV_GGUF=${VENV_GGUF:-.venv-gguf}

chuan_bi_cong_cu() {
  if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "Chưa có llama.cpp -> clone vào $LLAMA_CPP (cần mạng)"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP" || { echo "clone lỗi"; return 1; }
  fi
  if [ "${1:-}" = "can_quantize" ]; then
    QUANTIZE=$(command -v llama-quantize || true)
    if [ -z "$QUANTIZE" ] && [ -x "$LLAMA_CPP/build/bin/llama-quantize" ]; then QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"; fi
    if [ -z "$QUANTIZE" ] && command -v brew >/dev/null; then
      echo "Cài llama.cpp bằng brew (có llama-quantize)"; brew install llama.cpp && QUANTIZE=$(command -v llama-quantize || true)
    fi
    [ -n "$QUANTIZE" ] || { echo "Không có llama-quantize. Chạy: brew install llama.cpp  (hoặc: cd $LLAMA_CPP && cmake -B build && cmake --build build -j --target llama-quantize)"; return 1; }
  fi
  if [ ! -x "$VENV_GGUF/bin/python" ]; then
    echo "Tạo $VENV_GGUF + cài thư viện cho convert_*.py (vài phút, tải torch CPU)..."
    uv venv --python 3.12 "$VENV_GGUF" || return 1
    uv pip install --python "$VENV_GGUF/bin/python" -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt" || return 1
  fi
  return 0
}

hf_tai() {   # hf_tai <repo> [tham số thêm của hf download]
  if [ -x .venv/bin/hf ]; then .venv/bin/hf download "$@"; else .venv/bin/huggingface-cli download "$@"; fi
}
