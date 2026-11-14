#!/bin/bash
# =============================================================================
# DataMate – Zalo AI Challenge 2026: lệnh chạy chính của Docker (BTC gọi /code/predict.sh, không tham số)
#   Đọc   : /data   (chỉ đọc: questions.jsonl, rulebook.md, calendar.csv, data/<shop>/...)
#   Ghi   : /result/submission.jsonl  (ghi ngay sau mỗi câu)
#           /result/logs/<id>.json + /result/logs/log.jsonl  (id, answer, elapsed_ms, thinking, code, output)
# Model: Qwen3-8B (bản qwen3:8b của Ollama, đã có sẵn trong image), chạy bằng Ollama trên GPU, mạng tắt.
# Các bước:
#   1. Bật Ollama ở nền (chỉ nghe 127.0.0.1), trong lúc đó đọc trước dữ liệu mọi shop vào cache (/tmp).
#   2. Chờ Ollama sẵn sàng, nạp model.
#   3. Chạy agent từng câu (greedy, seed 0) với hạn chót tổng để không quá 6 giờ.
# Chạy thử không cần GPU / model:  docker run -e LLM_BACKEND=mock ...
# =============================================================================
set -u

T0=$(date +%s)
DATA_DIR=${DATA_DIR:-/data}
OUT_DIR=${OUT_DIR:-/result}
TOTAL_SECONDS=${TOTAL_SECONDS:-20400}     # 5 giờ 40 phút tính từ lúc container bắt đầu (BTC cho 6 giờ)
WORKERS=${AGENT_WORKERS:-1}               # 1 câu một lúc: kết quả giống hệt khi chạy trên máy của đội

export LLM_BACKEND=${LLM_BACKEND:-ollama}
export LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:11434}
export LLM_MODEL=${LLM_MODEL:-qwen3:8b}
export LLM_THINK=${LLM_THINK:-0}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-0}
export LLM_SEED=${LLM_SEED:-0}
export LLM_NUM_CTX=${LLM_NUM_CTX:-32768}
export LLM_TIMEOUT=${LLM_TIMEOUT:-300}
export AGENT_TIME_BUDGET=${AGENT_TIME_BUDGET:-300}
export AGENT_VERBOSE=0
# v5.0: model Qwen3-8B điều phối công cụ (chọn công cụ, chuẩn hoá câu, kiểm duyệt); câu không có công cụ -> bỏ qua
export AGENT_MODE=${AGENT_MODE:-dieu_phoi}
export AGENT_ONLY_FORMULA=${AGENT_ONLY_FORMULA:-1}
export DATAMATE_CACHE=${DATAMATE_CACHE:-/tmp/datamate_cache}
export PYTHONUNBUFFERED=1
# Ollama: chỉ trong máy, giữ model trên GPU suốt lúc chạy, 1 yêu cầu một lúc
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=${OLLAMA_MODELS:-/models/ollama}
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NOPRUNE=1
# chỉ dùng 1 GPU (máy chấm có 2): chạy lại cho cùng kết quả như máy của đội (1 GPU)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=${PY:-/opt/agent/bin/python}          # Python riêng của agent (pandas...), cài sẵn trong image
[ -x "$PY" ] || PY=python3
SRC=$(cd "$(dirname "$0")" && pwd)/src

mkdir -p "$OUT_DIR/logs"
echo "[predict] bắt đầu $(date)  backend=$LLM_BACKEND  model=$LLM_MODEL  workers=$WORKERS"

# ---- 1. bật Ollama ------------------------------------------------------------
SERVER_PID=""
if [ "$LLM_BACKEND" = "ollama" ]; then
    nvidia-smi -L 2>/dev/null || echo "[predict] (không thấy nvidia-smi)"
    ollama serve > "$OUT_DIR/logs/ollama.log" 2>&1 &
    SERVER_PID=$!
fi

# ---- 2. đọc trước dữ liệu (song song với lúc bật server) -----------------------
"$PY" "$SRC/main.py" --data "$DATA_DIR" --out "$OUT_DIR" --prewarm 2>&1 | tail -n 3

# ---- 3. chờ Ollama sẵn sàng và nạp model ---------------------------------------
if [ -n "$SERVER_PID" ]; then
    READY=0
    for i in $(seq 1 120); do                     # chờ tối đa 10 phút
        if curl -sf "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then READY=1; break; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[predict] Ollama đã dừng do lỗi"; break; fi
        sleep 5
    done
    if [ "$READY" = "1" ]; then
        # nạp model lên GPU trước (câu đầu không bị chậm), cùng num_ctx với lúc chạy
        curl -sf "$LLM_BASE_URL/api/generate" -d "{\"model\":\"$LLM_MODEL\",\"prompt\":\"\",\"keep_alive\":-1,\"options\":{\"num_ctx\":$LLM_NUM_CTX}}" > /dev/null 2>&1
        echo "[predict] Ollama sẵn sàng sau $(( $(date +%s) - T0 ))s"
        curl -s "$LLM_BASE_URL/api/ps" | head -c 400; echo
    else
        echo "[predict] !! Ollama KHÔNG bật được, xem $OUT_DIR/logs/ollama.log. Mọi câu sẽ là abstain."
        tail -n 30 "$OUT_DIR/logs/ollama.log"
    fi
fi

# ---- 4. chạy agent -------------------------------------------------------------
DEADLINE=$(( T0 + TOTAL_SECONDS ))
"$PY" "$SRC/main.py" --data "$DATA_DIR" --out "$OUT_DIR" \
    --workers "$WORKERS" --deadline "$DEADLINE" --quiet
RC=$?

[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
echo "[predict] xong sau $(( $(date +%s) - T0 ))s (mã thoát $RC)"
exit 0
