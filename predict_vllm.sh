#!/bin/bash
# (DỰ PHÒNG) predict.sh bản vLLM + Qwen3-14B, dùng với Dockerfile.vllm (đổi tên thành predict.sh khi build bản đó).
# =============================================================================
# DataMate – Zalo AI Challenge 2026: lệnh chạy chính của Docker (BTC gọi /code/predict.sh, không tham số)
#   Đọc   : /data   (chỉ đọc: questions.jsonl, rulebook.md, calendar.csv, data/<shop>/...)
#   Ghi   : /result/submission.jsonl  (ghi ngay sau mỗi câu)
#           /result/logs/<id>.json + /result/logs/log.jsonl  (id, answer, elapsed_ms, thinking, code, output)
# Các bước:
#   1. Bật vLLM (Qwen3-14B, chia đều cho mọi GPU) ở nền, cổng 8000, chỉ nghe trong máy (localhost).
#   2. Trong lúc model đang nạp: đọc trước dữ liệu mọi shop vào cache (/tmp).
#   3. Chờ vLLM sẵn sàng, rồi chạy agent song song nhiều câu, có hạn chót tổng để không quá 6 giờ.
# Chạy thử không cần GPU:  docker run -e LLM_BACKEND=mock ...   (bỏ qua bước 1)
# =============================================================================
set -u

T0=$(date +%s)
DATA_DIR=${DATA_DIR:-/data}
OUT_DIR=${OUT_DIR:-/result}
MODEL_DIR=${MODEL_DIR:-/models/Qwen3-14B}
SERVED_NAME=${SERVED_NAME:-qwen3-14b}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
TOTAL_SECONDS=${TOTAL_SECONDS:-20400}     # 5 giờ 40 phút tính từ lúc container bắt đầu (BTC cho 6 giờ)
WORKERS=${AGENT_WORKERS:-4}               # số câu chạy song song

export LLM_BACKEND=${LLM_BACKEND:-openai}
export LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}
export LLM_MODEL=${LLM_MODEL:-$SERVED_NAME}
export LLM_THINK=${LLM_THINK:-0}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-0}
export LLM_SEED=${LLM_SEED:-0}
export LLM_TIMEOUT=${LLM_TIMEOUT:-180}
export AGENT_TIME_BUDGET=${AGENT_TIME_BUDGET:-240}
export AGENT_VERBOSE=0
export DATAMATE_CACHE=${DATAMATE_CACHE:-/tmp/datamate_cache}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

AGENT_LIB=${AGENT_LIB:-/opt/agent/lib}   # thư viện riêng của agent (pandas...), tách khỏi thư viện của vLLM
run_agent() { PYTHONPATH="$AGENT_LIB${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"; }
SRC=$(cd "$(dirname "$0")" && pwd)/src

mkdir -p "$OUT_DIR/logs"
echo "[predict] bắt đầu $(date)  backend=$LLM_BACKEND  workers=$WORKERS"

# ---- 1. bật vLLM -------------------------------------------------------------
VLLM_PID=""
if [ "$LLM_BACKEND" = "openai" ]; then
    NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
    [ "${NGPU:-0}" -ge 1 ] || NGPU=1
    TP=${TP:-$NGPU}
    echo "[predict] GPU: $NGPU -> tensor-parallel $TP; model $MODEL_DIR"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 --port "$PORT" \
        --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization ${GPU_UTIL:-0.90} --dtype auto --seed 0 \
        --max-num-seqs ${MAX_SEQS:-16} ${VLLM_EXTRA:-} \
        > "$OUT_DIR/logs/vllm.log" 2>&1 &
    VLLM_PID=$!
fi

# ---- 2. đọc trước dữ liệu (song song với lúc model đang nạp) ------------------
run_agent "$SRC/main.py" --data "$DATA_DIR" --out "$OUT_DIR" --prewarm 2>&1 | tail -n 3

# ---- 3. chờ vLLM sẵn sàng ------------------------------------------------------
if [ -n "$VLLM_PID" ]; then
    READY=0
    for i in $(seq 1 360); do                     # chờ tối đa 30 phút
        if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=3)" > /dev/null 2>&1; then READY=1; break; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "[predict] vLLM đã dừng do lỗi"; break; fi
        sleep 5
    done
    if [ "$READY" = "1" ]; then
        echo "[predict] vLLM sẵn sàng sau $(( $(date +%s) - T0 ))s"
    else
        echo "[predict] !! vLLM KHÔNG bật được, xem $OUT_DIR/logs/vllm.log. Mọi câu sẽ là abstain."
        tail -n 30 "$OUT_DIR/logs/vllm.log"
    fi
fi

# ---- 4. chạy agent -------------------------------------------------------------
DEADLINE=$(( T0 + TOTAL_SECONDS ))
run_agent "$SRC/main.py" --data "$DATA_DIR" --out "$OUT_DIR" \
    --workers "$WORKERS" --deadline "$DEADLINE" --quiet
RC=$?

[ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null
echo "[predict] xong sau $(( $(date +%s) - T0 ))s (mã thoát $RC)"
exit 0
