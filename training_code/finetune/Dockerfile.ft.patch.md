# Đề xuất sửa Docker để dùng model đã fine-tune (CHƯA áp vào `Dockerfile` / `predict.sh` thật)

Điều kiện: bản ft đã qua `danh_gia_nhanh.sh` và **tốt hơn rõ** model gốc. Hạn nộp Docker: 23:59 16/11 — chừa ít nhất 3 giờ cho build + chạy thử.
Chọn đường theo `models_ft/duong_xuat_cuoi.txt` (do `xuat_model.sh` ghi): **A** (GGUF đã gộp) hoặc **B** (GGUF-LoRA trên `qwen3:8b`).

## 1. File nặng NGOÀI `training_code/`, và `.dockerignore`
`Dockerfile` có `COPY training_code /code/training_code`; file model nằm ở `models_ft/` (ngoài `training_code/`) nên không bị chép thừa. Tạo/sửa `.dockerignore`:
```
adapters
.venv
.venv-gguf
llama.cpp
outputs
models_ft/*/hf
models_ft/*/model-bf16.gguf
models_ft/*/base_cfg
models_ft/smoke_*
```
(Giữ `models_ft/qwen3_8b_ft/{model.gguf,Modelfile}` cho A; `models_ft/qwen3_8b_ft_B/{lora.gguf,Modelfile}` cho B.)

## 2A. Dockerfile — ĐƯỜNG A (GGUF đã gộp): thay đoạn `ollama pull` (bước 3)
Thay
```dockerfile
ARG MODEL=qwen3:8b
RUN (ollama serve > /tmp/ollama_build.log 2>&1 &) && sleep 8 \
 && ollama pull ${MODEL} && ollama list && ollama show ${MODEL} --modelfile | head -n 5
```
bằng (bind-mount BuildKit: GGUF không nằm lại trong layer; cần BuildKit, mặc định ở Docker mới):
```dockerfile
ARG MODEL_FT=datamate-qwen3:8b-ft
RUN --mount=type=bind,source=models_ft/qwen3_8b_ft,target=/ft \
    (ollama serve > /tmp/ollama_build.log 2>&1 &) \
 && for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 2; done \
 && cd /ft && ollama create ${MODEL_FT} -f Modelfile \
 && ollama list && ollama show ${MODEL_FT} --modelfile | head -n 8
```
- `Modelfile` có `FROM ./model.gguf` nên phải `cd /ft`. Không cần giữ `qwen3:8b` gốc (tiết kiệm ~5 GB); muốn dự phòng thì thêm `ollama pull qwen3:8b` (+5,2 GB).

## 2B. Dockerfile — ĐƯỜNG B (ADAPTER GGUF): giữ `ollama pull`, thêm một bước
```dockerfile
ARG MODEL=qwen3:8b
ARG MODEL_FT=datamate-qwen3:8b-ft
RUN --mount=type=bind,source=models_ft/qwen3_8b_ft_B,target=/ft \
    (ollama serve > /tmp/ollama_build.log 2>&1 &) && sleep 8 \
 && ollama pull ${MODEL} \
 && cd /ft && ollama create ${MODEL_FT} -f Modelfile \
 && ollama list
```
(`Modelfile` B: `FROM qwen3:8b` + `ADAPTER ./lora.gguf`.) Image gần như bằng bản gốc (+ vài chục MB). **Rủi ro:** phải xác nhận Ollama **trong image** (đúng `OLLAMA_VERSION`) nạp được adapter qwen3 — kiểm bằng `docker run --rm --entrypoint ... ` (mục 5) với GPU hoặc CPU; B chỉ được kiểm trên Metal ở Mac.

## 3. `predict.sh` (một dòng, cả A và B)
```bash
export LLM_MODEL=${LLM_MODEL:-datamate-qwen3:8b-ft}     # đang là qwen3:8b
```
Giữ `LLM_THINK=0`, `OLLAMA_NUM_PARALLEL=1`, temperature 0, seed 0. `src/llm.py` đọc `LLM_MODEL` từ biến môi trường nên **không cần sửa**. Sửa chú thích đầu `Dockerfile`/`predict.sh`/README nộp bài: "Qwen3-8B (Apache 2.0) + LoRA SFT, GGUF Q4_K_M (A) / adapter GGUF (B)"; dữ liệu train sinh từ `data/train` (nằm trong `data_manifest.json`). Ollama trong image phải **cùng phiên bản** với Mac nơi đã kiểm (`--build-arg OLLAMA_VERSION=...`).

## 4. Ước tính kích thước image
| Thành phần | A | B |
|---|---|---|
| `ollama/ollama` (kèm CUDA) | ~4–5 GB | ~4–5 GB |
| Python 3.12 + pandas/numpy/openpyxl | ~0,5–0,8 GB | ~0,5–0,8 GB |
| Model | GGUF Q4_K_M 8B ~5,0 GB | `qwen3:8b` ~5,2 GB + lora.gguf (rank 16, 8 tầng) ~0,05 GB |
| `data/train` + code + docs | xem `du -sh data/train` | như A |
| **Tổng** | **~10–12 GB** | **~10–12 GB** |
Nếu giữ thêm model gốc: +5,2 GB. Đo thật bằng `docker images` / `docker history`.

## 5. Kiểm sau khi build
```bash
docker build --platform linux/amd64 --build-arg OLLAMA_VERSION=<số> -t zaic2026_datamate_<ten_doi>:ft .
docker run --rm --entrypoint ollama zaic2026_datamate_<ten_doi>:ft list     # thấy datamate-qwen3:8b-ft
# chạy vài câu public như cách chạy cũ, đối chiếu với bản Mac (chấp nhận lệch nhỏ: CUDA ≠ Metal).
```
Đường B: nếu `ollama create` hoặc lần gọi đầu báo lỗi adapter/arch → quay về A (hoặc nộp model gốc).
