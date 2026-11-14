# Fine-tune Qwen3-8B (SFT LoRA) bằng MLX trên MacBook Pro M1 Pro 16 GB, rồi đưa vào Ollama / Docker

Mục tiêu (`docs/kien_truc_v6.md` mục 3): dạy Qwen3-8B sinh kế hoạch + DSL tốt hơn bằng dữ liệu đã kiểm chứng; dùng bản `datamate-qwen3:8b-ft` thay `qwen3:8b` trong Docker. Model ≤ 15B, chạy offline.

> Quy tắc đội: **chưa chạy và chưa đánh giá thì không được ghi "đã fine-tune"** vào báo cáo/README nộp bài. Nếu bản ft không tốt hơn, nộp bản `qwen3:8b` gốc (Dockerfile hiện tại).
> Mọi lệnh chạy trong Terminal, **từ thư mục `datamate`**. Máy mục tiêu: M1 Pro, 16 GB RAM, còn 230 GB ổ đĩa. Các con số thời gian/RAM dưới đây là **ước tính, chưa đo**; bước 4 (SMOKE) sẽ đo số thật.

## 0. Thứ tự lệnh (lịch gợi ý: hạn Docker 23:59 16/11, train thật bắt đầu sáng 15/11)

| # | Việc | Lệnh | Thời gian ước tính (M1 Pro) |
|---|---|---|---|
| 1 | Kiểm máy, cài `mlx-lm` | mục 2–3 | 5 phút |
| 2 | Tải model 4-bit (4,5 GB) và `ollama pull qwen3:8b` | mục 4 | 10–30 phút |
| 3 | Dữ liệu | `chuan_bi_du_lieu.py` | < 1 phút |
| 4 | **SMOKE**: đo tốc độ/RAM thật | `SMOKE=1 bash training_code/finetune/train_lora.sh` | 5–10 phút |
| 5 | **Kiểm đường xuất** (tối hôm trước hoặc sáng sớm) | `bash training_code/finetune/kiem_xuat_nhanh.sh` | B ~5–10 phút; A ~40–70 phút (có swap) |
| 6 | **Train thật** | `bash training_code/finetune/train_lora.sh` | 900 ví dụ x 1 epoch: ~1,5–3 giờ |
| 7 | Xuất model thật | `bash training_code/finetune/xuat_model.sh` | A ~40–70 phút / B ~10 phút |
| 8 | Đánh giá 300 câu so với gốc | `danh_gia_nhanh.sh` | như `chay_model_300.command` (vài giờ) |
| 9 | Docker | `Dockerfile.ft.patch.md` | build ~30 phút + chạy thử |

Mốc an toàn: nếu đến **trưa 16/11** chưa có bản ft đánh giá xong và tốt hơn rõ, **dừng, nộp bản gốc**. Train có checkpoint mỗi 50 bước nên có thể Ctrl+C bất kỳ lúc nào và lấy checkpoint gần nhất (`ADAPTER_FILE=...`).

## 1. Yêu cầu RAM / ổ đĩa

| Cấu hình | Trọng số | Đỉnh RAM khi train (batch 1, 1536 token, 8 tầng LoRA, `--grad-checkpoint`) |
|---|---|---|
| Qwen3-8B 4-bit (QLoRA) — **dùng cái này** | ~4,5 GB | ~8–11 GB (ước tính) |
| Qwen3-4B 4-bit (dự phòng, `SIZE=4b`) | ~2,5 GB | ~5–7 GB |

16 GB là **vừa đủ** cho 8B: macOS tự dùng ~3–4 GB và GPU Metal chỉ được cấp một phần RAM (thường ~10–11 GB trên máy 16GB). Vì vậy trước khi train:
- Tắt Ollama (`pkill ollama` hoặc Quit ở thanh menu) — nó giữ model trong RAM. Đóng Chrome, VS Code, Docker Desktop. Cắm sạc.
- Nếu báo "Insufficient Memory"/máy đơ: `NUM_LAYERS=4 MAXLEN=1024 SMOKE=1 bash ...`, hoặc (macOS 14+) nâng giới hạn GPU: `sudo sysctl iogpu.wired_limit_mb=13000` (tự reset khi khởi động lại; dùng thận trọng), hoặc đổi `SIZE=4b`.
- Đường xuất A (fuse + de-quantize) tạo model bf16 ~16 GB nên **sẽ dùng swap** trên máy 16 GB: chậm nhưng chạy được; khi đó đóng hết ứng dụng khác.
- Ổ đĩa: đường A tạm thời ~45 GB (HF 16GB + GGUF bf16 16GB + GGUF Q4 5GB + llama.cpp + torch); đủ với 230 GB. Sau khi xong có thể xoá `models_ft/*/hf` và `*/model-bf16.gguf`.

## 2. Kiểm tra máy
```bash
uname -m                                   # phải ra: arm64
sw_vers -productVersion                    # macOS >= 14
sysctl hw.memsize                          # 17179869184 = 16 GB
sysctl -n hw.memsize | awk '{printf "%.0f GB RAM\n", $1/1073741824}'
df -h . | tail -1                          # còn >= 60 GB trống
ls .venv/bin/python                        # .venv (tạo bằng uv, KHÔNG có pip)
```

## 3. Cài `mlx-lm`
`.venv` tạo bằng `uv` nên **không có `pip`**:
```bash
uv pip install --python .venv/bin/python mlx-lm
.venv/bin/python -c "import mlx_lm, mlx.core as mx; print('mlx-lm', mlx_lm.__version__, mx.default_device())"   # phải ra Device(gpu, 0)
```
Cần mlx-lm >= 0.24 (có Qwen3). Cũ hơn: thêm `-U`. Kiểm tên flag nếu bản mới đổi: `.venv/bin/python -m mlx_lm.lora --help`.

## 4. Tải model, chuẩn bị dữ liệu, SMOKE
```bash
.venv/bin/hf download mlx-community/Qwen3-8B-4bit      # 4,5 GB (nếu "hf: command not found": .venv/bin/huggingface-cli download ...)
ollama pull qwen3:8b                                    # để lấy Modelfile mẫu và làm base cho đường B
.venv/bin/python training_code/finetune/chuan_bi_du_lieu.py        # cần training_code/dsl/sft_raw.jsonl; thêm --tokenizer Qwen/Qwen3-8B để đếm token chính xác + in mẫu chat template
pkill ollama; SMOKE=1 bash training_code/finetune/train_lora.sh   # 10 bước, 20 ví dụ
```
Chuẩn bị dữ liệu: loại ví dụ trùng eval300/public (theo id và nội dung), chia **theo shop** 90/10 → `data/train.jsonl`, `data/valid.jsonl` (tên giống thư mục `data/train/` nhưng là file; muốn tách thì `--out-dir data/sft` và `DATA=data/sft`), loại ví dụ > 1536 token.
Mẫu đã áp chat template (khi dùng `--tokenizer`) phải có `<think>\n\n</think>` rỗng ở lượt assistant cuối (= chế độ không suy nghĩ, khớp `LLM_THINK=0`).

SMOKE in `TỔNG KẾT`: token/giây, RAM đỉnh, và phút ước tính cho 300/600/900/1200 ví dụ. **Dùng số này để chọn `MAX_EXAMPLES`**: chọn sao cho train thật xong trong ≲ 3 giờ. Nếu RAM đỉnh > ~12 GB hoặc máy swap nặng: giảm `NUM_LAYERS`/`MAXLEN` rồi SMOKE lại.

## 5. Kiểm đường xuất (làm TRƯỚC khi train thật)
```bash
bash training_code/finetune/kiem_xuat_nhanh.sh
```
Dùng adapter SMOKE để thử xuất theo **B** (adapter → PEFT → `lora.gguf` → `FROM qwen3:8b` + `ADAPTER`) rồi **A** (fuse → GGUF Q4_K_M), gọi 3 câu từ `data/valid.jsonl` qua `ollama` (think=false) và kiểm JSON hợp lệ, không rò `<think>`. Cuối cùng in tổng kết đường nào chạy được và ghi `models_ft/duong_xuat.txt` (A | B | NONE). Chạy tách: `ONLY=B ...` (nhanh) hoặc `ONLY=A ...`. Script tự clone `llama.cpp`, tạo `.venv-gguf`, `brew install llama.cpp` nếu thiếu.

### Hai đường xuất và độ chắc chắn
- **Đường A – fuse + GGUF thường (chắc chắn nhất).** `mlx_lm.fuse --dequantize` lên `mlx-community/Qwen3-8B-4bit` (đúng base đã train) → HF bf16 → `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` → Modelfile `FROM ./model.gguf` với TEMPLATE/PARAMETER/RENDERER/PARSER chép từ `ollama show qwen3:8b --modelfile`. Ollama nạp GGUF Qwen3 đầy đủ bình thường, đây là đường tôi chắc chắn nhất. Nhược: chậm, dùng swap, lượng tử hoá lại lần nữa (Q4_K_M) nên lệch nhẹ so với lúc train.
- **Đường B – adapter GGUF (nhanh, nhỏ, nhưng KHÔNG chắc).** Ollama chính thức ghi hỗ trợ `ADAPTER` kiểu safetensors cho Llama/Mistral/Gemma; với GGUF-LoRA thì dùng cơ chế LoRA của llama.cpp nên về nguyên tắc chạy được với kiến trúc có trong llama.cpp, nhưng **tôi không tìm thấy/không nhớ tài liệu xác nhận cho qwen3**, và phụ thuộc phiên bản Ollama (qwen3 chạy ở engine mới của Ollama, engine mới có thể chưa hỗ trợ adapter hoặc tự rơi về engine cũ). Có thể gặp: lỗi lúc `ollama create`, lỗi lúc nạp model, hoặc nạp được nhưng adapter không có tác dụng. Vì thế phải kiểm (bước 5) và so với model gốc; đừng dùng B nếu chưa chạy thật trên **cùng phiên bản Ollama với Docker**. Ngoài ra adapter huấn luyện trên base MLX 4-bit nhưng áp lên Q4_K_M của Ollama (base lượng tử khác) → lệch nhẹ.
- Cách kiểm nhanh (cả hai đường): `.venv/bin/python training_code/finetune/kiem_model_ollama.py <tên-model> --so-voi qwen3:8b` hoặc `ollama run <tên-model> "Trả lời bằng JSON: {\"a\": 1+1}" --think=false`. Phải ra JSON hợp lệ, không có `<think>`. "Khác model gốc 0/3 câu" ở adapter SMOKE chưa kết luận được (10 bước quá yếu); với adapter thật phải có khác biệt và điểm eval300 thay đổi.
- Chọn tự động: `xuat_model.sh` đọc `models_ft/duong_xuat.txt`; chưa có thì thử A rồi B, kiểm xong mới nhận, hỏng đường này tự thử đường kia. Ưu tiên A khi cả hai chạy.

## 6. Train thật, xuất, đánh giá
```bash
pkill ollama
bash training_code/finetune/train_lora.sh               # mặc định: 8B-4bit, batch 1, 8 tầng, rank 16, max-seq 1536, 1 epoch, tối đa 900 ví dụ
# tuỳ chỉnh: MAX_EXAMPLES=600 NUM_LAYERS=16 EPOCHS=2 LR=1e-5 bash ...   RESUME=1 để train tiếp
```
- Log: `adapters/qwen3_8b_lora/train.log`; tổng kết: `tong_ket.txt` cùng thư mục. Checkpoint mỗi 50 bước (`0000050_adapters.safetensors`...). Dừng giữa chừng: Ctrl+C.
- Xem `grep "Val loss" adapters/qwen3_8b_lora/train.log`: giảm rồi phẳng = tốt; tăng lại = overfit → dùng checkpoint sớm hơn.
- Dữ liệu train là `head -n MAX_EXAMPLES data/train.jsonl` (đã xáo trộn) nên giới hạn thời gian không lệch theo shop.
```bash
ollama serve >/dev/null 2>&1 &                          # hoặc mở app Ollama
bash training_code/finetune/xuat_model.sh               # -> model Ollama datamate-qwen3:8b-ft + models_ft/duong_xuat_cuoi.txt
LIMIT=20 bash training_code/finetune/danh_gia_nhanh.sh  # thử 20 câu
bash training_code/finetune/danh_gia_nhanh.sh           # đủ 300 câu -> outputs/eval300_ft, so với outputs/eval300_model (qwen3:8b gốc; chạy ./chay_model_300.command trước nếu chưa có)
```
**Chỉ dùng bản ft nếu tổng đúng tăng rõ và không tụt ở mức nào**; ghi kết quả (kể cả không tốt hơn) vào ablation mục E của kiến trúc v6. Docker: xem `Dockerfile.ft.patch.md`.

## 7. Rủi ro và cách kiểm
1. **RAM không đủ trên 16GB** ("Insufficient Memory", swap nặng, máy đơ): mục 1; đo bằng `Peak mem`. Thứ tự giảm: `MAXLEN=1024` → `NUM_LAYERS=4` → `RANK=8` → `SIZE=4b` (nhớ đổi model trong Docker).
2. **Thời gian**: M1 Pro (16 GPU core, băng thông ~200 GB/s) chậm hơn M3/M4; ước tính 60–120 token/giây cho 8B (chưa đo). Luôn SMOKE trước; giới hạn bằng `MAX_EXAMPLES`; mốc cắt: trưa 16/11 chưa xong thì nộp bản gốc. Đường A thêm ~1 giờ do swap; có thể rút bằng B nếu B chạy.
3. **Mất chat template / thinking** sau convert: (a) `--tokenizer` ở bước 4 phải thấy `<think>\n\n</think>` rỗng; (b) `diff <(ollama show qwen3:8b --modelfile) <(ollama show datamate-qwen3:8b-ft --modelfile)` chỉ khác `FROM`/`ADAPTER`; (c) `kiem_model_ollama.py` (think=false không được có `<think>`/trường thinking). Dữ liệu SFT chỉ có chế độ không suy nghĩ nên `think=true` có thể yếu hơn; nếu `src/` còn gọi `think=True` thì đo riêng. Ollama mới có thể thêm `RENDERER`/`PARSER` cho qwen3: giữ nguyên (script A giữ; B thừa hưởng từ `FROM qwen3:8b`). **Dùng cùng phiên bản Ollama ở Mac và Docker.**
4. **Lệch MLX ↔ GGUF**: 4-bit (train) → bf16 → Q4_K_M (A) hoặc Q4_K_M của Ollama + LoRA (B) cho kết quả không giống từng token. Chỉ kết luận dựa trên bản **Ollama/GGUF** (đúng thứ nộp). Lệch lớn → `QUANT=Q5_K_M` (A).
5. **Đường B không chạy trong Ollama cho qwen3** (mục 5) → dùng A. **Mac (Metal) ≠ PC chấm (CUDA)**: B còn phụ thuộc đường code LoRA trên CUDA của Ollama; A không phụ thuộc.
6. **Phiên bản công cụ** (mlx-lm, llama.cpp, Ollama) đổi tên flag/kiến trúc: lỗi "unknown architecture qwen3" → cập nhật `mlx-lm` / `git -C llama.cpp pull`. Ghi lại `uv pip list`, `git -C llama.cpp rev-parse HEAD`, `ollama -v` cho báo cáo tái lập.
7. **Rò rỉ dữ liệu / quá khớp**: script loại câu eval300/public; nhìn mắt vài ví dụ trong `data/valid.jsonl`; 1 epoch, lr thấp, theo dõi Val loss. Với ~900 ví dụ, cải thiện có thể nhỏ — đó là lý do phải đo trên eval300.
