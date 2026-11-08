# DataMate – Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bán hàng của cửa hàng (CSV / Excel / SQLite).
Model ngôn ngữ đọc câu hỏi, viết code Python (pandas) để tính, chương trình chạy code đó và trả kết quả theo đúng định dạng §9 của rulebook.

## Khai báo model (theo luật cuộc thi)
| Mục | Giá trị |
|---|---|
| Model duy nhất dùng khi chạy | **Qwen/Qwen3-14B** (14,8 tỉ tham số, bản gốc bf16, không fine-tune, không quantize) |
| Nguồn | https://huggingface.co/Qwen/Qwen3-14B (công bố 04/2025) |
| Giấy phép | Apache 2.0 |
| Model khác (embedding, reranker…) | Không dùng |
| Huấn luyện / fine-tune | Không. Dữ liệu train chỉ dùng để đo điểm và sửa code. Không dùng dữ liệu public/private test để huấn luyện. |
| Chế độ sinh | Greedy (temperature 0, seed 0), tắt chế độ suy nghĩ (`enable_thinking=false`) để kết quả lặp lại được |
| Mạng | Chạy hoàn toàn offline (`--network none`) |

Code được đội viết với sự hỗ trợ của trợ lý lập trình AI (Claude) trong lúc phát triển. Khi chạy chấm điểm, **chỉ** dùng Qwen3-14B ở trên, không gọi dịch vụ hay model nào khác.

## Cấu trúc thư mục
```
datamate/
├── predict.sh              # lệnh BTC chạy trong Docker: bật vLLM -> chạy agent -> /result
├── Dockerfile              # đóng image nộp bài
├── requirements.txt        # thư viện khi phát triển trên máy cá nhân
├── requirements-docker.txt # thư viện của agent trong Docker (cài riêng, không đụng vLLM)
├── scripts/
│   ├── tai_model.sh        # tải Qwen3-14B vào models/Qwen3-14B
│   └── kiem_tra_ket_qua.py # kiểm tra /result có đúng định dạng BTC
├── src/
│   ├── main.py        # đọc câu hỏi -> agent -> ghi submission.jsonl + log từng câu (song song, có hạn giờ)
│   ├── agent.py       # bộ não: prompt, chạy code, tự sửa lỗi, chuẩn hoá đáp án
│   ├── llm.py         # gọi model: Ollama (máy cá nhân) / vLLM (Docker) / mock (thử không cần model)
│   ├── loader.py      # đọc & làm sạch dữ liệu mọi định dạng (rulebook §8)
│   ├── periods.py     # đổi "tuần rồi", "tháng này", "Tết"... thành khoảng ngày (rulebook §2)
│   ├── scorer.py      # chấm điểm Penalized Accuracy
│   ├── tu_kiem_tra.py # bộ tự kiểm tra (phải ĐẠT 100% trước khi nộp)
│   └── mock_llm.py, kiem_tra_loader.py, khao_sat.py   # công cụ phụ
├── data/              # dữ liệu cuộc thi (KHÔNG đưa lên GitHub)
├── models/            # trọng số model (KHÔNG đưa lên GitHub)
└── outputs/           # kết quả chạy (KHÔNG đưa lên GitHub)
```

## Chạy trong Docker (cách BTC chạy)
```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -v /đường/dẫn/private_test:/data:ro -v /đường/dẫn/result:/result \
  datamate:v1
```
`/code/predict.sh` (không tham số):
1. Bật vLLM với Qwen3-14B, tự chia cho mọi GPU (`--tensor-parallel-size` = số GPU), chỉ nghe ở 127.0.0.1.
2. Trong lúc model nạp: đọc trước dữ liệu tất cả shop.
3. Chạy agent song song 4 câu; mỗi câu ghi ngay 1 dòng vào `/result/submission.jsonl`, log vào `/result/logs/<id>.json` và `/result/logs/log.jsonl` (id, answer, elapsed_ms, thinking, code, output).
4. Hạn chót tổng 5 giờ 40 phút kể cả lúc nạp model: thời gian được chia theo số câu còn lại; nếu đến hạn mà còn câu chưa xong thì các câu đó ghi `abstain` để file nộp luôn đủ câu.

Thử không cần GPU (LLM giả): thêm `-e LLM_BACKEND=mock`, bỏ `--gpus all`.

Các biến có thể chỉnh khi chạy (`-e TÊN=giá_trị`): `AGENT_WORKERS` (4), `TOTAL_SECONDS` (20400), `MAX_LEN` (32768), `GPU_UTIL` (0.90), `LLM_THINK` (0).

## Phát triển trên máy cá nhân
```bash
cd datamate
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
# Ollama + qwen3:8b trên Mac
python src/main.py --data data/train --out outputs/thu50 --sample 50 --seed 2
python src/scorer.py data/train/answers.jsonl outputs/thu50/submission.jsonl --subset
python src/tu_kiem_tra.py --data data/train
```

## Quy tắc làm việc nhóm
- Mỗi thay đổi ở agent đều chạy `tu_kiem_tra.py` (100% ĐẠT) và chấm lại bằng `scorer.py`, ghi điểm.
- Không đưa dữ liệu cuộc thi, model, file kết quả lên GitHub.
- Luật cấm: hard-code đáp án, logic theo id câu hỏi hoặc business_id, dùng dữ liệu test để huấn luyện, can thiệp khi chạy private test.
