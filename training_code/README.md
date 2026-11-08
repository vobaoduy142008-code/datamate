# training_code – DataMate (Zalo AI Challenge 2026)

## Không huấn luyện model
Đội **không huấn luyện / fine-tune** bất kỳ model nào. Model dùng khi chạy là **Qwen3-8B** nguyên bản
(bản `qwen3:8b` của Ollama, lượng tử hoá 4-bit, Apache 2.0), trọng số nằm sẵn trong image tại `/models/ollama`.
Không có checkpoint, adapter hay log huấn luyện.

## Dữ liệu đã dùng
- Chỉ bộ **train** do BTC cung cấp: `/code/data/train` (3000 câu hỏi + đáp án, 30 shop, rulebook, calendar).
- Dùng để: (1) đo điểm khi phát triển (`src/scorer.py`, mẫu 50 câu với `--sample 50 --seed N`);
  (2) dựng 2 bảng tra cứu chữ bên dưới.
- **Không** dùng dữ liệu public test / private test để huấn luyện, chỉnh tham số hay gán nhãn.
- Danh sách file + SHA-256: `training_code/data_manifest.json` (tạo lúc build bằng `scripts/tao_manifest.py`).

## Dữ liệu sinh ra (tiền xử lý, chạy được lại, không ngẫu nhiên)
| File | Script | Từ đâu |
|---|---|---|
| `src/ngram_dau.json` (thêm dấu cho câu hỏi gõ không dấu) | `data_generation/tao_ngram_dau.py data/train` | chữ có dấu trong câu hỏi train, rulebook, README shop train, từ điển |
| `src/am_tiet.txt` (âm tiết hợp lệ để sửa lỗi gõ Telex) | `data_generation/tao_am_tiet.py data/train src/am_tiet.txt` | như trên |
| `docs/tu_dien_viet_tat.md` (từ viết tắt / địa phương) | đội tự viết | – |

Chạy lại (trong image, thư mục `/code`):
```
/opt/agent/bin/python training_code/data_generation/tao_ngram_dau.py data/train
/opt/agent/bin/python training_code/data_generation/tao_am_tiet.py data/train src/am_tiet.txt
```
Cả hai chỉ đếm chữ, không có bước ngẫu nhiên; chạy trên CPU trong vài giây.

## Kiểm tra (validation)
```
/opt/agent/bin/python src/main.py --data data/train --out /tmp/val --sample 50 --seed 13
/opt/agent/bin/python src/scorer.py data/train/answers.jsonl /tmp/val/submission.jsonl --subset
```
