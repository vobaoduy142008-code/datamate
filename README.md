# DataMate – Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng (CSV / Excel / SQLite).

## Cấu trúc thư mục
```
datamate/
├── src/
│   ├── main.py      # chương trình chính: đọc câu hỏi -> agent -> ghi submission + log
│   ├── agent.py     # hàm solve(): bộ não của agent (bản 0: luôn abstain)
│   ├── periods.py   # đổi "tuần rồi", "tháng này", "Tết"... thành khoảng ngày (rulebook §2)
│   └── scorer.py    # chấm điểm PA đúng luật cuộc thi
├── data/            # dữ liệu cuộc thi (KHÔNG đưa lên GitHub)
│   ├── train/       #   chép thư mục train vào đây
│   └── public_test/ #   chép thư mục public_test (bên trong có questions.jsonl, data/...)
├── outputs/         # kết quả chạy (KHÔNG đưa lên GitHub)
└── requirements.txt
```

## Cài đặt (mỗi máy làm 1 lần)
macOS / Linux / WSL:
```bash
cd datamate
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```
Windows (PowerShell):
```powershell
cd datamate
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## Chạy thử
```bash
python src/main.py --data data/train --out outputs/train_run
python src/scorer.py data/train/answers.jsonl outputs/train_run/submission.jsonl
```
Bản 0 phải ra `Penalized Accuracy = 0.0000` (abstain toàn bộ).

## Quy tắc làm việc nhóm
- Mỗi người làm trên nhánh (branch) riêng, xong thì tạo Pull Request để người khác xem.
- Mỗi thay đổi ở agent đều phải chấm lại bằng `scorer.py` và ghi điểm vào PR.
- Không đưa dữ liệu cuộc thi, model, file kết quả lên GitHub.
- Luật cấm: hard-code đáp án, logic theo id câu hỏi hoặc business_id.
