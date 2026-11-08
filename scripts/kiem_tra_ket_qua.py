"""Kiểm tra thư mục kết quả của Docker có đúng định dạng BTC không.
Chạy:  python scripts/kiem_tra_ket_qua.py <thư mục result> <file questions.jsonl>
"""
import json
import os
import sys

TYPES = {"number", "percent", "string", "date", "list", "table", "boolean", "unanswerable", "abstain"}
res, qpath = sys.argv[1], sys.argv[2]
qids = [json.loads(l)["id"] for l in open(qpath, encoding="utf-8") if l.strip()]
sub = [json.loads(l) for l in open(os.path.join(res, "submission.jsonl"), encoding="utf-8") if l.strip()]
ids = [s["id"] for s in sub]
loi = []
if len(ids) != len(set(ids)):
    loi.append(f"có {len(ids) - len(set(ids))} id bị trùng")
thieu = set(qids) - set(ids)
if thieu:
    loi.append(f"thiếu {len(thieu)} câu, ví dụ {sorted(thieu)[:3]}")
for s in sub:
    if s.get("type") not in TYPES:
        loi.append(f"{s['id']}: type lạ {s.get('type')}")
    elif s["type"] not in ("abstain",) and "answer" not in s:
        loi.append(f"{s['id']}: thiếu answer")
logs = os.path.join(res, "logs")
for i in ids[:50]:
    p = os.path.join(logs, f"{i}.json")
    if not os.path.exists(p):
        loi.append(f"thiếu log {p}"); break
    d = json.load(open(p, encoding="utf-8"))
    for k in ("id", "answer", "elapsed_ms", "thinking", "code", "output"):
        if k not in d:
            loi.append(f"log {i} thiếu trường {k}")
from collections import Counter
print("Số câu:", len(sub), "/", len(qids), " Loại:", dict(Counter(s.get("type") for s in sub)))
print("KHÔNG CÓ LỖI ✅" if not loi else "LỖI:\n  " + "\n  ".join(loi[:20]))
