"""Tạo file nộp ĐỦ mọi câu: câu nào đã chạy thì lấy đáp án, câu chưa chạy thì ghi abstain (0 điểm, không bị trừ).

Chạy:  python scripts/tao_file_nop.py <questions.jsonl> <submission.jsonl đang chạy dở hoặc 'none'> <file ra>
Ví dụ: python scripts/tao_file_nop.py data/public_test/questions.jsonl outputs/public/submission.jsonl outputs/nop_public.jsonl
"""
import json
import os
import sys

TYPES = {"number", "percent", "string", "date", "list", "table", "boolean", "unanswerable", "abstain"}

qpath, spath, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["id"] for l in open(qpath, encoding="utf-8") if l.strip()]
done = {}
if spath != "none" and os.path.exists(spath):
    for line in open(spath, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue                                   # dòng cuối bị cắt dở
        if d.get("id") in ids and d.get("type") in TYPES:
            done[d["id"]] = d
with open(out, "w", encoding="utf-8") as f:
    for i in ids:
        f.write(json.dumps(done.get(i, {"id": i, "type": "abstain"}), ensure_ascii=False) + "\n")
n_ans = sum(1 for d in done.values() if d["type"] != "abstain")
print(f"Đã ghi {out}: {len(ids)} câu, trong đó {len(done)} câu đã chạy ({n_ans} câu có trả lời), "
      f"{len(ids) - len(done)} câu chưa chạy -> abstain.")
