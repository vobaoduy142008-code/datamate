"""Ghép file nộp: lấy file GỐC, thay đáp án các câu có trong file MỚI. Dùng để đo trên leaderboard tác dụng của một
thay đổi chỉ chạy lại ở một nhóm câu (mọi câu khác giữ nguyên như bản gốc).
Chạy:  python3 scripts/ghep_ket_qua.py <questions.jsonl> <file gốc> <file mới> <file ra>"""
import json, sys

qs, goc, moi, ra = sys.argv[1:5]
order = [json.loads(l)["id"] for l in open(qs, encoding="utf-8") if l.strip()]
L = lambda p: {d["id"]: d for d in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
a, b = L(goc), L(moi)
n = 0
with open(ra, "w", encoding="utf-8") as f:
    for i in order:
        r = b.get(i) or a.get(i) or {"id": i, "type": "abstain"}
        n += i in b
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{ra}: {len(order)} câu, {n} câu lấy từ {moi}")
