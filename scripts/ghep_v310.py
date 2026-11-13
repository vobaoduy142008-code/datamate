"""v3.10: ghép file nộp public từ (1) file v3.7 và (2) các câu chạy lại bằng v3.10.
Ở 10 shop dev_001..dev_010, câu KHÔNG hỏi luật riêng và không bị luật "không đủ dữ liệu" bắt có prompt y hệt v3.7
-> model (greedy, seed 0) cho đúng đáp án v3.7 -> giữ đáp án v3.7. Câu kiểm chứng (kiem_chung.txt) được chạy lại
để xác nhận điều đó; đáp án của chúng vẫn lấy từ v3.7.
Chạy: python3 scripts/ghep_v310.py <questions> <nop_v37> <thu_muc_chay> <file_ra>"""
import json, sys, os
qs, goc, out_dir, ra = sys.argv[1:5]
order = [json.loads(l)["id"] for l in open(qs, encoding="utf-8") if l.strip()]
L = lambda p: {d["id"]: d for d in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())} if os.path.exists(p) else {}
a, b = L(goc), L(os.path.join(out_dir, "submission.jsonl"))
kc = [l.strip() for l in open(os.path.join(out_dir, "kiem_chung.txt")) if l.strip()]
k = lambda d: (d.get("type"), json.dumps(d.get("answer"), ensure_ascii=False, sort_keys=True))
khop = [i for i in kc if i in b and k(a[i]) == k(b[i])]
print(f"Câu kiểm chứng: {len(khop)}/{len([i for i in kc if i in b])} câu ra GIỐNG HỆT v3.7")
n = 0
with open(ra, "w", encoding="utf-8") as f:
    for i in order:
        if i in b and i not in kc:
            r = b[i]; n += 1
        else:
            r = a.get(i) or {"id": i, "type": "abstain"}
        r = {kk: v for kk, v in r.items() if kk in ("id", "type", "answer", "reason")}
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{ra}: {len(order)} câu, {n} câu lấy từ bản chạy v3.10, còn lại từ v3.7")
