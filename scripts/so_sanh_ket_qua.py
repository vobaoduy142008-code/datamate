"""So sánh đáp án của hai lần chạy (ví dụ Mac và PC Pro) trên các câu CẢ HAI đã chạy.

Chạy:  python scripts/so_sanh_ket_qua.py <submission.jsonl máy A> <submission.jsonl máy B>
In ra: số câu chung, tỉ lệ giống nhau, danh sách câu khác nhau.
Không cần đáp án đúng: chỉ xem hai máy có ra cùng kết quả không (luật cho phép lệch PA tối đa 0,03).
"""
import json
import sys


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d["id"]] = d
    return out


def key(d):
    return json.dumps({k: d.get(k) for k in ("type", "answer")}, ensure_ascii=False, sort_keys=True)


a, b = load(sys.argv[1]), load(sys.argv[2])
common = [i for i in a if i in b]
diff = [i for i in common if key(a[i]) != key(b[i])]
print(f"Câu chung: {len(common)}  |  giống nhau: {len(common) - len(diff)}  |  khác nhau: {len(diff)}"
      + (f"  ({100 * (len(common) - len(diff)) / len(common):.1f}% giống)" if common else ""))
for i in diff[:30]:
    print(f"- {i}:  A = {key(a[i])[:80]}   |   B = {key(b[i])[:80]}")
