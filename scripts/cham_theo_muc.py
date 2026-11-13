"""Chấm điểm một lượt chạy trên bộ 300 câu, tách theo mức (dễ / trung bình / khó) và theo nhóm chủ đề.
Có thể so với một lượt khác (vd bản trước) trên cùng bộ câu: câu nào tốt lên / tệ đi.
Chạy:  python3 scripts/cham_theo_muc.py data/eval300.jsonl data/train/answers.jsonl outputs/eval300_v3.8/submission.jsonl [outputs/eval300_v3.7/submission.jsonl]"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scorer import score_one  # noqa: E402

TEN = {"de": "Dễ", "tb": "Trung bình", "kho": "Khó"}


def load(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        try:
            d = json.loads(l)
            out[d["id"]] = d
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def bang(rows, sub, an):
    g = defaultdict(lambda: [0, 0, 0, 0])            # đúng, bỏ, sai, chưa chạy
    for r in rows:
        for k in (r["muc"], "TỔNG", r["muc"] + "|" + r["nhom"]):
            if r["id"] not in sub:
                g[k][3] += 1
                continue
            s = score_one(sub[r["id"]], an[r["id"]])
            g[k][{1: 0, 0: 1, -1: 2}[s]] += 1
    return g


def in_bang(g, title):
    print(f"\n== {title}")
    print(f"{'Mức':<12}{'Đúng':>6}{'Bỏ':>5}{'Sai':>5}{'Chưa':>6}{'Điểm':>8}")
    for k in ("de", "tb", "kho", "TỔNG"):
        d, b, s, c = g[k]
        n = d + b + s
        print(f"{TEN.get(k, k):<12}{d:>6}{b:>5}{s:>5}{c:>6}{((d - s) / n if n else 0):>+8.3f}")
    print("\n  Theo nhóm (đúng/bỏ/sai):")
    for k in sorted(x for x in g if "|" in x):
        d, b, s, c = g[k]
        print(f"   {TEN[k.split('|')[0]]:<11} {k.split('|')[1]:<45} {d:>3}/{b:>2}/{s:>2}")


def main():
    ev, ans, cur = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in open(ev, encoding="utf-8") if l.strip()]
    an = load(ans)
    a = load(cur)
    in_bang(bang(rows, a, an), f"{cur}")
    if len(sys.argv) > 4:
        b = load(sys.argv[4])
        in_bang(bang(rows, b, an), f"{sys.argv[4]} (bản so sánh)")
        up, down = [], []
        for r in rows:
            if r["id"] in a and r["id"] in b:
                sa, sb = score_one(a[r["id"]], an[r["id"]]), score_one(b[r["id"]], an[r["id"]])
                (up if sa > sb else down if sa < sb else []).append(r)
        print(f"\n== So từng câu: TỐT LÊN {len(up)} | TỆ ĐI {len(down)}")
        for r in down:
            print(f"   TỆ ĐI  [{TEN[r['muc']]}] {r['business_id']} {r['question'][:90]}")
        for r in up:
            print(f"   TỐT LÊN [{TEN[r['muc']]}] {r['business_id']} {r['question'][:90]}")


if __name__ == "__main__":
    main()
