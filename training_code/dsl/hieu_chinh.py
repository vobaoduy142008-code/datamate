#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hiệu chỉnh chính sách trả lời / bỏ qua của v6 từ log MỘT lần chạy TRAIN -> src/chinh_sach_v6.json.

Đọc <run>/logs/log.jsonl (dòng V6_JSON do agent_v6 ghi), đối chiếu data/train/answers.jsonl, đếm đúng/sai của ĐÁP ÁN MODEL
(ans_model — kể cả câu chính sách đã bỏ qua) theo từng ô đặc trưng:
    mức 0: duong | doi_chieu | so_sua | buoc | loai      (chi tiết nhất)
    mức 1: duong | doi_chieu | so_sua | buoc
    mức 2: duong | doi_chieu                              (thô nhất)
Ô được ghi khi đủ mẫu (mức 2: n >= --n-min; mức 0/1: n >= 2 x --n-min). Trả lời ô nào có cận dưới Wilson (--z) của kỳ vọng
điểm 2*p-1 lớn hơn --bien (biên an toàn); ngược lại bỏ qua. Lúc chạy, agent_v6.quyet_dinh tra ô chi tiết nhất có trong bảng,
không có -> quy tắc mặc định. Câu đáp án rỗng/bằng 0 đã bị luật AGENT_ABSTAIN_ZERO bỏ riêng -> không tính vào bảng.

KHÔNG dùng public: script từ chối nếu < 90% id trong log thuộc data/train/answers.jsonl.
In kèm ước lượng ngoài mẫu: chia shop thành --folds phần, hiệu chỉnh trên phần còn lại, chấm trên phần giữ -> so với quy tắc mặc định
và với "luôn trả lời". Nếu bảng chỉ hiệu chỉnh trên eval300 thì hãy đánh giá lại trên một tập giữ riêng khác (đừng báo số trên chính eval300).

    python3 training_code/dsl/hieu_chinh.py --run outputs/eval300_v6_qwen3_8b [--run ...] [--out src/chinh_sach_v6.json]
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def doc_jsonl(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def wilson_duoi(k, n, z):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    tam = p + z * z / (2 * n)
    bien = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (tam - bien) / d)


def doc_ban_ghi(runs, gold, A, score_one):
    """-> list {id, shop, dt, sc} (sc = điểm của ans_model nếu model trả lời: +1/-1)."""
    rec, bo_public, khong_v6 = [], 0, 0
    for run in runs:
        p = os.path.join(run, "logs", "log.jsonl")
        for d in doc_jsonl(p):
            if d.get("id") not in gold:
                bo_public += 1
                continue
            m = re.search(r"V6_JSON (\{.*\})", d.get("output") or "")
            if not m:
                khong_v6 += 1
                continue
            meta = json.loads(m.group(1))
            ans = meta.get("ans_model")
            if not isinstance(ans, dict) or ans.get("type") in (None, "abstain"):
                continue
            if (ans.get("type") in ("number", "percent") and ans.get("answer") == 0) or (ans.get("type") == "list" and not ans.get("answer")):
                continue
            dt = A.dac_trung_tu_meta(meta)
            if dt["duong"] not in ("dsl", "khong_du", "tu_giai"):
                continue
            rec.append({"id": d["id"], "shop": d.get("business_id"), "dt": dt, "sc": score_one(ans, gold[d["id"]])})
    return rec, bo_public, khong_v6


def xay_bang(rec, A, n_min, bien, z):
    dem = defaultdict(lambda: [0, 0])
    for r in rec:
        for muc in (0, 1, 2):
            c = dem[(muc, A.khoa_o(r["dt"], muc))]
            c[0 if r["sc"] == 1 else 1] += 1
    bang = {}
    for (muc, khoa), (dung, sai) in dem.items():
        n = dung + sai
        if n < (n_min if muc == 2 else 2 * n_min):
            continue
        p_lo = wilson_duoi(dung, n, z)
        bang[khoa] = {"khoa": khoa, "muc": muc, "n": n, "dung": dung, "sai": sai, "p": round(dung / n, 4), "p_duoi": round(p_lo, 4),
                      "quyet_dinh": "tra_loi" if 2 * p_lo - 1 > bien else "bo_qua"}
    return bang


def tra_bang(bang, dt, A):
    for muc in (0, 1, 2):
        c = bang.get(A.khoa_o(dt, muc))
        if c:
            return c["quyet_dinh"]
    return A.quyet_dinh_mac_dinh(dt)[0]


def diem(rec, chon):
    d = sum(1 for r in rec if chon(r) and r["sc"] == 1)
    s = sum(1 for r in rec if chon(r) and r["sc"] == -1)
    return d, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="thư mục một lần chạy v6 trên TRAIN (có logs/log.jsonl); lặp lại để gộp")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None, help="mặc định <root>/src/chinh_sach_v6.json")
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--bien", type=float, default=0.05, help="kỳ vọng điểm (cận dưới) phải > biên này mới trả lời")
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    sys.path.insert(0, a.src or os.path.join(root, "src"))
    import agent_v6 as A
    from scorer import score_one

    gold = {r["id"]: r for r in doc_jsonl(os.path.join(root, "data", "train", "answers.jsonl"))}
    rec, bo_public, khong_v6 = doc_ban_ghi(a.run, gold, A, score_one)
    tong_log = len(rec) + bo_public + khong_v6
    if tong_log and bo_public / tong_log > 0.10:
        sys.exit(f"{bo_public}/{tong_log} id trong log KHÔNG thuộc data/train (public?) -> từ chối hiệu chỉnh trên public.")
    if len(rec) < 30:
        sys.exit(f"Chỉ {len(rec)} bản ghi dùng được (cần log AGENT_MODE=v6 trên câu TRAIN) -> chưa đủ để hiệu chỉnh.")
    print(f"Đọc {len(rec)} câu model trả lời được từ {len(a.run)} lần chạy ({khong_v6} câu không có V6_JSON bị bỏ, {bo_public} không thuộc train bị bỏ)")

    bang = xay_bang(rec, A, a.n_min, a.bien, a.z)
    print(f"\n{'ô':<62}{'n':>5}{'đúng':>6}{'sai':>5}{'p':>7}{'p_dưới':>8}  quyết định")
    for k in sorted(bang):
        c = bang[k]
        if c["muc"] >= 1:
            print(f"{k:<62}{c['n']:>5}{c['dung']:>6}{c['sai']:>5}{c['p']:>7.2f}{c['p_duoi']:>8.2f}  {c['quyet_dinh']}")

    # ---- ước lượng ngoài mẫu: chia shop thành folds
    shops = sorted({r["shop"] for r in rec})
    fold_of = {s: int(hashlib.md5(s.encode()).hexdigest(), 16) % max(2, a.folds) for s in shops}
    tot = {"bảng hiệu chỉnh": [0, 0], "quy tắc mặc định": [0, 0], "luôn trả lời": [0, 0]}
    for f in range(max(2, a.folds)):
        train = [r for r in rec if fold_of[r["shop"]] != f]
        test = [r for r in rec if fold_of[r["shop"]] == f]
        if not train or not test:
            continue
        b = xay_bang(train, A, a.n_min, a.bien, a.z)
        for ten, chon in (("bảng hiệu chỉnh", lambda r: tra_bang(b, r["dt"], A) == "tra_loi"),
                          ("quy tắc mặc định", lambda r: A.quyet_dinh_mac_dinh(r["dt"])[0] == "tra_loi"),
                          ("luôn trả lời", lambda r: True)):
            d, s = diem(test, chon)
            tot[ten][0] += d
            tot[ten][1] += s
    print(f"\nƯớc lượng NGOÀI MẪU (chia {max(2, a.folds)} phần theo shop; tính trên {len(rec)} câu model có đáp án; điểm = đúng - sai, bỏ = 0):")
    for ten, (d, s) in tot.items():
        print(f"  {ten:<18} đúng {d:>4}  sai {s:>4}  điểm {d - s:>+5}  ({(d - s) / len(rec):+.3f}/câu)")
    d, s = diem(rec, lambda r: tra_bang(bang, r["dt"], A) == "tra_loi")
    print(f"  (trong mẫu, bảng cuối: đúng {d} sai {s} điểm {d - s:+d} — lạc quan, chỉ để tham khảo)")

    out = a.out or os.path.join(root, "src", "chinh_sach_v6.json")
    js = {"phien_ban": 1, "tao_luc": time.strftime("%Y-%m-%d %H:%M:%S"), "nguon": [os.path.relpath(r, root) if os.path.isabs(r) else r for r in a.run],
          "tham_so": {"n_min": a.n_min, "bien": a.bien, "z": a.z}, "n_ban_ghi": len(rec),
          "ghi_chu": "Hiệu chỉnh trên dữ liệu TRAIN (không public). Ô không có trong bảng -> quy tắc mặc định của agent_v6.quyet_dinh_mac_dinh.",
          "o": [bang[k] for k in sorted(bang)]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)
    print(f"\nĐã ghi {out} ({len(bang)} ô). Chạy lại v6 để dùng bảng mới (copy src/ sang thư mục chạy nếu dùng bản chép).")


if __name__ == "__main__":
    main()
