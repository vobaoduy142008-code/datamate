#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In bảng đánh giá v6 trên bộ 300 câu (hoặc bộ giữ riêng), từ một hay nhiều lần chạy (outputs/<run>/).

    python3 training_code/dsl/bang_v6.py outputs/eval300_v6_qwen3_8b [outputs/eval300_ablation_A ...] [--eval data/eval300.jsonl]

Mỗi lần chạy in: đúng / bỏ / sai / điểm theo mức (dễ, TB, khó) và TỔNG; với log v6 (có V6_JSON) in thêm
  - theo ĐƯỜNG (dsl / tu_giai / khong_du / loi / tu_choi / công thức-trước-model): n, đúng, sai, bỏ, và "nếu luôn trả lời" (đo bản thân model,
    không qua chính sách bỏ qua);
  - theo (đường, doi_chieu);
  - TỔNG QUÁT HOÁ: câu model giải mà công thức cũ KHÔNG bắt (doi_chieu=khong_co) -> đúng / sai / bỏ;
  - độ trễ: TB, trung vị, p90 giây/câu; số lần gọi model TB.
Nhiều lần chạy -> thêm bảng so sánh gọn ở cuối.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TEN = {"de": "Dễ", "tb": "Trung bình", "kho": "Khó"}


def doc_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def pct(x, n):
    return f"{100 * x / n:.1f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--eval", default=None)
    ap.add_argument("--answers", default=None)
    a = ap.parse_args()
    root = a.root
    sys.path.insert(0, os.path.join(root, "src"))
    from scorer import score_one
    ev = doc_jsonl(a.eval or os.path.join(root, "data", "eval300.jsonl"))
    gold = {r["id"]: r for r in doc_jsonl(a.answers or os.path.join(root, "data", "train", "answers.jsonl"))}
    tom = []
    for run in a.runs:
        sub = {r["id"]: r for r in doc_jsonl(os.path.join(run, "submission.jsonl"))}
        logs = {r["id"]: r for r in doc_jsonl(os.path.join(run, "logs", "log.jsonl"))}
        print(f"\n{'=' * 100}\n{run}  (đã chạy {sum(1 for r in ev if r['id'] in sub)}/{len(ev)} câu)")
        muc = defaultdict(lambda: [0, 0, 0])             # đúng, bỏ, sai
        meta, lat = {}, []
        for r in ev:
            if r["id"] not in sub:
                continue
            s = score_one(sub[r["id"]], gold[r["id"]])
            for k in (r.get("muc", "khac"), "TỔNG"):
                muc[k][{1: 0, 0: 1, -1: 2}[s]] += 1
            lg = logs.get(r["id"]) or {}
            if lg.get("elapsed_ms") is not None:
                lat.append(lg["elapsed_ms"] / 1000)
            m = re.search(r"V6_JSON (\{.*\})", lg.get("output") or "")
            meta[r["id"]] = (json.loads(m.group(1)), s) if m else (None, s)
        print(f"{'Mức':<12}{'Đúng':>6}{'Bỏ':>5}{'Sai':>5}{'Điểm':>9}")
        for k in [x for x in ("de", "tb", "kho") if x in muc] + [x for x in muc if x not in ("de", "tb", "kho", "TỔNG")] + ["TỔNG"]:
            d, b, s = muc[k]
            n = d + b + s
            print(f"{TEN.get(k, k):<12}{d:>6}{b:>5}{s:>5}{((d - s) / n if n else 0):>+9.3f}")
        d, b, s = muc["TỔNG"]
        n = max(1, d + b + s)
        row = {"run": os.path.basename(run.rstrip('/')), "dung": d, "bo": b, "sai": s, "diem": (d - s) / n}
        if lat:
            lat.sort()
            row.update(tb=statistics.mean(lat), p50=statistics.median(lat), p90=lat[int(0.9 * (len(lat) - 1))])
            print(f"\nĐộ trễ giây/câu: TB {row['tb']:.1f} | trung vị {row['p50']:.1f} | p90 {row['p90']:.1f} | tổng {sum(lat) / 60:.0f} phút")
        v6 = {i: m for i, (m, s) in meta.items() if m}
        if v6:
            def duong_of(i):
                return v6[i]["duong"] if i in v6 else "cong_thuc_truoc_model"
            by = defaultdict(lambda: {"n": 0, "dung": 0, "bo": 0, "sai": 0, "m_dung": 0, "m_sai": 0})
            byc = defaultdict(lambda: {"n": 0, "dung": 0, "bo": 0, "sai": 0})
            tq = {"n": 0, "dung": 0, "bo": 0, "sai": 0, "m_dung": 0, "m_sai": 0}
            for i, (m, s) in meta.items():
                key = duong_of(i)
                for tgt in ([by[key]] + ([tq] if (m and m["duong"] in ("dsl", "tu_giai", "khong_du") and m.get("doi_chieu") == "khong_co") else [])):
                    tgt["n"] += 1
                    tgt[{1: "dung", 0: "bo", -1: "sai"}[s]] += 1
                    if m and isinstance(m.get("ans_model"), dict):
                        ms = score_one(m["ans_model"], gold[i])
                        if ms == 1:
                            tgt["m_dung"] += 1
                        elif ms == -1:
                            tgt["m_sai"] += 1
                if m:
                    c = byc[(m["duong"], m.get("doi_chieu"))]
                    c["n"] += 1
                    c[{1: "dung", 0: "bo", -1: "sai"}[s]] += 1
            print(f"\nTheo ĐƯỜNG giải (kết quả sau chính sách; 'nếu luôn trả lời' = model đúng/sai khi bỏ qua chính sách):")
            print(f"{'đường':<24}{'n':>5}{'đúng':>6}{'sai':>5}{'bỏ':>5}   {'nếu luôn trả lời: đúng':>24}{'sai':>5}")
            for k in sorted(by):
                v = by[k]
                print(f"{k:<24}{v['n']:>5}{v['dung']:>6}{v['sai']:>5}{v['bo']:>5}   {v['m_dung']:>24}{v['m_sai']:>5}")
            print(f"\nTheo (đường, doi_chieu với công thức cũ):")
            for k in sorted(byc, key=str):
                v = byc[k]
                print(f"  {k[0]:<10}{str(k[1]):<10}{v['n']:>5} câu: đúng {v['dung']:>3} sai {v['sai']:>3} bỏ {v['bo']:>3}")
            print(f"\nTỔNG QUÁT HOÁ — câu model giải mà công thức cũ KHÔNG bắt: {tq['n']} câu ({pct(tq['n'], n)} bộ): "
                  f"sau chính sách đúng {tq['dung']} / sai {tq['sai']} / bỏ {tq['bo']}; nếu luôn trả lời: đúng {tq['m_dung']} / sai {tq['m_sai']}")
            nl = [m["n_llm"] for m in v6.values() if m.get("n_llm") is not None]
            if nl:
                print(f"Số lần gọi model M1/M2 TB: {statistics.mean(nl):.2f}/câu (chưa tính vòng viết code của tu_giai)")
            row["qh"] = (tq["n"], tq["dung"], tq["sai"])
        tom.append(row)
    if len(tom) > 1:
        print(f"\n{'=' * 100}\nSO SÁNH (cùng bộ {len(ev)} câu)")
        print(f"{'lần chạy':<34}{'đúng':>6}{'bỏ':>5}{'sai':>5}{'điểm':>9}{'s/câu TB':>10}{'p90':>7}")
        for r in tom:
            print(f"{r['run']:<34}{r['dung']:>6}{r['bo']:>5}{r['sai']:>5}{r['diem']:>+9.3f}{r.get('tb', 0):>10.1f}{r.get('p90', 0):>7.1f}")


if __name__ == "__main__":
    main()
