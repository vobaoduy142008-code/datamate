#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo tập giữ riêng data/giu_rieng.jsonl: N câu TRAIN (cân bằng giữa các shop) KHÔNG thuộc eval300 và không phải ví dụ viết tay.
Dùng để hiệu chỉnh chính sách (hieu_chinh.py) mà vẫn còn eval300 sạch để báo số. sinh_du_lieu.py tự loại các câu này khỏi dữ liệu huấn luyện,
nên PHẢI tạo tập này TRƯỚC khi chạy sinh_du_lieu.py. Chạy: .venv/bin/python training_code/dsl/tao_giu_rieng.py [--n 300] [--seed 7]
Chạy bằng chay_v6_300.command với IDS=data/giu_rieng.jsonl OUT=outputs/giu_rieng_v6_<model>."""
import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def doc(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
out = os.path.join(ROOT, "data", "giu_rieng.jsonl")
if os.path.exists(out):
    raise SystemExit(f"{out} đã có ({len(doc(out))} câu) -> giữ nguyên (xoá tay nếu muốn tạo lại).")
cam = {r["id"] for r in doc(os.path.join(ROOT, "data", "eval300.jsonl"))} | {r["id"] for r in doc(os.path.join(HERE, "vi_du_tay.jsonl"))}
theo = defaultdict(list)
for q in doc(os.path.join(ROOT, "data", "train", "questions.jsonl")):
    if q["id"] not in cam:
        theo[q["business_id"]].append(q)
rng = random.Random(a.seed)
for v in theo.values():
    rng.shuffle(v)
chon, i = [], 0
shops = sorted(theo)
while len(chon) < a.n and any(theo.values()):
    s = shops[i % len(shops)]
    if theo[s]:
        q = theo[s].pop()
        chon.append({"id": q["id"], "business_id": q["business_id"], "muc": "khac", "nhom": "giữ riêng", "question": q["question"]})
    i += 1
with open(out, "w", encoding="utf-8") as f:
    for r in chon:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Đã ghi {out}: {len(chon)} câu từ {len({r['business_id'] for r in chon})} shop")
