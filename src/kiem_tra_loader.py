"""
Kiểm tra bộ đọc dữ liệu trên mọi cửa hàng của một split.

Chạy:
    python src/kiem_tra_loader.py --data data/train
    python src/kiem_tra_loader.py --data data/public_test

Kết quả:
    - Màn hình: mỗi shop 1 dòng (số bảng, số dòng, thời gian, số cảnh báo)
    - outputs/kiem_tra_loader_<split>.txt: báo cáo chi tiết + schema_text() của từng shop
      -> gửi file này cho Claude để rà soát.

Các "cờ" cần chú ý trong báo cáo:
    [README]  cột có trong README nhưng bộ đọc không tìm thấy (có thể đọc sai dòng tên cột)
    [SỐ?]     cột kiểu chữ nhưng phần lớn giá trị trông như số (chuyển kiểu chưa được)
    [NGÀY?]   cột tên có chữ "ngày/thời gian" nhưng chưa thành kiểu ngày
    [CỘT?]    bảng có cột tên tự đặt "cot_..." (dòng tên cột bị thiếu ô)
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from loader import fold, load_business, parse_readme  # noqa: E402


def check(biz):
    flags = []
    readme_cols = {}
    for fname, cols in parse_readme(biz.readme).items():
        for c in cols:
            readme_cols[fold(c)] = fname
    found = {fold(c) for df in biz.tables.values() for c in df.columns}
    for c, fname in readme_cols.items():
        if c not in found:
            flags.append(f"[README] không thấy cột '{c}' (README ghi ở {fname})")
    for name, df in biz.tables.items():
        for c in df.columns:
            if c == "_nguon":
                continue
            s = df[c]
            if c.startswith("cot_"):
                flags.append(f"[CỘT?] {name}.{c}")
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                continue
            vals = s.dropna().astype(str).str.strip()
            if len(vals) == 0:
                continue
            numeric_share = vals.str.match(r"^[+-]?[\d.,\s]+(đ|d|vnd|vnđ)?$", case=False).mean()
            if numeric_share > 0.8 and not re.search(r"(^|\s)(ma|id|sdt|dien thoai|phone|imei|so)(\s|$)", fold(c)):
                flags.append(f"[SỐ?] {name}.{c} ({numeric_share:.0%} giống số) vd: {vals.iloc[0]}")
            if re.search(r"ngay|thoi gian|date|time|thoi diem", fold(c)) and not re.search(r"so ngay|gio", fold(c)):
                flags.append(f"[NGÀY?] {name}.{c} vd: {vals.iloc[0]}")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    root = os.path.join(args.data, "data")
    split = os.path.basename(os.path.normpath(args.data))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"kiem_tra_loader_{split}.txt")
    t_all = time.time()
    n_flags = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for biz_id in sorted(os.listdir(root)):
            p = os.path.join(root, biz_id)
            if not os.path.isdir(p):
                continue
            t0 = time.time()
            try:
                biz = load_business(p, use_cache=not args.no_cache)
            except Exception as e:
                print(f"{biz_id}: LỖI {e}")
                fo.write(f"\n==================== {biz_id}: LỖI {e}\n")
                continue
            flags = check(biz) + [f"[CẢNH BÁO] {w}" for w in biz.warnings]
            n_flags += len(flags)
            rows = sum(len(d) for d in biz.tables.values())
            print(f"{biz_id}: {len(biz.tables)} bảng, {rows:,} dòng, {time.time() - t0:.1f}s, {len(flags)} cờ")
            fo.write(f"\n==================== {biz_id} ({time.time() - t0:.1f}s)\n")
            fo.write("Bảng: " + biz.summary() + "\n")
            for f in flags:
                fo.write("  " + f + "\n")
            fo.write("\n" + biz.schema_text() + "\n")
    print(f"\nXong {time.time() - t_all:.0f}s, tổng {n_flags} cờ. Báo cáo: {out_path}")


if __name__ == "__main__":
    main()
