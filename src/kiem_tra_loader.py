"""
Kiểm tra bộ đọc dữ liệu trên mọi cửa hàng của một split.

Chạy:
    python src/kiem_tra_loader.py --data data/train
    python src/kiem_tra_loader.py --data data/public_test

Kết quả:
    - Màn hình: mỗi shop 1 dòng (số bảng, số dòng, thời gian, số cờ)
    - outputs/kiem_tra_loader_<split>.txt: báo cáo chi tiết + schema_text() của từng shop

Các "cờ" cần chú ý:
    [README]   cột có trong mục "Cột trong `file`" của README nhưng bộ đọc không tìm thấy
    [SỐ?]      cột kiểu chữ nhưng phần lớn giá trị trông như số/tiền (chuyển kiểu chưa được)
    [NGÀY?]    cột tên gợi ý ngày nhưng chưa thành kiểu ngày
    [VƯỢT HẠN] ngày giao dịch lớn hơn "Dữ liệu có đến hết ..." trong README (có thể đọc sai ngày/tháng)
    [CỘT?]     bảng có cột tên tự đặt "cot_..." (dòng tên cột bị thiếu ô)
    [GHI CHÚ]  việc bộ đọc đã tự làm (điền xuống theo nhóm, chọn ngày/tháng...) - đọc để biết
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from loader import _is_code, _is_date_name, fold, load_business, parse_readme, to_number  # noqa: E402

_NOT_TXN_DATE = re.compile(r"(sinh|han|hsd|het han|den ngay|tu ngay|ngay phai tra|ngay vao|tao|khai truong|nhan viec|dang ky)")


def check(biz):
    flags = []
    meanings, _, end = parse_readme(biz.readme)
    readme_cols = {fold(c): f for f, cols in meanings.items() if f != "*" for c in cols}
    found = {fold(c) for df in biz.tables.values() for c in df.columns}
    for c, fname in readme_cols.items():
        if c not in found:
            flags.append(f"[README] không thấy cột '{c}' (README ghi ở {fname})")
    for name, df in biz.tables.items():
        for c in df.columns:
            if c == "_nguon" or c.endswith("(đơn vị ghi kèm)"):
                continue
            s = df[c]
            if c.startswith("cot_"):
                flags.append(f"[CỘT?] {name}.{c}")
            if pd.api.types.is_datetime64_any_dtype(s):
                if end is not None and s.notna().any() and not _NOT_TXN_DATE.search(fold(c)):
                    over = (s.dt.date > end).sum()
                    if over:
                        flags.append(f"[VƯỢT HẠN] {name}.{c}: {over} ô sau {end} (max {s.max():%Y-%m-%d})")
                continue
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                continue
            vals = s.dropna().astype(str).str.strip()
            if len(vals) == 0:
                continue
            share = vals.head(500).map(lambda v: to_number(v) is not None).mean()
            if share > 0.8 and not _is_code(c):
                flags.append(f"[SỐ?] {name}.{c} ({share:.0%} giống số) vd: {vals.iloc[0]}")
            if _is_date_name(c, biz.col_meaning.get(name, {}).get(c, "")):
                flags.append(f"[NGÀY?] {name}.{c} vd: {', '.join(vals.head(3))}")
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
            flags = check(biz)
            notes = [f"[GHI CHÚ] {w}" for w in biz.warnings]
            n_flags += len(flags)
            rows = sum(len(d) for d in biz.tables.values())
            print(f"{biz_id}: {len(biz.tables)} bảng, {rows:,} dòng, {time.time() - t0:.1f}s, "
                  f"{len(flags)} cờ, {len(notes)} ghi chú")
            fo.write(f"\n==================== {biz_id} ({time.time() - t0:.1f}s)\n")
            fo.write(f"Bảng: {biz.summary()}\nDữ liệu đến hết (README): {biz.data_end}\n")
            for f in flags + notes:
                fo.write("  " + f + "\n")
            for tname, items in biz.ffill_samples.items():
                for cl, sample in items:
                    fo.write(f"\n  [MẪU ĐIỀN XUỐNG] {tname}: cột {cl} - các dòng TRƯỚC khi điền:\n")
                    with pd.option_context("display.width", 200, "display.max_columns", 12,
                                           "display.max_colwidth", 25):
                        fo.write("    " + sample.to_string().replace("\n", "\n    ") + "\n")
            fo.write("\n" + biz.schema_text(with_notes=False) + "\n")
    print(f"\nXong {time.time() - t_all:.0f}s, tổng {n_flags} cờ. Báo cáo: {out_path}")


if __name__ == "__main__":
    main()
