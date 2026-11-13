"""
Khảo sát dữ liệu: liệt kê mọi file / sheet / bảng của từng cửa hàng,
kèm số dòng, số cột, tên cột và dòng tiêu đề (header) nằm ở đâu.

Chạy:
    python src/khao_sat.py --data data/train
    python src/khao_sat.py --data data/public_test

Kết quả:
    - In tóm tắt ra màn hình
    - Ghi bảng chi tiết vào outputs/khao_sat_<tên split>.csv (mở bằng Excel / Google Sheets)

Giải thích nhanh:
    - "header_row" = số thứ tự dòng chứa tên cột (0 = dòng đầu tiên), đếm sau khi bỏ các dòng trống.
      Nếu > 0 nghĩa là phía trên có dòng tiêu đề/ghi chú thừa -> khi đọc phải bỏ qua.
"""
import argparse
import csv
import os
import sqlite3
import sys
import time

import pandas as pd


def guess_header_row(df_head: pd.DataFrame) -> int:
    """Đoán dòng tiêu đề: dòng đầu tiên có nhiều ô chữ gần bằng dòng 'đầy' nhất."""
    if df_head.empty:
        return 0
    counts = df_head.notna().sum(axis=1).tolist()
    best = max(counts)
    for i, row in df_head.iterrows():
        filled = row.dropna()
        n_text = sum(isinstance(v, str) for v in filled)
        if len(filled) >= 0.8 * best and n_text >= 0.8 * len(filled):
            return int(i)
    return 0


def read_csv_any(path, **kw):
    """Thử vài kiểu mã hoá chữ (encoding) phổ biến cho file CSV tiếng Việt."""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw), enc
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Không đọc được {path}")


def survey_csv(path):
    head, enc = read_csv_any(path, header=None, nrows=15)
    h = guess_header_row(head)
    df, _ = read_csv_any(path, skiprows=h)
    yield {"sheet": "", "n_rows": len(df), "n_cols": df.shape[1],
           "header_row": h, "columns": list(map(str, df.columns)), "note": f"encoding={enc}"}


def survey_xlsx(path):
    sheets = pd.read_excel(path, sheet_name=None, header=None)   # đọc mọi sheet
    for name, raw in sheets.items():
        raw = raw.dropna(how="all")                               # bỏ dòng trống hoàn toàn
        if raw.empty:
            yield {"sheet": name, "n_rows": 0, "n_cols": 0, "header_row": "", "columns": [], "note": "sheet trống"}
            continue
        raw = raw.reset_index(drop=True)
        h = guess_header_row(raw.head(15))
        cols = [str(c) for c in raw.iloc[h].tolist() if pd.notna(c)]
        yield {"sheet": name, "n_rows": len(raw) - h - 1, "n_cols": len(cols),
               "header_row": h, "columns": cols, "note": ""}


def survey_sqlite(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)       # mở chỉ đọc
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            yield {"sheet": t, "n_rows": n, "n_cols": len(cols), "header_row": "", "columns": cols, "note": "bảng sqlite"}
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train", help="thư mục split, bên trong có data/<business_id>/")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    biz_root = os.path.join(args.data, "data")
    if not os.path.isdir(biz_root):
        sys.exit(f"Không thấy thư mục {biz_root}. Kiểm tra lại --data.")
    os.makedirs(args.out, exist_ok=True)
    split = os.path.basename(os.path.normpath(args.data))
    out_csv = os.path.join(args.out, f"khao_sat_{split}.csv")

    rows = []
    t_all = time.time()
    for biz in sorted(os.listdir(biz_root)):
        bdir = os.path.join(biz_root, biz)
        if not os.path.isdir(bdir):
            continue
        files = sorted(f for f in os.listdir(bdir) if f.lower().endswith((".csv", ".xlsx", ".xls", ".sqlite", ".db")))
        has_readme = os.path.exists(os.path.join(bdir, "README.md"))
        total = 0
        for f in files:
            path = os.path.join(bdir, f)
            ext = f.lower().rsplit(".", 1)[-1]
            t0 = time.time()
            try:
                gen = survey_csv(path) if ext == "csv" else survey_sqlite(path) if ext in ("sqlite", "db") else survey_xlsx(path)
                for r in gen:
                    total += r["n_rows"] or 0
                    rows.append({"business_id": biz, "file": f, **r,
                                 "columns": " | ".join(r["columns"]), "seconds": round(time.time() - t0, 1)})
            except Exception as e:                                # ghi lỗi lại, không dừng chương trình
                rows.append({"business_id": biz, "file": f, "sheet": "", "n_rows": "", "n_cols": "",
                             "header_row": "", "columns": "", "note": f"LỖI: {e}", "seconds": ""})
        print(f"{biz}: {len(files)} file, ~{total:,} dòng dữ liệu, README: {'có' if has_readme else 'KHÔNG'}")

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["business_id", "file", "sheet", "n_rows", "n_cols",
                                           "header_row", "columns", "note", "seconds"])
        w.writeheader()
        w.writerows(rows)

    n_err = sum(1 for r in rows if str(r["note"]).startswith("LỖI"))
    n_hdr = sum(1 for r in rows if isinstance(r["header_row"], int) and r["header_row"] > 0)
    print(f"\nTổng: {len(rows)} file/sheet/bảng | header không ở dòng đầu: {n_hdr} | lỗi: {n_err}")
    print(f"Thời gian: {time.time() - t_all:.0f} giây. Bảng chi tiết: {out_csv}")


if __name__ == "__main__":
    main()
