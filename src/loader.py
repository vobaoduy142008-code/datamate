"""
Bộ đọc dữ liệu chung cho MỌI cửa hàng (không viết riêng cho shop nào - đúng luật cuộc thi).

    from loader import load_business
    biz = load_business("data/train/data/biz_101")
    biz.tables            # dict: tên bảng -> pandas DataFrame đã làm sạch
    biz.schema_text()     # bản mô tả các bảng để đưa vào prompt LLM
    biz.readme            # nội dung README.md

Các bước làm sạch (giải thích cho người mới):
  1. Đọc mọi file .csv / .xlsx (tất cả sheet) / .sqlite trong thư mục shop.
  2. Tìm dòng tên cột (có file có 1-2 dòng tiêu đề phía trên), bỏ dòng trống, dòng "Tổng cộng".
  3. Ghép các mảnh cùng một bảng (mỗi tháng 1 sheet, file _1/_2, file theo tháng/năm).
     Thêm cột "_nguon" (file|sheet) để biết dòng đến từ đâu, vì rulebook có luật
     "ngày không ghi năm thì lấy năm của sheet".
  4. Chuẩn hoá kiểu dữ liệu: chữ số kiểu "1.234.567đ" -> số; "05/03/2025" -> ngày.
     Cột mã (mã hàng, SĐT, IMEI, biển số...) giữ nguyên dạng chữ.
  5. Đọc README.md để lấy nghĩa từng cột, đưa vào schema_text() cho LLM.
  6. Lưu cache (file .pkl trong /tmp) để lần sau đọc lại cực nhanh.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CACHE_DIR = os.environ.get("DATAMATE_CACHE", "/tmp/datamate_cache")

# --------------------------------------------------------------------------- tiện ích chữ


def fold(s) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng: 'Mã Hóa Đơn' -> 'ma hoa don'."""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()

# --------------------------------------------------------------------------- đọc file thô


def _read_excel_raw(path):
    """Trả về {tên sheet: DataFrame không header}. Dùng calamine (nhanh) nếu có, không thì openpyxl."""
    for engine in ("calamine", "openpyxl"):
        try:
            return pd.read_excel(path, sheet_name=None, header=None, engine=engine)
        except ImportError:
            continue
        except ValueError as e:              # engine không được hỗ trợ
            if "calamine" in str(e):
                continue
            raise
    return pd.read_excel(path, sheet_name=None, header=None)


def _read_csv_raw(path):
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc,
                               keep_default_na=True, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Không đọc được {path}")


def _guess_header_row(df: pd.DataFrame, look=15) -> int:
    head = df.head(look)
    counts = head.notna().sum(axis=1).tolist()
    if not counts:
        return 0
    best = max(counts)
    for i, (_, row) in enumerate(head.iterrows()):
        filled = row.dropna()
        n_text = sum(isinstance(v, str) for v in filled)
        if len(filled) >= 0.8 * best and n_text >= 0.8 * len(filled):
            return i
    return 0


def _clean_columns(values) -> list[str]:
    cols, seen = [], {}
    for i, v in enumerate(values):
        name = nfc(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else ""
        name = re.sub(r"\s+", " ", name) or f"cot_{i + 1}"
        if name in seen:                      # tên cột trùng -> thêm hậu tố
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    return cols


_TOTAL_RE = re.compile(r"^(tong cong|tong|cong|total|sum)\b")


def _frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Bảng thô (không header) -> bảng có tên cột, đã bỏ dòng tiêu đề/trống/tổng."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()
    h = _guess_header_row(raw)
    cols = _clean_columns(raw.iloc[h].tolist())
    df = raw.iloc[h + 1:].copy()
    df.columns = cols
    # bỏ dòng lặp lại tên cột (hay gặp khi nối nhiều bảng xuất) - chỉ xét dòng nghi ngờ cho nhanh
    header_fold = [fold(c) for c in cols]
    first = df.iloc[:, 0]
    cand = first.map(lambda v: isinstance(v, str) and fold(v) == header_fold[0])
    if cand.any():
        again = df[cand].apply(lambda r: [fold(x) for x in r.tolist()] == header_fold, axis=1)
        df = df.drop(again[again].index)
    # bỏ dòng "Tổng cộng": dòng thưa (ít ô có dữ liệu) và ô chữ đầu tiên bắt đầu bằng tổng/cộng
    n_filled = df.notna().sum(axis=1)
    sparse = df[n_filled <= 0.6 * df.shape[1]]
    if len(sparse):
        def is_total(r):
            v = next((x for x in r.dropna() if isinstance(x, str)), None)
            return bool(v and _TOTAL_RE.match(fold(v)))
        tot = sparse.apply(is_total, axis=1)
        df = df.drop(tot[tot].index)
    return df.reset_index(drop=True)

# --------------------------------------------------------------------------- gom mảnh thành bảng


_MONTHY = re.compile(
    r"^(sheet\s*\d*|t\s*\d{1,2}([\s._\-/]*\d{2,4})?|thang\s*\d{1,2}([\s._\-/]*\d{2,4})?|"
    r"\d{1,2}[\s._\-/]+\d{4}|\d{4}[\s._\-/]+\d{1,2}|\d{4})$")
_SUFFIX = re.compile(r"([\s_\-]+(\d{1,3}|\d{4}|t\d{1,2}[\s_\-]*\d{4}|thang\d{1,2}[\s_\-]*\d{4}))$", re.I)


def _base_name(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    prev = None
    while prev != stem:                       # bỏ dần đuôi _1, _2026, _T1_2025, _Thang01_2025
        prev = stem
        stem = _SUFFIX.sub("", stem)
    return stem


def _table_name(file_name: str, sheet: str | None, n_sheets: int) -> str:
    base = _base_name(file_name)
    if sheet is None or _MONTHY.match(fold(sheet)):
        return base
    return nfc(sheet) if n_sheets > 1 or fold(sheet) != "sheet1" else base


# --------------------------------------------------------------------------- kiểu dữ liệu

_CODE_HINT = re.compile(r"(^|\s)(ma|id|code|sku|barcode|ean|so dien thoai|dien thoai|sdt|phone|di dong|"
                        r"imei|bien so|so ct|so phieu|so hd|so hoa don|cccd|cmnd|mst|ma so thue|so the|lo)(\s|$)")
_NUM_RE = re.compile(r"^[+-]?[\d.,\s]+(đ|d|vnd|vnđ|k)?$", re.I)
_DATE_RE = re.compile(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$")


def _to_number(s: str):
    t = s.strip().lower().replace(" ", "")
    t = re.sub(r"(đ|vnđ|vnd|d)$", "", t)
    if not t:
        return np.nan
    neg = t.startswith("-")
    t = t.lstrip("+-")
    if "." in t and "," in t:                 # dấu xuất hiện sau cùng là dấu thập phân
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        parts = t.split(sep)
        if len(parts) > 2 or (len(parts[-1]) == 3 and parts[0] != "0"):
            t = t.replace(sep, "")            # dấu phân cách hàng nghìn: 1.234.567
        else:
            t = t.replace(",", ".")           # dấu thập phân: 18,6
    try:
        v = float(t)
    except ValueError:
        return None                           # None = không phải số
    return -v if neg else v


def _convert_column(name: str, s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s
    s = s.astype(object).where(s.notna(), np.nan)
    non_null = s.dropna()
    if non_null.empty:
        return s
    sample = non_null.astype(str).str.strip()
    sample = sample[sample != ""]
    if sample.empty:
        return s
    is_code = bool(_CODE_HINT.search(fold(name)))
    # đã là datetime (từ Excel) ở phần lớn ô
    if non_null.map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year")).mean() > 0.9:
        return pd.to_datetime(s, errors="coerce")
    # chuỗi ngày dd/mm/yyyy
    if sample.str.match(_DATE_RE).mean() > 0.95:
        iso = sample.str.match(r"^\d{4}").mean() > 0.5
        out = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", dayfirst=not iso, format="mixed")
        if out.notna().sum() >= 0.95 * len(sample):
            return out
    if is_code:
        return s.map(lambda v: v if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else nfc(v)))
    # số (kể cả ô đã là số trong Excel)
    numeric_like = non_null.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool)) | \
        sample.reindex(non_null.index).fillna("").str.match(_NUM_RE)
    if numeric_like.mean() > 0.95:
        conv = s.map(lambda v: v if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                     else (np.nan if pd.isna(v) else _to_number(str(v))))
        if conv.map(lambda v: v is None).sum() == 0:
            return pd.to_numeric(conv, errors="coerce")
    return s.map(lambda v: nfc(v) if isinstance(v, str) else v)

# --------------------------------------------------------------------------- README


def parse_readme(text: str) -> dict:
    """Lấy nghĩa cột từ README: {tên file hoặc '*': {tên cột: nghĩa}}."""
    out, current = {}, "*"
    for line in text.splitlines():
        m = re.match(r"^#+\s*Cột trong\s+`?([^`]+?)`?\s*$", line.strip())
        if m:
            current = m.group(1).strip()
            continue
        m = re.match(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", line)
        if m:
            out.setdefault(current, {})[nfc(m.group(1))] = m.group(2).strip()
    return out

# --------------------------------------------------------------------------- Business


@dataclass
class Business:
    path: str
    tables: dict = field(default_factory=dict)       # tên bảng -> DataFrame
    sources: dict = field(default_factory=dict)      # tên bảng -> danh sách "file|sheet"
    readme: str = ""
    col_meaning: dict = field(default_factory=dict)  # tên bảng -> {cột: nghĩa}
    warnings: list = field(default_factory=list)
    load_seconds: float = 0.0

    def schema_text(self, n_samples: int = 3, max_cols: int = 60) -> str:
        """Mô tả ngắn gọn các bảng cho LLM: tên cột, kiểu, ví dụ giá trị, nghĩa theo README."""
        lines = []
        for name, df in self.tables.items():
            srcs = self.sources.get(name, [])
            src_txt = srcs[0] if len(srcs) == 1 else f"{len(srcs)} mảnh, vd {srcs[0]}"
            lines.append(f"### Bảng `{name}`: {len(df):,} dòng (nguồn: {src_txt})")
            meaning = self.col_meaning.get(name, {})
            for col in list(df.columns)[:max_cols]:
                if col == "_nguon":
                    continue
                s = df[col]
                kind = ("ngày" if pd.api.types.is_datetime64_any_dtype(s) else
                        "số" if pd.api.types.is_numeric_dtype(s) else "chữ")
                vals = s.dropna().unique()[:n_samples]
                ex = ", ".join(_fmt(v) for v in vals)
                null = s.isna().mean()
                extra = f"; trống {null:.0%}" if null > 0.05 else ""
                if kind == "ngày" and s.notna().any():
                    extra += f"; từ {s.min():%Y-%m-%d} đến {s.max():%Y-%m-%d}"
                m = meaning.get(col)
                lines.append(f"- `{col}` ({kind}{extra}){' = ' + m if m else ''} | vd: {ex}")
            lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        return ", ".join(f"{n} ({len(d):,}×{d.shape[1]})" for n, d in self.tables.items())


def _fmt(v):
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M") if getattr(v, "hour", 0) or getattr(v, "minute", 0) else v.strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    t = str(v)
    return t if len(t) <= 40 else t[:37] + "..."


def _cache_key(path: str) -> str:
    h = hashlib.md5()
    for root, _, files in os.walk(path):
        for f in sorted(files):
            p = os.path.join(root, f)
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(b"v1")
    return h.hexdigest()


def load_business(path: str, use_cache: bool = True) -> Business:
    t0 = time.time()
    key = _cache_key(path)
    cache_file = os.path.join(CACHE_DIR, f"{os.path.basename(os.path.normpath(path))}_{key}.pkl")
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    biz = Business(path=path)
    rp = os.path.join(path, "README.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8", errors="replace") as f:
            biz.readme = f.read()
    meanings = parse_readme(biz.readme)

    parts = []   # (tên bảng, file, sheet, DataFrame)
    for fname in sorted(os.listdir(path)):
        fp = os.path.join(path, fname)
        ext = fname.lower().rsplit(".", 1)[-1]
        try:
            if ext == "csv":
                df = _frame_from_raw(_read_csv_raw(fp))
                parts.append((_table_name(fname, None, 1), fname, None, df))
            elif ext in ("xlsx", "xlsm", "xls"):
                sheets = _read_excel_raw(fp)
                for sh, raw in sheets.items():
                    df = _frame_from_raw(raw)
                    if not df.empty:
                        parts.append((_table_name(fname, sh, len(sheets)), fname, sh, df))
            elif ext in ("sqlite", "db", "sqlite3"):
                con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True)
                try:
                    names = [r[0] for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                    for t in names:
                        parts.append((t, fname, t, pd.read_sql_query(f'SELECT * FROM "{t}"', con)))
                finally:
                    con.close()
        except Exception as e:                # không dừng cả shop vì 1 file lỗi
            biz.warnings.append(f"Lỗi đọc {fname}: {e}")

    # gom các mảnh cùng tên bảng + cùng bộ cột
    groups: dict = {}
    for tname, fname, sh, df in parts:
        if df.empty:
            continue
        colkey = tuple(sorted(fold(c) for c in df.columns))
        gkey = (tname, colkey)
        groups.setdefault(gkey, []).append((fname, sh, df))

    used = {}
    for (tname, _), items in groups.items():
        name = tname
        if name in used:                      # cùng tên nhưng khác cột -> đặt tên phân biệt
            used[name] += 1
            name = f"{tname}_{used[tname]}"
        else:
            used[name] = 0
        cols = list(items[0][2].columns)
        frames = []
        for fname, sh, df in items:
            df = df.copy()
            # sắp cột theo bảng đầu tiên (so khớp không phân biệt hoa thường/dấu)
            fmap = {fold(c): c for c in df.columns}
            df = df[[fmap[fold(c)] for c in cols]]
            df.columns = cols
            df["_nguon"] = f"{fname}|{sh}" if sh is not None else fname
            frames.append(df)
        table = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        for c in table.columns:
            if c != "_nguon":
                try:
                    table[c] = _convert_column(c, table[c])
                except Exception as e:
                    biz.warnings.append(f"Không chuyển kiểu được {name}.{c}: {e}")
        biz.tables[name] = table
        biz.sources[name] = [f"{f}|{s}" if s is not None else f for f, s, _ in items]
        # nghĩa cột: theo đúng file trong README, không có thì dùng mục chung
        m = {}
        for fname, _, _ in items:
            m.update(meanings.get(fname, {}))
        for k, v in meanings.get("*", {}).items():
            m.setdefault(k, v)
        fm = {fold(k): v for k, v in m.items()}
        biz.col_meaning[name] = {c: fm[fold(c)] for c in table.columns if fold(c) in fm}

    biz.load_seconds = round(time.time() - t0, 1)
    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(biz, f)
        except OSError:
            pass
    return biz


if __name__ == "__main__":
    import sys
    b = load_business(sys.argv[1], use_cache=False)
    print(f"Đọc xong trong {b.load_seconds}s: {b.summary()}")
    for w in b.warnings:
        print("CẢNH BÁO:", w)
    print(b.schema_text())
