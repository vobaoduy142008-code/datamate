"""
Bộ đọc dữ liệu chung cho MỌI cửa hàng (không viết riêng cho shop nào - đúng luật cuộc thi).

    from loader import load_business
    biz = load_business("data/train/data/biz_101")
    biz.tables            # dict: tên bảng -> pandas DataFrame đã làm sạch
    biz.schema_text()     # bản mô tả các bảng (kèm ghi chú README) để đưa vào prompt LLM
    biz.notes             # phần "Quy ước"/ghi chú của README (không gồm danh sách cột)
    biz.data_end          # ngày cuối có dữ liệu theo README ("Dữ liệu có đến hết ...") hoặc None

Các bước làm sạch (giải thích cho người mới):
  1. Đọc mọi file .csv / .xlsx (tất cả sheet) / .sqlite trong thư mục shop.
  2. Tìm dòng tên cột, bỏ dòng trống, dòng tiêu đề, dòng "Tổng cộng", dòng lặp tên cột.
  3. Ghép các mảnh cùng một bảng (mỗi tháng 1 sheet, file _1/_2, file theo tháng/năm).
     Cột "_nguon" = file|sheet cho biết dòng đến từ đâu.
  4. Điền xuống ô trống theo nhóm (rulebook §8): các cột cấp đơn (mã đơn, ngày, khách...)
     chỉ ghi ở dòng đầu đơn thì được điền cho các dòng sau của cùng đơn.
  5. Chuẩn hoá kiểu dữ liệu:
       - ngày: datetime Excel, số serial Excel (45658), "1 thg 6 2025", "01/02/2025",
         "1/1" (không năm -> lấy năm theo tên sheet/file), lẫn kiểu tháng/ngày -> chọn cách
         đọc khớp với các dòng xung quanh.
       - tiền: "1.234.567đ", "1tr440", "16,56tr", "396k", "20,1tr" -> số.
       - số lượng ghi kèm đơn vị: "3 chai", "1 bao 25kg" -> số + cột "(đơn vị ghi kèm)".
       - cột mã (mã hàng, SĐT, IMEI, MST, biển số...) giữ nguyên dạng chữ.
  6. Lấy nghĩa từng cột từ README (mục "Cột trong `file`"), phần còn lại là ghi chú.
  7. Lưu cache (.pkl trong /tmp) để lần sau đọc lại tức thì.
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
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.environ.get("DATAMATE_CACHE", "/tmp/datamate_cache")
LOADER_VERSION = "v2.4"

# --------------------------------------------------------------------------- tiện ích chữ


def fold(s) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng: 'Mã Hóa Đơn' -> 'ma hoa don'."""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def _isnull(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False

# --------------------------------------------------------------------------- đọc file thô


def _read_excel_raw(path):
    """{tên sheet: DataFrame không header}. Dùng calamine (nhanh) nếu có, không thì openpyxl."""
    try:
        return pd.read_excel(path, sheet_name=None, header=None, engine="calamine")
    except (ImportError, ValueError):
        return pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")


def _read_csv_raw(path):
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, header=None, dtype=object, encoding=enc,
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
        name = "" if _isnull(v) else re.sub(r"\s+", " ", nfc(v))
        name = name or f"cot_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    return cols


# Dòng tổng/cộng chèn giữa dữ liệu: "Tổng ngày 01/01", "Cộng ngày 01/01/2025", "tổng 01/05", "Tổng tháng",
# "Cộng tháng 05/2025", "Cộng", "tong cong". So khớp CÓ DẤU (chữ thường, NFC) để không nhầm họ "Tống"
# (Tống Thắng, Tống Hằng...) hay "Công ty ...". Chữ không dấu chỉ nhận khi theo sau là ngày/tháng/số.
_TOT_TAIL = r"(\s*$|\s*[:\-(=]|\s*\d|\s+(cộng|ngày|tháng|tuần|quý|năm|kỳ|ca|tiền|doanh thu|số|hóa đơn|hoá đơn|đơn|phiếu|bill|giá trị|thanh toán)(\s|$|:))"
_TOTAL_RE = re.compile(r"^(tổng cộng|tổng|cộng)" + _TOT_TAIL + r"|^(tong cong|tong|cong)(\s*$|\s*[:\-(=]|\s*\d|\s+(cong|ngay|thang|tuan|quy|nam|ky)(\s|$|:))"
                       r"|^(grand total|total|sum)(\s*$|\s*[:=])")


def _is_total_text(x) -> bool:
    return isinstance(x, str) and to_number(x) is None and bool(_TOTAL_RE.match(nfc(x).lower()))


def _frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Bảng thô (không header) -> bảng có tên cột, đã bỏ dòng tiêu đề/trống/tổng/lặp header."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()
    h = _guess_header_row(raw)
    cols = _clean_columns(raw.iloc[h].tolist())
    df = raw.iloc[h + 1:].copy()
    df.columns = cols
    header_fold = [fold(c) for c in cols]
    first = df.iloc[:, 0]
    cand = first.map(lambda v: isinstance(v, str) and fold(v) == header_fold[0])
    if cand.any():
        again = df[cand].apply(lambda r: [fold(x) for x in r.tolist()] == header_fold, axis=1)
        df = df.drop(again[again].index)
    n_filled = df.notna().sum(axis=1)
    # dòng tổng luôn thiếu vài ô (mã đơn, ngày...): xét các dòng có ít nhất 1 ô trống và <=80% ô có dữ liệu
    sparse = df[(n_filled < df.shape[1]) & (n_filled <= 0.8 * df.shape[1])]
    if len(sparse):
        tot = sparse.apply(lambda r: any(_is_total_text(x) for x in r.dropna()), axis=1)
        df = df.drop(tot[tot].index)
        n_total = int(tot.sum())
    else:
        n_total = 0
    df = df.reset_index(drop=True)
    df.attrs["n_total"] = n_total
    return df

# --------------------------------------------------------------------------- gom mảnh thành bảng


_MONTHY = re.compile(
    r"^(sheet\s*\d*|t\s*\d{1,2}([\s._\-/]*\d{2,4})?|thang\s*\d{1,2}([\s._\-/]*\d{2,4})?|"
    r"\d{1,2}[\s._\-/]+\d{4}|\d{4}[\s._\-/]+\d{1,2}|\d{4})$")
_SUFFIX = re.compile(r"([\s_\-]+(\d{1,3}|\d{4}|t\d{1,2}[\s_\-]*\d{4}|thang\d{1,2}[\s_\-]*\d{4}))$", re.I)


def _base_name(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    prev = None
    while prev != stem:
        prev = stem
        stem = _SUFFIX.sub("", stem)
    return stem


def _table_name(file_name: str, sheet, n_sheets: int) -> str:
    base = _base_name(file_name)
    if sheet is None or _MONTHY.match(fold(sheet)):
        return base
    return nfc(sheet) if n_sheets > 1 or fold(sheet) != "sheet1" else base


def _year_month_hint(source: str):
    """Từ 'SoBanHang.xlsx|Tháng 01-2025' hoặc 'BanHang_T10_2025.xlsx|...' -> (2025, 1 hoặc 10)."""
    s = fold(source)
    m = re.search(r"(?:^|[^\d])(\d{1,2})[\s._\-/]+(20\d{2})(?:[^\d]|$)", s) or \
        re.search(r"(?:t|thang)\s*(\d{1,2})[\s._\-]*(20\d{2})", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"(20\d{2})[\s._\-/]+(\d{1,2})(?:[^\d]|$)", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(20\d{2})", s)
    return (int(m.group(1)), None) if m else (None, None)

# --------------------------------------------------------------------------- nhận diện loại cột


_CODE_HINT = re.compile(
    r"(^|[\s_(])(ma|id|code|sku|barcode|ean|so dien thoai|dien thoai|sdt|so dt|phone|di dong|imei|"
    r"bien so|so ct|so phieu|so hd|so hoa don|cccd|cmnd|mst|ma so thue|so the|ma lo|so lo|"
    r"so chung tu|ma vach)([\s_)]|$)")
# từ ngắn dễ trùng ("Giảm giá HĐ", "Tổng bill"): chỉ coi là mã khi đứng MỘT MÌNH làm tên cột
_CODE_EXACT = re.compile(r"^(hd|bill|lo|don|ma)$")
# tên cột tiền/số lượng: không bao giờ là mã (trừ khi bắt đầu bằng "mã")
_MONEY_WORD = re.compile(r"(^|\s)(tien|gia|tong|giam|chiet khau|thanh toan|phai thu|phai tra|con no|du no|"
                         r"no|tra|thu|chi|so luong|sl|vat|thue|phi|coc|luong)(\s|$)")
_DATE_HINT = re.compile(r"(^|\s)(ngay|thoi gian|thoi diem|date|time|han|hsd|sinh nhat|tu ngay|den ngay|"
                        r"ngay gio|tao luc|ngaylap|ngayban|ngay ban)(\s|$)")
_NOT_DATE = re.compile(r"(so ngay|\(ngay\)|ky han|han no \(|han gop \(|gio$|^gio )")
_QTY_HINT = re.compile(r"(^|\s)(so luong|sl|slg|sluong|soluong|s\.l|qty|so luong san pham|sl ban)(\s|$)")


def _fname(name):
    """Tên cột -> dạng so khớp: 'SoDienThoai' -> 'so dien thoai', 'ma_kh' -> 'ma kh'."""
    name = str(name)
    name = "".join(" " + ch if i and ch.isupper() and name[i - 1].islower() else ch for i, ch in enumerate(name))
    return fold(name).replace("_", " ")


def _is_code(name, meaning=""):
    n = _fname(name)
    if _CODE_EXACT.match(n):
        return True
    if _CODE_HINT.search(n):
        if _MONEY_WORD.search(n) and not re.match(r"^(ma|so dien thoai|dien thoai|sdt|imei|mst|ma so thue)(\s|$)", n):
            return False
        return True
    return bool(re.search(r"^(ma|so dien thoai|imei|bien so)(\s|$)", fold(meaning)))


_PHONE_HINT = re.compile(r"(^|\s)(so dien thoai|dien thoai|sdt|so dt|phone|di dong|mobile|tel)(\s|$)")
_EXCEL_ERR = {"#value!", "#n/a", "#ref!", "#div/0!", "#name?", "#null!", "#num!"}


def _is_phone(name, meaning=""):
    return bool(_PHONE_HINT.search(_fname(name)) or re.match(r"^so dien thoai", fold(meaning)))


def _norm_phone(v):
    """974532329 / 974532329.0 / '0974 532 329' -> '0974532329' (số di động VN 10 chữ số, mất số 0 đầu khi lưu dạng số)."""
    if _isnull(v):
        return v
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    t = re.sub(r"^(\d+)\.0+$", r"\1", nfc(v))          # '974532329.0' (CSV ghi số thực)
    d = re.sub(r"[\s.\-()]", "", t)
    if re.fullmatch(r"\+?84\d{9}", d):
        return "0" + d[-9:]
    if re.fullmatch(r"[1-9]\d{8}", d):
        return "0" + d
    return d if re.fullmatch(r"\d{9,11}", d) else t


def _clean_text(v):
    if not isinstance(v, str):
        return v
    t = re.sub(r"\s+", " ", nfc(v)).strip()
    return np.nan if (not t or t.lower() in _EXCEL_ERR) else t


def _is_date_name(name, meaning=""):
    n, m = _fname(name), fold(meaning)
    n2 = _fname(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name))   # NgayLap -> Ngay Lap
    if _NOT_DATE.search(n) or re.search(r"^(so ngay|gio )", m):
        return False
    return bool(_DATE_HINT.search(n) or _DATE_HINT.search(n2) or re.search(r"^(ngay|thoi diem|han)(\s|$)", m))

# --------------------------------------------------------------------------- số và tiền


_MONEY_TR = re.compile(r"^(\d+)tr(\d{1,3})$")
_UNIT_MULT = [(re.compile(r"^(\d+(?:[.,]\d+)?)(tr|trieu|m)$"), 1e6),
              (re.compile(r"^(\d+(?:[.,]\d+)?)(k|nghin|ngan)$"), 1e3)]


def to_number(v):
    """Chuỗi số kiểu Việt Nam -> float. Trả về None nếu không phải số."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    t = fold(v).replace(" ", "")
    t = re.sub(r"(vnd|d)$", "", t)
    if not t:
        return None
    neg = t.startswith("-")
    t = t.lstrip("+-")
    m = _MONEY_TR.match(t)
    if m:                                              # 1tr440 -> 1.440.000 ; 19tr990 ; 1tr5 -> 1.500.000
        v = int(m.group(1)) * 1e6 + int(m.group(2).ljust(3, "0")) * 1e3
        return -v if neg else v
    for rx, mult in _UNIT_MULT:                        # 16,56tr ; 396k ; 2tr
        m = rx.match(t)
        if m:
            v = float(m.group(1).replace(",", ".")) * mult
            return -round(v) if neg else round(v)
    if not re.fullmatch(r"[\d.,]+", t):
        return None
    if "." in t and "," in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        parts = t.split(sep)
        if len(parts) > 2 or (len(parts[-1]) == 3 and parts[0] not in ("0", "")):
            t = t.replace(sep, "")                     # 1.234.567 (hàng nghìn)
        else:
            t = t.replace(",", ".")                    # 18,6 (thập phân)
    try:
        x = float(t)
    except ValueError:
        return None
    return -x if neg else x

# --------------------------------------------------------------------------- ngày


_EXCEL0 = datetime(1899, 12, 30)
_RX_THG = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})[\s,]+(\d{4})$")
_RX_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_RX_YMD = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?$")
_RX_DM = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})$")


def _mk(y, m, d, hh=0, mi=0, ss=0):
    try:
        y = int(y)
        y = y + 2000 if y < 100 else y
        return datetime(y, int(m), int(d), int(hh or 0), int(mi or 0), int(ss or 0))
    except (ValueError, TypeError):
        return None


def _parse_date_value(v, year_hint):
    """-> (datetime | None, datetime_thay_the | None). Có thay thế khi d/m và m/d đều hợp lệ."""
    if isinstance(v, (datetime, pd.Timestamp)):
        return (pd.Timestamp(v).to_pydatetime(), None)
    if isinstance(v, date):
        return (datetime(v.year, v.month, v.day), None)
    if hasattr(v, "hour") and not hasattr(v, "year"):     # chỉ có giờ (datetime.time)
        return (None, None)
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        if 20000 <= float(v) <= 80000:                 # số serial Excel
            return (_EXCEL0 + timedelta(days=float(v)), None)
        return (None, None)
    t = nfc(v).lower()
    if re.fullmatch(r"\d{5}(\.\d+)?", t) and 20000 <= float(t) <= 80000:
        return (_EXCEL0 + timedelta(days=float(t)), None)
    m = _RX_THG.match(t)
    if m:
        return (_mk(m[3], m[2], m[1]), None)
    m = _RX_YMD.match(t)
    if m:
        return (_mk(m[1], m[2], m[3], m[4], m[5], m[6]), None)
    m = _RX_DMY.match(t)
    if m:
        a, b = int(m[1]), int(m[2])
        d1 = _mk(m[3], b, a, m[4], m[5], m[6])                    # ngày/tháng (kiểu Việt Nam)
        d2 = _mk(m[3], a, b, m[4], m[5], m[6]) if a != b else None  # tháng/ngày
        if d1 and d2:
            return (d1, d2)
        return (d1 or d2, None)
    m = _RX_DM.match(t)
    if m and year_hint:
        a, b = int(m[1]), int(m[2])
        d1, d2 = _mk(year_hint, b, a), (_mk(year_hint, a, b) if a != b else None)
        return (d1, d2) if (d1 and d2) else (d1 or d2, None)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t):         # chỉ có giờ
        return (None, None)
    try:                                               # dạng khác (vd '2025-01-01 10:30:00.000')
        x = pd.to_datetime(t, dayfirst=True)
        return (x.to_pydatetime(), None) if not pd.isna(x) else (None, None)
    except (ValueError, TypeError, OverflowError):
        return (None, None)


# cột ngày có thể nằm ở tương lai (hạn, hết hạn, dự kiến...): không dùng mốc "dữ liệu đến hết"
_FUTURE_DATE = re.compile(r"(^|\s)(han|hsd|het han|den ngay|du kien|hen|bao hanh|tai kham|ngay phai tra|ngay tra|lich)(\s|$)")


def _convert_dates(s: pd.Series, sources: pd.Series, data_end=None):
    """Chuyển cột sang datetime. Trả về (Series mới, số ô phải đoán thứ tự ngày/tháng) hoặc (None, 0).
    data_end: ngày cuối dữ liệu (README). Ô mơ hồ mà 1 cách đọc vượt quá mốc này thì chọn cách còn lại."""
    idx = s.index
    vals = s.tolist()
    src = sources.tolist() if sources is not None else [""] * len(vals)
    hint_cache, primary, alt = {}, [], []
    for v, so in zip(vals, src):
        if _isnull(v) or (isinstance(v, str) and not v.strip()):
            primary.append(None)
            alt.append(None)
            continue
        if so not in hint_cache:
            hint_cache[so] = _year_month_hint(so)
        d1, d2 = _parse_date_value(v, hint_cache[so][0])
        primary.append(d1)
        alt.append(d2)
    n_val = sum(1 for v in vals if not _isnull(v) and str(v).strip())
    n_ok = sum(1 for d in primary if d is not None)
    if n_val == 0 or n_ok < 0.9 * n_val:
        return None, 0
    p = pd.Series(primary, index=idx, dtype=object)
    a = pd.Series(alt, index=idx, dtype=object)
    amb = a.notna()
    n_amb = int(amb.sum())
    if n_amb:
        # tham chiếu 1: tháng của sheet/file (nếu có)
        hint_m = pd.Series([hint_cache[so][1] if so in hint_cache else None for so in src], index=idx)
        # tham chiếu 2: ngày không mơ hồ gần nhất (theo thứ tự dòng, trong cùng nguồn)
        sure = pd.to_datetime(p.where(~amb), errors="coerce")
        grp = sources if sources is not None else pd.Series("", index=idx)
        ref = sure.groupby(grp.values).ffill()
        ref = ref.fillna(sure.groupby(grp.values).bfill())
        choose_alt = []
        end = pd.Timestamp(data_end) + pd.Timedelta(days=1) if data_end is not None else None
        for i in idx[amb]:
            d1, d2, hm, r = p[i], a[i], hint_m[i], ref[i]
            if end is not None and (d1 >= end) != (d2 >= end):
                choose_alt.append(d1 >= end)           # cách đọc kia vượt quá ngày cuối dữ liệu
            elif hm and d1.month != d2.month and (d1.month == hm) != (d2.month == hm):
                choose_alt.append(d2.month == hm)
            elif not pd.isna(r):
                choose_alt.append(abs((d2 - r).days) < abs((d1 - r).days))
            else:
                choose_alt.append(False)
        sel = pd.Series(choose_alt, index=idx[amb])
        p.loc[sel[sel].index] = a.loc[sel[sel].index]
    return pd.to_datetime(p, errors="coerce"), n_amb


def _split_qty(s: pd.Series):
    """'3 chai' -> (3, 'chai'). Trả về (Series số, Series đơn vị) hoặc None."""
    txt = s.dropna().astype(str).str.strip()
    if txt.empty:
        return None
    m = txt.str.extract(r"^(-?\d+(?:[.,]\d+)?)\s*(.*)$")
    if m[0].notna().mean() < 0.97 or (m[1].fillna("") == "").all():
        return None
    num = pd.Series(np.nan, index=s.index)
    num.loc[m.index] = m[0].str.replace(",", ".").astype(float)
    unit = pd.Series(np.nan, index=s.index, dtype=object)
    u = m[1].where(m[1].fillna("") != "")
    unit.loc[u.index] = u
    return num, unit


def _convert_column(name, s: pd.Series, sources: pd.Series, meaning: str = "", data_end=None):
    """-> (Series đã chuyển, cột phụ hoặc None, ghi chú hoặc None)."""
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s, None, None
    non_null = s.dropna()
    if non_null.empty:
        return s, None, None
    if _is_date_name(name, meaning):
        fut = _FUTURE_DATE.search(_fname(name)) or _FUTURE_DATE.search(fold(meaning))
        conv, n_amb = _convert_dates(s, sources, None if fut else data_end)
        if conv is not None:
            note = f"{n_amb} ô ngày mơ hồ ngày/tháng, đã chọn theo tên sheet/ngày cuối dữ liệu/dòng lân cận" if n_amb else None
            return conv, None, note
    if _is_phone(name, meaning):
        return s.astype(object).map(_norm_phone), None, None
    if pd.api.types.is_numeric_dtype(s):
        return s, None, None
    s = s.astype(object)
    sample = non_null.astype(str).str.strip()
    low = set(sample.str.lower().unique()[:5])
    if low and low <= {"true", "false"}:
        return s.map(lambda v: v if _isnull(v) else str(v).strip().lower() == "true"), None, None
    if _is_code(name, meaning):
        return s.map(lambda v: v if _isnull(v) else
                     (str(int(v)) if isinstance(v, float) and v.is_integer() else _clean_text(str(v)))), None, None
    # chuỗi ngày dù tên cột không gợi ý
    if sample.head(500).str.match(r"^(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s*thg)").mean() > 0.9:
        conv, _ = _convert_dates(s, sources, data_end)
        if conv is not None:
            return conv, None, None
    # số / tiền: thử trên mẫu trước cho nhanh
    head = non_null.head(300)
    if head.map(lambda v: to_number(v) is not None).mean() >= 0.9:
        conv = s.map(lambda v: np.nan if _isnull(v) else to_number(v))
        bad = conv.map(lambda v: v is None).sum()
        if bad <= 0.01 * len(non_null):
            return pd.to_numeric(conv.map(lambda v: np.nan if v is None else v), errors="coerce"), None, \
                (f"{bad} ô không đọc được số" if bad else None)
    # số lượng ghi kèm đơn vị
    if _QTY_HINT.search(_fname(name)) or "so luong" in fold(meaning):
        r = _split_qty(s)
        if r is not None:
            return r[0], r[1], None
    return s.map(_clean_text), None, None

# --------------------------------------------------------------------------- điền xuống theo nhóm (§8)


_ID_NAME = re.compile(r"^(ma (don|hoa don|hd|bill|phieu|chung tu|don hang|giao dich|so don)|"
                      r"so (phieu|ct|hd|hoa don|chung tu|don|bill)|hd|bill|don so|order id|order|invoice|stt|ma bill|ma don)$")


def _ffill_groups(df: pd.DataFrame, meanings: dict, samples: list = None) -> list:
    """Điền xuống các cột cấp đơn bị để trống ở dòng sau của cùng một nhóm. Trả về tên cột đã điền.
    samples (nếu truyền vào): nhận vài dòng mẫu TRƯỚC khi điền, để người kiểm tra xem lại."""
    cols = [c for c in df.columns if c != "_nguon"]
    if len(df) < 3:
        return []
    blanks = {c: df[c].isna() | (df[c].astype(str).str.strip() == "") for c in cols}
    share = {c: blanks[c].mean() for c in cols}
    cand = [c for c in cols if 0.02 < share[c] < 0.95]
    if not cand:
        return []
    clusters = []
    for c in cand:                                     # gom các cột có cùng mẫu ô trống
        for cl in clusters:
            if (blanks[c] == blanks[cl[0]]).mean() >= 0.98:
                cl.append(c)
                break
        else:
            clusters.append([c])

    def id_like(c):
        return bool(_ID_NAME.match(_fname(c))) or fold(meanings.get(c, "")).startswith(("ma don", "so thu tu don"))

    full_ids = [c for c in cols if id_like(c) and share[c] <= 0.02]
    src = df["_nguon"] if "_nguon" in df else pd.Series("", index=df.index)
    done = []
    for cl in clusters:
        mask = blanks[cl[0]]
        if mask.iloc[0]:
            continue
        if any(id_like(c) for c in cl):
            ok = True                                  # chính mã đơn bị trống ở dòng sau -> dòng tiếp nối
        elif full_ids:
            # chỉ điền khi trong mỗi nhóm cùng mã, dòng đầu có giá trị và các dòng sau trống
            idc = df[full_ids[0]].astype(str)
            gid = ((idc != idc.shift()) | (src != src.shift())).cumsum()
            g = pd.DataFrame({"gid": gid, "blank": mask})
            gg = g.groupby("gid")["blank"].agg(["first", "sum", "size"])
            gg = gg[gg["sum"] > 0]
            ok = len(gg) > 0 and ((~gg["first"]) & (gg["sum"] == gg["size"] - 1)).mean() >= 0.9
        else:
            ok = len(cl) >= 2
        if not ok:
            continue
        if samples is not None:
            pos = int(np.argmax(mask.values))         # dòng trống đầu tiên của cụm
            lo, hi = max(0, pos - 2), min(len(df), pos + 3)
            show = list(dict.fromkeys(full_ids[:1] + cl + [c for c in cols if c not in cl][:4]))
            samples.append((cl, df.iloc[lo:hi][show].copy()))
        for c in cl:
            ser = df[c].where(~blanks[c])
            df[c] = ser.groupby(src.values).ffill()
        done.extend(cl)

    # Bước 2: cột "cấp đơn" chỉ ghi ở một số dòng của đơn (vd dòng Voucher/Phí ship để trống Ngày, Khách).
    # Điền trong cùng mã đơn nếu có BẰNG CHỨNG cột này là cấp đơn: ở >=20 đơn có >=2 ô có giá trị,
    # và >=99% các đơn đó mọi ô có giá trị đều giống nhau. Cột cấp dòng (đơn giá, SL) không qua được.
    ids = sorted([c for c in cols if id_like(c) and df[c].notna().mean() >= 0.98],
                 key=lambda c: _fname(c) == "stt")             # ưu tiên mã đơn thật, STT để sau
    if ids:
        idc = df[ids[0]].astype(str)
        gid = ((idc != idc.shift()) | (src != src.shift())).cumsum().values
        for c in cand:
            if c in done or c in ids:
                continue
            b = df[c].isna() | (df[c].astype(str).str.strip() == "")
            if not b.any():
                continue
            vals = df[c].where(~b)
            if _is_date_name(c, meanings.get(c, "")):
                # cùng một ngày có thể ghi nhiều kiểu trong một đơn (01/01/2026, 1 thg 1 2026, 46023, 1/1):
                # mỗi ô -> tập các ngày có thể (ngày/tháng và tháng/ngày); đơn "nhất quán" nếu các tập có phần chung
                hints = {}
                inter, cnt = {}, {}
                for g_, v, so in zip(gid, vals.tolist(), src.tolist()):
                    if _isnull(v):
                        continue
                    if so not in hints:
                        hints[so] = _year_month_hint(so)[0]
                    d1, d2 = _parse_date_value(v, hints[so])
                    cs = frozenset(x.date() for x in (d1, d2) if x is not None) or frozenset([fold(v)])
                    inter[g_] = inter[g_] & cs if g_ in inter else cs
                    cnt[g_] = cnt.get(g_, 0) + 1
                multi_g = [g_ for g_, n in cnt.items() if n >= 2]
                if len(multi_g) < 20 or sum(1 for g_ in multi_g if inter[g_]) / len(multi_g) < 0.99:
                    continue
            else:
                key = vals.map(lambda v: v if _isnull(v) else fold(v))
                g = key.groupby(gid)
                cnt, nun = g.count(), g.nunique()
                multi = cnt >= 2
                if multi.sum() < 20 or (nun[multi] == 1).mean() < 0.99:
                    continue
            new = vals.groupby(gid).ffill().groupby(gid).bfill()
            changed = new.notna() & vals.isna()
            if not changed.any():
                continue                               # đơn nào trống thì trống cả đơn (vd khách lẻ): không có gì để điền
            if samples is not None:
                pos = int(np.argmax(changed.values))
                lo, hi = max(0, pos - 2), min(len(df), pos + 3)
                show = list(dict.fromkeys(ids[:1] + [c] + [x for x in cols if x != c][:5]))
                samples.append(([c], df.iloc[lo:hi][show].copy()))
            df[c] = new
            done.append(c)
    return done


# --------------------------------------------------------------------------- README


def parse_readme(text: str):
    """-> (nghĩa cột {file|'*': {cột: nghĩa}}, ghi chú dạng chữ, ngày cuối dữ liệu)."""
    meanings, notes, current = {}, [], None
    for line in text.splitlines():
        s = line.strip()
        h = re.match(r"^#+\s*(.*)$", s)
        if h:
            m = re.match(r"^Cột trong\s+`?([^`]+?)`?\s*$", h.group(1).strip())
            current = m.group(1).strip() if m else None
            if not m:
                notes.append(line)
            continue
        m = re.match(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", line)
        if current is not None:
            if m:
                meanings.setdefault(current, {})[nfc(m.group(1))] = m.group(2).strip()
            elif s:
                meanings.setdefault(current, {})  # dòng khác trong mục cột: bỏ qua
            continue
        notes.append(line)
        if m:                                          # gạch đầu dòng trong ghi chú: có thể là nghĩa cột
            meanings.setdefault("*", {})[nfc(m.group(1))] = m.group(2).strip()
    end = None
    m = re.search(r"có đến hết (?:ngày )?(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        end = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    notes_txt = re.sub(r"\n{3,}", "\n\n", "\n".join(notes)).strip()
    return meanings, notes_txt, end

# --------------------------------------------------------------------------- Business


@dataclass
class Business:
    path: str
    tables: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    readme: str = ""
    notes: str = ""
    data_end: object = None
    col_meaning: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    ffill_samples: dict = field(default_factory=dict)   # {bảng: [(cột đã điền, DataFrame mẫu trước khi điền)]}
    dup_rows: dict = field(default_factory=dict)        # {bảng: số dòng trùng y hệt (rulebook §8: tính 1 lần)}
    load_seconds: float = 0.0

    def schema_text(self, n_samples: int = 3, max_cols: int = 60, with_notes: bool = True) -> str:
        lines = []
        if with_notes and self.notes:
            lines += ["## Ghi chú của cửa hàng (README)", self.notes, ""]
        lines.append("## Các bảng dữ liệu (pandas DataFrame, tên bảng là khoá của dict `tables`)")
        for name, df in self.tables.items():
            srcs = self.sources.get(name, [])
            src_txt = srcs[0] if len(srcs) == 1 else f"{len(srcs)} mảnh ghép lại, vd {srcs[0]}"
            lines.append(f"### Bảng `{name}`: {len(df):,} dòng (nguồn: {src_txt})")
            meaning = self.col_meaning.get(name, {})
            for col in list(df.columns)[:max_cols]:
                if col == "_nguon":
                    continue
                s = df[col]
                kind = ("ngày" if pd.api.types.is_datetime64_any_dtype(s) else
                        "đúng/sai" if pd.api.types.is_bool_dtype(s) else
                        "số" if pd.api.types.is_numeric_dtype(s) else "chữ")
                null = s.isna().mean()
                if null >= 0.999:
                    lines.append(f"- `{col}` (trống hoàn toàn)")
                    continue
                vals = s.dropna().unique()[:n_samples]
                ex = ", ".join(_fmt(v) for v in vals)
                extra = f"; trống {null:.0%}" if null > 0.05 else ""
                if kind == "ngày" and s.notna().any():
                    extra += f"; từ {s.min():%Y-%m-%d} đến {s.max():%Y-%m-%d}"
                if kind == "chữ":
                    nu = s.nunique()
                    if nu <= 12:
                        ex = ", ".join(_fmt(v) for v in s.dropna().unique()[:12])
                        extra += f"; {nu} giá trị khác nhau"
                m = meaning.get(col)
                lines.append(f"- `{col}` ({kind}{extra}){' = ' + m if m else ''} | vd: {ex}")
            lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        return ", ".join(f"{n} ({len(d):,}×{d.shape[1]})" for n, d in self.tables.items())


def _fmt(v):
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d %H:%M") if (v.hour or v.minute) else v.strftime("%Y-%m-%d")
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
    h.update(LOADER_VERSION.encode())
    return h.hexdigest()


def load_business(path: str, use_cache: bool = True) -> Business:
    t0 = time.time()
    key = _cache_key(path)
    cache_file = os.path.join(CACHE_DIR, f"{os.path.basename(os.path.normpath(path))}_{key}.pkl")
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    biz = Business(path=path)
    rp = os.path.join(path, "README.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8", errors="replace") as f:
            biz.readme = f.read()
    meanings, biz.notes, biz.data_end = parse_readme(biz.readme)

    parts = []
    for fname in sorted(os.listdir(path)):
        fp = os.path.join(path, fname)
        ext = fname.lower().rsplit(".", 1)[-1]
        try:
            if ext == "csv":
                parts.append((_table_name(fname, None, 1), fname, None, _frame_from_raw(_read_csv_raw(fp))))
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
        except Exception as e:
            biz.warnings.append(f"Lỗi đọc {fname}: {e}")

    groups: dict = {}
    for tname, fname, sh, df in parts:
        if df.empty:
            continue
        colkey = tuple(sorted(fold(c) for c in df.columns))
        groups.setdefault((tname, colkey), []).append((fname, sh, df))
        n_tot = df.attrs.get("n_total", 0)
        if n_tot:
            biz.warnings.append(f"{fname}{'|' + str(sh) if sh is not None else ''}: đã bỏ {n_tot} dòng tổng/cộng (vd 'Tổng ngày ...')")

    used = {}
    for (tname, _), items in groups.items():
        name = tname
        if name in used:
            used[name] += 1
            name = f"{tname}_{used[tname]}"
        else:
            used[name] = 0
        cols = list(items[0][2].columns)
        frames = []
        for fname, sh, df in items:
            fmap = {fold(c): c for c in df.columns}
            df = df[[fmap[fold(c)] for c in cols]].copy()
            df.columns = cols
            df["_nguon"] = f"{fname}|{sh}" if sh is not None else fname
            frames.append(df)
        table = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)

        # nghĩa cột theo README: ưu tiên đúng file, sau đó mục chung (chỉ khi khớp tên cột)
        m = {}
        for fname, _, _ in items:
            m.update(meanings.get(fname, {}))
        fm = {fold(k): v for k, v in m.items()}
        gm = {fold(k): v for k, v in meanings.get("*", {}).items()}
        col_mean = {}
        for c in table.columns:
            if fold(c) in fm:
                col_mean[c] = fm[fold(c)]
            elif fold(c) in gm:
                col_mean[c] = gm[fold(c)]

        samples = []
        filled = _ffill_groups(table, col_mean, samples) if "_nguon" in table else []
        if filled:
            biz.warnings.append(f"{name}: đã điền xuống theo nhóm (rulebook §8) các cột {filled}")
            biz.ffill_samples[name] = samples

        extra_cols = {}
        for c in list(table.columns):
            if c == "_nguon":
                continue
            try:
                new, unit, note = _convert_column(c, table[c], table["_nguon"], col_mean.get(c, ""), biz.data_end)
                table[c] = new
                if unit is not None:
                    extra_cols[f"{c} (đơn vị ghi kèm)"] = unit
                if note:
                    biz.warnings.append(f"{name}.{c}: {note}")
            except Exception as e:
                biz.warnings.append(f"Không chuyển kiểu được {name}.{c}: {e}")
        for k, v in extra_cols.items():
            table[k] = v
            col_mean[k] = "phần chữ ghi kèm sau con số ở cột số lượng (vd 'chai', 'bao 25kg'); trống = không ghi"
        data_cols = [c for c in table.columns if c != "_nguon"]
        try:
            n_dup = int(table.duplicated(subset=data_cols).sum())
        except TypeError:
            n_dup = 0
        if n_dup:
            biz.dup_rows[name] = n_dup
            has_id = any(_ID_NAME.match(_fname(c)) or fold(col_mean.get(c, "")).startswith("ma don") for c in data_cols)
            if has_id:
                # rulebook §8: hai dòng của cùng một đơn giống nhau ở mọi ô là lỗi xuất file, tính một lần
                table = table.drop_duplicates(subset=data_cols).reset_index(drop=True)
                biz.warnings.append(f"{name}: đã bỏ {n_dup} dòng trùng y hệt dòng khác cùng mã đơn (rulebook §8: tính 1 lần)")
            else:
                biz.warnings.append(f"{name}: {n_dup} dòng giống hệt nhau nhưng bảng KHÔNG có mã đơn - có thể là các lần bán khác nhau, không gộp")
        biz.tables[name] = table
        biz.sources[name] = [f"{f}|{s}" if s is not None else f for f, s, _ in items]
        biz.col_meaning[name] = col_mean

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
        print("GHI CHÚ:", w)
    print(b.schema_text())
