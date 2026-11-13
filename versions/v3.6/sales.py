"""
BẢNG BÁN HÀNG CHUẨN (ban_hang_chuan): gom dữ liệu bán hàng của mọi kiểu shop về MỘT bảng cùng tên cột.

Vì sao cần: mỗi shop đặt tên cột một kiểu ("Thanh tien", "Trị giá", "Amount", "TriGia"...), có shop tách đơn
(header) và chi tiết (dòng hàng), có shop ghi trả hàng ở bảng riêng... Model nhỏ hay chọn nhầm cột / quên lọc.
Ở đây ta làm sẵn một lần, đúng rulebook §1 + §8, rồi đưa cho model một bảng thống nhất:

  Mỗi dòng = một dòng hàng trong đơn (shop chỉ có danh sách đơn thì mỗi dòng = một đơn).
  ma_don       mã đơn (NaN nếu sổ không có mã đơn)
  ngay         ngày bán (Timestamp, đã bỏ giờ)
  trang_thai   'hoàn thành' / 'huỷ' / 'trả hàng' / 'đang giao'  (chuẩn hoá từ trạng thái gốc)
  tra_hang     True nếu ĐƠN này có trả hàng (kể cả trả một phần, ghi ở bảng khác)
  tinh_dt      True nếu dòng được tính doanh thu (§1: đơn không huỷ, không có trả hàng)
  ma_hang, ten_hang, nhom_hang, don_vi
  so_luong     số lượng theo đơn vị của dòng;   so_luong_co_ban: đã quy ra đơn vị cơ bản (nếu biết)
  don_gia, thanh_tien (thành tiền sau giảm của dòng), gia_von (giá vốn 1 đơn vị bán, nếu có)
  ma_khach, ten_khach, nhom_khach, kenh

Cách nhận cột: ưu tiên ý nghĩa cột trong README (loader đã đọc vào biz.col_meaning), không có thì đoán theo tên cột.
Hàm build_sales(biz) KHÔNG bao giờ ném lỗi: không dựng được thì trả về None.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

from loader import fold

SALES_VERSION = "s16"          # đổi số này khi sửa file để cache của loader tự dựng lại bảng chuẩn

# ----------------------------------------------------------------------------- trạng thái đơn (rulebook §8)
ST_HUY = re.compile(r"\bhuy\b|cancel|khong giao duoc|giao that bai|\bthat bai")   # \b: "chuyen" có chứa "huy"!
ST_TRA = re.compile(r"\btra hang|\btra lai|hoan tra|hoan hang|doi tra|return|hang tra ve|khach tra\b|mang tra|\btra ve")
ST_GIAO = re.compile(r"\bdang\b|cho giao|chua giao|shipping|pending|processing|van chuyen")


def status_one(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return "hoàn thành"          # §8: sổ không ghi trạng thái / ghi chú không nhắc trạng thái = hoàn thành
    t = fold(v)
    if ST_HUY.search(t):
        return "huỷ"
    if ST_TRA.search(t):
        return "trả hàng"
    if ST_GIAO.search(t):
        return "đang giao"
    return "hoàn thành"


def status_of(series):
    s = pd.Series(series)
    m = {v: status_one(v) for v in s.dropna().unique()}
    return s.map(lambda v: "hoàn thành" if (isinstance(v, float) and math.isnan(v)) else m.get(v, status_one(v)))


# ----------------------------------------------------------------------------- nhận vai trò của cột
# (vai trò, mẫu theo ý nghĩa README, mẫu theo tên cột)  -- tất cả đã bỏ dấu, chữ thường
_ROLES = [
    ("id", r"^ma don\b|^so don\b", r"^(ma |so )?(hoa don|don hang|don|bill|giao dich|chung tu|ct|phieu|hd)$"
                                   r"|^ma (hoa don|don hang|don|bill|giao dich|chung tu|phieu|gd|hd)\b|^order id$|^mahoadon$"
                                   r"|^so (phieu|ct|hd|hoa don|chung tu)$|^invoice"),
    ("date", r"^ngay ban|^thoi diem dat|^ngay dat", r"^(ngay|ngay ban|ngay mua|ngay lap|ngaylap|ngay giao dich|ngay dat hang"
                                                    r"|ngay thang|thoi gian|ngay_ban|ngay chung tu)$"),
    ("status", r"^trang thai don", r"^(trang thai|trang thai don hang|tinh trang|tinh trang don|tinhtrang|status|xu ly"
                                   r"|trang thai xu ly|ket qua|ghi chu tt|trang_thai)$"),
    ("amount", r"^thanh tien sau giam", r"^(thanh tien|thanh tien \(d\)|t\.?tien|tri gia|trigia|tien ban|amount|tien)$"),
    ("goods", r"^tien hang cua don|^tong tien hang", r"^(tien hang|tong tien hang|tongtienhang)$"),
    ("factor", r"^he so", r"^(ti le quy doi|ty le quy doi|quy doi|bang|he so quy doi|hesoquydoi|he so qd|he so)$"),
    ("qty_base", r"^so luong da doi ra", r"quy doi|quy ra|^sl le$|^so luong le$|^sl co ban$|^slquydoi$|^sl quy ra le$"),
    ("qty", r"^so luong$", r"^(so luong|sl|slg|s\.l|soluong|so luong ban|sl ban|so luong san pham|so luong hang)$"),
    ("price", r"^don gia", r"^(don gia|gia ban|gia|gia le|gia 1 dv|giadonvi|gia don vi|price|don gia ban|gia san pham|gia niem yet)$"),
    ("item_code", r"^ma hang", r"^(ma hang|ma sp|ma sku|sku|ma mat hang|ma hh|id sp|mahanghoa|ma_sp|ma hang hoa|ma san pham"
                               r"|ma hang ban|ma san pham ban|ma mon)$"),
    ("item_name", r"^ten hang$|^ten hang ", r"^(ten hang|ten hang hoa|ten sp|ten san pham|san pham|mat hang|hang ban|noi dung"
                                           r"|chi tiet hang|hang|hang hoa|product|ten mon|mon|tengoi)$"),
    ("group", r"^nhom hang", r"^(nhom hang|nhom hang hoa|nhom hang\(3 cap\)|nhom hang \(3 cap\)|danh muc|danhmuc|loai san pham"
                             r"|phan loai)$"),
    ("unit", r"^don vi tinh|^don vi lon", r"^(don vi tinh|dvt|d\.vi|don vi|donvi|donvitinh|don vi quy doi|dv lon|dv)$"),
    ("cust_name", r"^ten khach", r"^(ten khach|ten khach hang|khach hang|khach|ben mua|doi tac|tenkh|ten kh|khach hang \(ten\))$"),
    ("cust_code", r"^ma khach", r"^(ma kh|ma khach|ma khach hang|customer id|so kh|makhachhang|ma kh mua|ma khach mua)$"),
    ("cust_group", r"^nhom khach", r"^(nhom khach|nhom khach hang|doi tuong kh|loai hinh khach|hang)$"),
    ("channel", r"^kenh", r"^(kenh|kenh ban|kenh ban hang|kenhbanhang|ban qua|nguon don|loai ban)$"),
    ("cost", r"^gia von|\*\*gia von\*\*", r"^(gia von|gia von luc ban|giavonlucban)$"),
    ("rx", r"^thuoc ke don|^ke don|^ban theo don", r"^(ke don|thuoc ke don|ban theo don|bantheodon|ke_don|rx)$"),
    ("ingredient", r"^hoat chat|^thanh phan", r"^(hoat chat|thanh phan|thanh phan chinh|thanhphanchinh)$"),
    ("staff", r"^ma nhan vien|^nhan vien", r"^(nguoi ban|ma nv|ma nv ban|manhanvien|ma nhan vien|nhan vien|nv ban|nv)$"),
    ("ret_qty", r"^so luong khach tra", r"^(so luong tra|sl tra|sltraquydoi|so luong tra \(dvt co ban\)|sl tra \(dvt co ban\))$"),
]
_ROLES_C = [(r, re.compile(a), re.compile(b)) for r, a, b in _ROLES]


def column_roles(df, meanings):
    """{vai trò: tên cột} cho một bảng. Ý nghĩa README thắng tên cột."""
    out = {}
    cols = [c for c in df.columns if c != "_nguon"]
    for by_meaning in (True, False):
        for c in cols:
            if c in out.values():
                continue
            key = fold(meanings.get(c, "")) if by_meaning else fold(c).replace("_", " ")
            if by_meaning and not key:
                continue
            for role, pm, pn in _ROLES_C:
                if role in out:
                    continue
                if (pm if by_meaning else pn).search(key):
                    out[role] = c
                    break
    # kiểm tra kiểu dữ liệu
    for r in ("amount", "goods", "qty", "qty_base", "price", "cost", "ret_qty", "factor"):
        if r in out and not pd.api.types.is_numeric_dtype(df[out[r]]):
            out.pop(r)
    if "date" in out and not pd.api.types.is_datetime64_any_dtype(df[out["date"]]):
        out.pop("date")
    # cột có cả giờ (thời điểm tạo đơn) -> dùng cho câu hỏi theo ca / giờ
    def has_time(c):
        v = df[c].dropna()
        if not pd.api.types.is_datetime64_any_dtype(v) or v.empty:
            return False
        v = v.iloc[:: max(1, len(v) // 2000)]
        return (v != v.dt.normalize()).mean() > 0.5
    for c in cols:
        key_m, key_n = fold(meanings.get(c, "")), fold(c).replace("_", " ")
        if re.search(r"^thoi diem dat|^gio dat", key_m) or re.fullmatch(
                r"gio|gio tao don|thoi gian lap|thoigianlap|thoi gian tao|thoi gian|gio ban|gio dat", key_n):
            if has_time(c):
                out["time"] = c
                break
    if "time" not in out and "date" in out and has_time(out["date"]):
        out["time"] = out["date"]
    return out


def _is_import(tname, roles, meanings):
    t = _tn(tname)
    if re.search(r"nhap|ton kho|tonkho|lo hang|lohang|the kho|gia$|lich su gia|quy doi|don vi|chi phi|so quy|thanh toan|phieu thu", t):
        return True
    m = " ".join(fold(v) for v in meanings.values())
    return "ma phieu nhap" in m


def _name_key(v):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", str(v)).lower()).strip()


def _item_key(v):
    """Khoá tên HÀNG: bỏ hoa thường, dấu câu và dấu tiếng Việt ('Tra chanh (M)' = 'Trà chanh (M)')."""
    return re.sub(r"[^\w]+", " ", fold(v)).strip()


def _jk(series, by_name):
    """Khoá nối bảng: mã thì bỏ khoảng trắng hai đầu; TÊN thì bỏ hoa thường / dấu câu ('ENAT 400 HỘP' = 'Enat 400 Hộp')."""
    if by_name:
        return series.map(lambda v: _item_key(v) if pd.notna(v) else None).astype("string")
    return series.astype("string").str.strip()


def _canon_names(names, codes, keyf=None):
    keyf = keyf or _name_key
    n = pd.Series(names, dtype="object")
    if n.dropna().empty:
        return names
    c = pd.Series(codes, dtype="object")
    use_code = c.notna().mean() > 0.5
    key = c.where(c.notna(), None) if use_code else n.map(lambda v: keyf(v) if pd.notna(v) else None)
    if use_code:      # dòng có tên nhưng thiếu mã: dùng khoá theo tên
        key = key.where(key.notna(), n.map(lambda v: "~" + keyf(v) if pd.notna(v) else None))
    df = pd.DataFrame({"k": key, "n": n}).dropna()
    if df.empty:
        return names
    best = df.groupby(["k", "n"]).size().reset_index(name="c").sort_values(["k", "c", "n"], ascending=[True, False, True])
    best = best.drop_duplicates("k").set_index("k")["n"]
    return key.map(best).where(n.notna(), np.nan).values


def _strip_contact(v):
    """'Dương Hùng Trung - 0328 548 455' / 'Đặng Hải Nhung (0913885954)' -> chỉ lấy tên."""
    if pd.isna(v):
        return v
    t = re.sub(r"\s*[(\[]?\s*(\+?84|0)[\d .-]{8,}\s*[)\]]?\s*$", "", str(v))
    return re.sub(r"\s*[-–,:|]\s*$", "", t).strip() or np.nan


def _guess_name_col(cdf, cr):
    """Danh mục khách không có cột tên rõ ràng: tìm cột chữ mà giá trị trông như họ tên người (có thể kèm số ĐT)."""
    best, score = None, 0.0
    for c in cdf.columns:
        if c in cr.values() or c == "_nguon" or pd.api.types.is_numeric_dtype(cdf[c]) or pd.api.types.is_datetime64_any_dtype(cdf[c]):
            continue
        v = cdf[c].dropna().astype(str)
        if len(v) < 5 or v.nunique() < 0.7 * len(v):
            continue
        nm = v.map(_strip_contact).map(lambda x: str(x) if pd.notna(x) else "")
        looks = nm.map(lambda x: bool(re.fullmatch(r"[^\W\d_]+( [^\W\d_]+){1,4}", x)) and x[:1].isupper()).mean()
        if looks > score:
            best, score = c, looks
    return (best, _strip_contact) if score >= 0.6 else (None, None)


_TIER = re.compile(r"than thiet|\bvip\b|\bsi\b|\ble\b|thuong|\bbac\b|\bvang\b|kim cuong|dong|moi|quen|dai ly|cong ty|doanh nghiep")


def _pick_cust_group(cat, cr):
    """Cột nhóm khách trong danh mục khách: ưu tiên cột 2–20 giá trị có chữ kiểu 'thân thiết / sỉ / lẻ / VIP'."""
    skip = {cr.get("cust_code"), cr.get("cust_name"), cr.get("item_name"), "_nguon"}
    best, best_score = None, 0.0
    for c in cat.columns:
        if c in skip or pd.api.types.is_numeric_dtype(cat[c]) or pd.api.types.is_datetime64_any_dtype(cat[c]):
            continue
        v = cat[c].dropna().astype(str)
        if not (2 <= v.nunique() <= 20):
            continue
        name_ok = bool(re.search(r"nhom|loai|hang|doi tuong|phan loai|cap", fold(c)))
        score = v.map(lambda x: bool(_TIER.search(fold(x)))).mean() + (0.5 if name_ok else 0) \
            + (0.5 if c == cr.get("cust_group") else 0)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


# ----------------------------------------------------------------------------- dựng bảng
def build_sales(biz):
    try:
        return _build(biz)
    except Exception as e:                    # không bao giờ làm hỏng agent
        biz.warnings.append(f"ban_hang_chuan: lỗi {e!r}")
        return None


def _tn(t):
    """Tên bảng đã bỏ dấu, '_' thành khoảng trắng: 'tra_hang' -> 'tra hang'."""
    return re.sub(r"[_\-.]+", " ", fold(t)).strip()


def _norm_id(s):
    return s.astype("string").str.strip()


def _build(biz):
    info = {}
    for t, df in biz.tables.items():
        m = biz.col_meaning.get(t, {})
        info[t] = (df, column_roles(df, m), m)

    # 1) bảng dòng hàng: có tiền dòng (hoặc SL × đơn giá) và có mặt hàng; lấy bảng nhiều dòng nhất
    lines = None
    for t, (df, r, m) in info.items():
        if _is_import(t, r, m):
            continue
        if ("amount" in r or ("qty" in r and "price" in r)) and ("item_name" in r or "item_code" in r):
            if lines is None or len(df) > len(info[lines][0]):
                lines = t
    # 2) bảng đơn (header): có mã đơn + (ngày hoặc trạng thái), mỗi mã một dòng, trùng mã với bảng dòng
    header = None
    if lines is not None and "id" in info[lines][1]:
        lid = set(_norm_id(info[lines][0][info[lines][1]["id"]]).dropna().unique())
        for t, (df, r, m) in info.items():
            if t == lines or _is_import(t, r, m) or "id" not in r or not ({"date", "status"} & set(r)):
                continue
            ids = _norm_id(df[r["id"]]).dropna()
            if ids.nunique() < 0.9 * len(ids) or not len(ids):
                continue
            overlap = len(lid & set(ids.unique())) / max(1, min(len(lid), ids.nunique()))
            if overlap >= 0.5 and (header is None or len(df) > len(info[header][0])):
                header = t
    if lines is None:            # shop chỉ có danh sách đơn (vd biz_110): mỗi đơn 1 dòng, tiền = tiền hàng của đơn
        for t, (df, r, m) in info.items():
            if _is_import(t, r, m):
                continue
            if "date" in r and ("goods" in r or "amount" in r):
                if header is None or len(df) > len(info[header][0]):
                    header = t
        if header is None:
            return None

    base_t = lines if lines is not None else header
    df, r, _ = info[base_t]
    out = pd.DataFrame(index=df.index)

    def take(role, src_df, src_r):
        return src_df[src_r[role]] if role in src_r else None

    out["ma_don"] = _norm_id(df[r["id"]]) if "id" in r else pd.Series(pd.NA, index=df.index, dtype="string")
    for col, role in [("ngay", "date"), ("thoi_diem", "time"), ("trang_thai_goc", "status"), ("ma_hang", "item_code"), ("ten_hang", "item_name"),
                      ("nhom_hang", "group"), ("don_vi", "unit"), ("so_luong", "qty"), ("so_luong_co_ban", "qty_base"),
                      ("don_gia", "price"), ("gia_von", "cost"), ("ma_khach", "cust_code"), ("ten_khach", "cust_name"),
                      ("nhom_khach", "cust_group"), ("kenh", "channel"), ("nhan_vien", "staff")]:
        v = take(role, df, r)
        out[col] = v.values if v is not None else np.nan
    if lines is not None:
        if "amount" in r:
            out["thanh_tien"] = df[r["amount"]].values
        else:
            out["thanh_tien"] = (df[r["qty"]] * df[r["price"]]).values
    else:
        out["thanh_tien"] = df[r["goods"] if "goods" in r else r["amount"]].values

    used = {base_t}
    out["ke_don"] = np.nan
    out["hoat_chat"] = np.nan
    # 3) ghép thông tin cấp đơn từ header
    if header is not None and lines is not None:
        hdf, hr, _ = info[header]
        h = pd.DataFrame({"ma_don": _norm_id(hdf[hr["id"]])})
        for col, role in [("ngay", "date"), ("thoi_diem", "time"), ("trang_thai_goc", "status"), ("ma_khach", "cust_code"),
                          ("ten_khach", "cust_name"), ("nhom_khach", "cust_group"), ("kenh", "channel"), ("nhan_vien", "staff")]:
            if role in hr:
                h[col] = hdf[hr[role]].values
        h = h.dropna(subset=["ma_don"]).drop_duplicates("ma_don")
        merged = out[["ma_don"]].merge(h, on="ma_don", how="left")
        merged.index = out.index
        for col in h.columns:
            if col == "ma_don":
                continue
            if out[col].isna().all():
                out[col] = merged[col].values
            else:                                 # dòng nào trống thì lấy từ header
                out[col] = out[col].where(out[col].notna(), merged[col])
        used.add(header)

    def _join_catalog():
    # 4) danh mục sản phẩm: bổ sung tên/nhóm/giá vốn theo mã hàng (hoặc nhóm theo tên)
        for t, (pdf, pr, pm) in info.items():
            if t in used or len(pdf) > 5000 or "item_code" not in pr and "item_name" not in pr:
                continue
            if not ({"group", "cost", "item_name", "rx", "ingredient"} & set(pr)):
                continue
            key = "item_code" if ("item_code" in pr and out["ma_hang"].notna().any()) else "item_name"
            okey = "ma_hang" if key == "item_code" else "ten_hang"
            if key not in pr or out[okey].isna().all():
                continue
            byn = key == "item_name"
            cat = pdf.assign(_k=_jk(pdf[pr[key]], byn).values).drop_duplicates("_k").set_index("_k")
            k = _jk(out[okey], byn)
            if k.isin(cat.index).mean() < 0.5:
                continue
            for col, role in [("ten_hang", "item_name"), ("nhom_hang", "group"), ("gia_von", "cost"),
                              ("ke_don", "rx"), ("hoat_chat", "ingredient")]:
                if role in pr and (col not in out.columns or out[col].isna().all()):
                    out[col] = k.map(cat[pr[role]]).values

    _join_catalog()
    # 4b) danh mục khách: tên / nhóm khách theo mã khách
    for t, (cdf, cr, cm) in info.items():
        if t in used or len(cdf) > 20000 or "cust_code" not in cr or out["ma_khach"].isna().all():
            continue
        name_col = cr.get("cust_name") or (cr.get("item_name") if "khach" in _tn(t) else None)
        if not name_col and "khach" in _tn(t):
            name_col = next((c for c in cdf.columns if fold(c) in ("kh", "ten", "ho ten", "ho va ten", "tengoi", "khach hang")), None)
        name_clean = None
        if not name_col and "khach" in _tn(t):
            name_col, name_clean = _guess_name_col(cdf, cr)
        if not name_col and _pick_cust_group(cdf, cr) is None:
            continue
        cat = cdf.drop_duplicates(cr["cust_code"])
        cat.index = cat[cr["cust_code"]].astype("string").str.strip()
        k = out["ma_khach"].astype("string").str.strip()
        if k.dropna().isin(cat.index).mean() < 0.5:
            continue
        if name_col and out["ten_khach"].isna().all():
            names = cat[name_col] if name_clean is None else cat[name_col].map(name_clean)
            out["ten_khach"] = k.map(names).values
        gcol = _pick_cust_group(cat, cr)
        if gcol is not None and out["nhom_khach"].isna().all():
            out["nhom_khach"] = k.map(cat[gcol]).values
    # đơn không có mã khách (khách lẻ) thì không có nhóm khách: ghi rõ để khi chia tỉ trọng không bị rơi mất (groupby bỏ NaN)
    if out["nhom_khach"].notna().any() and out["ma_khach"].notna().any():
        out.loc[out["ma_khach"].isna() & out["nhom_khach"].isna(), "nhom_khach"] = "(khách lẻ, không có mã)"
    # 4c) tên nhân viên theo mã nhân viên
    out["ten_nhan_vien"] = np.nan
    for t, (sdf, sr, sm) in info.items():
        if "staff" not in sr or "nhan vien" not in _tn(t) or out["nhan_vien"].isna().all():
            continue
        name_col = next((c for c in sdf.columns if c != sr["staff"] and re.search(r"^(ten|ho ten|tengoi|ho va ten)", fold(c))), None)
        if name_col:
            cat = sdf.drop_duplicates(sr["staff"]).set_index(sr["staff"])[name_col]
            out["ten_nhan_vien"] = out["nhan_vien"].map(cat).values
    # 4d) mã hàng theo tên (khi bảng bán chỉ có tên) để xếp hạng "bằng nhau thì mã tăng dần"
    if out["ma_hang"].isna().all() and out["ten_hang"].notna().any():
        for t, (pdf, pr, pm) in info.items():
            if t in used or len(pdf) > 5000 or not {"item_code", "item_name"} <= set(pr) or "nhap" in _tn(t):
                continue
            cat = pdf.assign(_k=_jk(pdf[pr["item_name"]], True).values).drop_duplicates("_k").set_index("_k")
            k = _jk(out["ten_hang"], True)
            if k.dropna().isin(cat.index).mean() >= 0.8:
                out["ma_hang"] = k.map(cat[pr["item_code"]]).values
                if out["nhom_hang"].isna().all() and "group" in pr:
                    out["nhom_hang"] = k.map(cat[pr["group"]]).values
                if out["gia_von"].isna().all() and "cost" in pr:
                    out["gia_von"] = k.map(cat[pr["cost"]]).values
                break
    if out["ma_hang"].notna().any():   # 4d vừa có mã hàng: nối lại danh mục theo mã (hoạt chất, kê đơn, nhóm...)
        _join_catalog()
    # 4e) số lượng quy về đơn vị cơ bản theo bảng quy đổi (đơn vị lớn × hệ số)
    if out["so_luong_co_ban"].isna().all() and out["so_luong"].notna().any():
        for t, (qdf, qr, qm) in info.items():
            if "factor" not in qr or "unit" not in qr:
                continue
            key = "item_code" if ("item_code" in qr and out["ma_hang"].notna().any()) else "item_name"
            okey = "ma_hang" if key == "item_code" else "ten_hang"
            if key not in qr:
                continue
            f = {(fold(a), fold(u)): float(x) for a, u, x in zip(qdf[qr[key]], qdf[qr["unit"]], qdf[qr["factor"]])
                 if pd.notna(a) and pd.notna(u) and pd.notna(x)}
            if not f:
                continue
            unit = out["don_vi"]
            fac = [f.get((fold(a), fold(u)), 1.0) if pd.notna(a) and pd.notna(u) else 1.0
                   for a, u in zip(out[okey], unit)]
            out["so_luong_co_ban"] = pd.to_numeric(out["so_luong"], errors="coerce") * np.array(fac)
            break
    # 5) trạng thái chuẩn + đơn có trả hàng (ghi ở bảng khác: số lượng trả > 0, hoặc bảng phiếu trả hàng)
    out["ngay"] = pd.to_datetime(out["ngay"], errors="coerce").dt.normalize()
    out["trang_thai"] = status_of(out["trang_thai_goc"])
    ret_ids = set()
    for t, (tdf, tr, tm) in info.items():
        if "id" not in tr:
            continue
        if "ret_qty" in tr:
            ids = _norm_id(tdf.loc[tdf[tr["ret_qty"]].fillna(0) > 0, tr["id"]])
            ret_ids |= set(ids.dropna())
        elif re.search(r"tra hang|phieu tra", _tn(t)):
            ret_ids |= set(_norm_id(tdf[tr["id"]]).dropna())
    # dòng trả hàng ghi số âm (cùng mã đơn với đơn đã bán): đơn đó có trả hàng
    neg = pd.Series(False, index=out.index)
    for c in ("so_luong", "so_luong_co_ban", "thanh_tien"):
        neg |= pd.to_numeric(out[c], errors="coerce").fillna(0) < 0
    out["dong_tra_hang"] = neg
    has_id = out["ma_don"].notna()
    ret_ids |= set(out.loc[neg & has_id, "ma_don"])
    # ngày của đơn = ngày bán (dòng trả hàng có thể ghi ngày trả, khác ngày bán)
    if has_id.any() and neg.any():
        for col in ("ngay", "thoi_diem"):
            d0 = pd.to_datetime(out.loc[has_id & ~neg, col], errors="coerce").groupby(out.loc[has_id & ~neg, "ma_don"]).min()
            od = out["ma_don"].map(d0)
            out[col] = od.where(od.notna() & has_id, out[col])
    # trả hàng theo trạng thái ở bất kỳ dòng nào của đơn -> cả đơn
    st_ids = set(out.loc[has_id & (out["trang_thai"] == "trả hàng"), "ma_don"])
    huy_ids = set(out.loc[has_id & (out["trang_thai"] == "huỷ"), "ma_don"])
    out["tra_hang"] = out["ma_don"].isin(ret_ids | st_ids).fillna(False).astype(bool) | (out["trang_thai"] == "trả hàng")
    huy = out["ma_don"].isin(huy_ids).fillna(False).astype(bool) | (out["trang_thai"] == "huỷ")
    out.loc[huy & (out["trang_thai"] != "huỷ"), "trang_thai"] = "huỷ"
    out["tinh_dt"] = ~huy & ~out["tra_hang"]
    out = out.drop(columns=["trang_thai_goc"])
    out["thoi_diem"] = pd.to_datetime(out["thoi_diem"], errors="coerce")
    # Tên hàng / tên khách ghi nhiều kiểu ("RED BULL 250ML", "Red Bull  250ml"): gom về MỘT cách viết
    # (cách viết gặp nhiều nhất) theo mã nếu có mã, không có mã thì theo tên đã bỏ hoa thường/khoảng trắng/dấu câu.
    out["ten_hang"] = _canon_names(out["ten_hang"], out["ma_hang"], _item_key)
    out["ten_khach"] = _canon_names(out["ten_khach"], out["ma_khach"])
    # khach = "một khách" theo §3: shop có mã khách -> mã khách (đơn không mã = khách lẻ -> trống);
    # shop không có mã -> tên khách (đã gom cách viết); tên kiểu "Khách lẻ" / "vãng lai" -> trống (không phải khách)
    le = out["ten_khach"].map(lambda v: bool(re.fullmatch(r"(khach )?(le|vang lai)|khach|kl|khach hang le|-|\.|0", fold(v)))
                              if pd.notna(v) else False).astype(bool)
    if out["ma_khach"].notna().mean() > 0.05:
        out["khach"] = out["ma_khach"].where(~le)
    else:
        out["khach"] = out["ten_khach"].where(~le)
    if out["ke_don"].notna().any():       # thuốc kê đơn: True/False (dữ liệu ghi True, 1, "Có", "x"...)
        out["ke_don"] = out["ke_don"].map(lambda v: (bool(v) if isinstance(v, (bool, np.bool_, int, float, np.integer))
                                                    else fold(v) in ("co", "x", "true", "yes", "ke don", "rx", "1"))
                                          if pd.notna(v) else False).astype(bool)
    # dòng phí (phí phục vụ, phí ship...) không phải mặt hàng: không xếp hạng, không đếm (vẫn tính doanh thu)
    out["la_phi"] = out["ten_hang"].astype("string").map(
        lambda v: bool(re.search(r"^(phi |phu thu|tien ship|ship\b)|phi phuc vu|phi van chuyen|phi giao hang", fold(v)))
        if pd.notna(v) else False).astype(bool)
    cols = ["ma_don", "ngay", "thoi_diem", "trang_thai", "tra_hang", "tinh_dt", "dong_tra_hang", "ma_hang", "ten_hang", "la_phi", "nhom_hang", "ke_don", "hoat_chat", "don_vi", "so_luong",
            "so_luong_co_ban", "don_gia", "thanh_tien", "gia_von", "ma_khach", "ten_khach", "khach", "nhom_khach", "kenh", "nhan_vien", "ten_nhan_vien"]
    if out["kenh"].notna().any():         # mã kênh tiếng Anh -> cách gọi trong câu hỏi ("tại quầy", "qua điện thoại")
        _k = {"store": "tại quầy", "offline": "tại quầy", "pos": "tại quầy", "counter": "tại quầy", "phone": "điện thoại",
              "hotline": "điện thoại", "zalo oa": "Zalo OA", "zalo": "Zalo", "facebook": "Facebook", "fb": "Facebook",
              "website": "website", "web": "website", "online": "online"}
        out["kenh"] = out["kenh"].map(lambda v: _k.get(re.sub(r"[_\-]+", " ", fold(v)).strip(), v) if pd.notna(v) else v)
    try:
        out["don_vi_co_ban"] = _base_unit(out, info)
        cols.insert(cols.index("so_luong") , "don_vi_co_ban")
    except Exception as e:
        biz.warnings.append(f"don_vi_co_ban: lỗi {e!r}")
    out = out[cols].reset_index(drop=True)
    out.attrs["nguon"] = {"dong": base_t, "don": header if lines is not None else None,
                          "moi_dong_la": "một dòng hàng" if lines is not None else "một đơn"}
    return out


def _base_unit(out, info):
    """Đơn vị cơ bản của từng mặt hàng (rulebook "Bán được bao nhiêu <đơn vị>"): đơn vị của các dòng bán có hệ số 1
    (so_luong_co_ban == so_luong); hàng chỉ bán theo đơn vị lớn thì lấy cột 'đơn vị cơ bản' của bảng quy đổi / danh mục."""
    item = out["ma_hang"].where(out["ma_hang"].notna(), out["ten_hang"]).astype("string")
    q, qb = pd.to_numeric(out["so_luong"], errors="coerce"), pd.to_numeric(out["so_luong_co_ban"], errors="coerce")
    one = (q > 0) & ((qb - q).abs() < 1e-9) & out["don_vi"].notna()
    df = pd.DataFrame({"i": item[one], "u": out.loc[one, "don_vi"].astype("string").str.strip()})
    base = df.groupby(["i", "u"]).size().reset_index(name="n").sort_values(["i", "n"], ascending=[True, False]) \
        .drop_duplicates("i").set_index("i")["u"].to_dict()
    # hàng chưa từng bán theo hệ số 1: tìm cột "đơn vị cơ bản" ở các bảng khác (quy đổi / danh mục)
    miss = set(item.dropna().unique()) - set(base)
    if miss:
        by_name = {}
        for t, (df2, r2, m2) in info.items():
            bcol = next((c for c in df2.columns if "co ban" in fold(c).replace("_", " ") or "co ban" in fold(m2.get(c, ""))
                         if not pd.api.types.is_numeric_dtype(df2[c])), None) if isinstance(m2, dict) else None
            if bcol is None:
                continue
            for role in ("item_code", "item_name"):
                if role in r2:
                    for a, u in zip(df2[r2[role]], df2[bcol]):
                        if pd.notna(a) and pd.notna(u):
                            by_name.setdefault(_item_key(a), str(u).strip())
        names = dict(zip(item, out["ten_hang"]))
        for i in miss:
            u = by_name.get(_item_key(i)) or (by_name.get(_item_key(names.get(i))) if names.get(i) is not None else None)
            if u:
                base[i] = u
    return item.map(base).values


def describe(ban):
    """Đoạn mô tả ngắn đưa vào prompt."""
    src = ban.attrs.get("nguon", {})
    empty = [c for c in ban.columns if ban[c].isna().all()]
    return (f"Bảng `ban_hang_chuan` ({len(ban):,} dòng, mỗi dòng là {src.get('moi_dong_la')}; dựng từ bảng "
            f"`{src.get('dong')}`" + (f" + `{src.get('don')}`" if src.get("don") else "") + "). "
            + (f"Cột không có dữ liệu: {', '.join(empty)}." if empty else ""))


# ============================================================================= BẢNG NHẬP HÀNG CHUẨN
# nhap_hang_chuan: mỗi dòng = một dòng hàng trong phiếu nhập.
#   ma_phieu, ngay (ngày phiếu), trang_thai_goc, huy (phiếu huỷ), tinh (True = phiếu được tính: không huỷ),
#   ncc (nhà cung cấp: tên nếu có, không thì mã), ma_hang, ten_hang, don_vi (đơn vị nhập), so_luong, don_gia (giá nhập
#   1 đơn vị nhập), thanh_tien (tiền của dòng), tong_phieu (tổng tiền cả phiếu, lặp lại ở mọi dòng của phiếu)
_IMP_ROLES = [
    ("pid", r"^ma phieu nhap", r"^(ma nhap hang|ma phieu nhap|maphieunhap|so phieu|so phieu nhap|ma don nhap|ma pn)$"),
    ("date", r"^ngay nhap", r"^(ngay nhap|ngaynhapkho|ngay nhap kho|thoi gian|ngay|ngay phieu)$"),
    ("status", r"^trang thai phieu|^trang thai", r"^(trang thai|tinh trang|tinhtrang|status)$"),
    ("total", r"^tong tien (cua ca )?phieu|^tong tien phieu", r"^(can tra ncc|tong gia tri nhap|tonggiatrinhap|tong tien phieu|tong tien)$"),
    ("amount", r"^tien nhap cua ca dong", r"^(thanh tien|tien nhap)$"),
    ("price", r"^gia nhap", r"^(gia nhap|don gia|dongianhap|don gia nhap|gia von nhap)$"),
    ("qty", r"^so luong nhap|^so luong$", r"^(sl nhap|so luong|so luong nhap|sl|soluong)$"),
    ("unit", r"^don vi nhap|^don vi tinh", r"^(dvt|dvt nhap|don vi tinh|don vi nhap|donvitinh|don vi)$"),
    ("item_code", r"^ma hang", r"^(ma hang|ma sku|sku|mahanghoa|ma hh|ma sp|ma mon|ma san pham)$"),
    ("item_name", r"^ten hang", r"^(ten hang|ten hang hoa|ten mat hang|product|san pham|hang hoa|ten sp)$"),
    ("supplier", r"^nha cung cap|^ma nha cung cap|^ma ncc", r"^(nha cung cap|ma ncc|ma nha cung cap|manhacungcap|ncc)$"),
]
_IMP_C = [(r, re.compile(a), re.compile(b)) for r, a, b in _IMP_ROLES]


def _imp_roles(df, meanings):
    out = {}
    cols = [c for c in df.columns if c != "_nguon"]
    for by_m in (True, False):
        for c in cols:
            if c in out.values():
                continue
            key = fold(meanings.get(c, "")) if by_m else fold(c).replace("_", " ")
            if by_m and not key:
                continue
            for role, pm, pn in _IMP_C:
                if role not in out and (pm if by_m else pn).search(key):
                    out[role] = c
                    break
    for r in ("total", "amount", "price", "qty"):
        if r in out and not pd.api.types.is_numeric_dtype(df[out[r]]):
            out.pop(r)
    if "date" in out and not pd.api.types.is_datetime64_any_dtype(df[out["date"]]):
        out.pop("date")
    return out


def build_imports(biz):
    try:
        return _build_imports(biz)
    except Exception as e:
        biz.warnings.append(f"nhap_hang_chuan: lỗi {e!r}")
        return None


def _build_imports(biz):
    cand = {}
    for t, df in biz.tables.items():
        if not re.search(r"nhap", _tn(t)):
            continue
        r = _imp_roles(df, biz.col_meaning.get(t, {}))
        if "pid" in r:
            cand[t] = (df, r)
    if not cand:
        return None
    # bảng dòng: có số lượng + (giá hoặc tiền dòng) + mặt hàng; bảng phiếu: có ngày, mỗi mã một dòng
    lines = [t for t, (df, r) in cand.items() if "qty" in r and ("item_code" in r or "item_name" in r)]
    if not lines:
        return None
    lt = max(lines, key=lambda t: len(cand[t][0]))
    df, r = cand[lt]
    out = pd.DataFrame({"ma_phieu": _norm_id(df[r["pid"]]).values})
    for col, role in [("ngay", "date"), ("trang_thai_goc", "status"), ("ncc", "supplier"), ("ma_hang", "item_code"),
                      ("ten_hang", "item_name"), ("don_vi", "unit"), ("so_luong", "qty"), ("don_gia", "price"),
                      ("thanh_tien", "amount"), ("tong_phieu", "total")]:
        out[col] = df[r[role]].values if role in r else np.nan
    for t, (hdf, hr) in cand.items():
        if t == lt:
            continue
        ids = _norm_id(hdf[hr["pid"]])
        if ids.nunique() < 0.9 * len(ids):
            continue
        h = hdf.assign(_k=ids.values).drop_duplicates("_k").set_index("_k")
        for col, role in [("ngay", "date"), ("trang_thai_goc", "status"), ("ncc", "supplier"), ("tong_phieu", "total")]:
            if role in hr and out[col].isna().all():
                out[col] = out["ma_phieu"].map(h[hr[role]]).values
    if out["thanh_tien"].isna().all() and out["don_gia"].notna().any():
        out["thanh_tien"] = pd.to_numeric(out["so_luong"], errors="coerce") * pd.to_numeric(out["don_gia"], errors="coerce")
    if out["don_gia"].isna().all() and out["thanh_tien"].notna().any():
        out["don_gia"] = pd.to_numeric(out["thanh_tien"], errors="coerce") / pd.to_numeric(out["so_luong"], errors="coerce")
    # tên hàng theo mã (danh mục), tên nhà cung cấp theo mã (bảng nhà cung cấp)
    ban = getattr(biz, "ban", None)
    if out["ten_hang"].isna().all() and out["ma_hang"].notna().any():
        names = {}
        if ban is not None and ban["ma_hang"].notna().any():
            names = ban.dropna(subset=["ma_hang", "ten_hang"]).drop_duplicates("ma_hang").set_index("ma_hang")["ten_hang"].to_dict()
        for t, d in biz.tables.items():
            rr = column_roles(d, biz.col_meaning.get(t, {}))
            if {"item_code", "item_name"} <= set(rr) and len(d) <= 5000 and "nhap" not in _tn(t):
                for k, v in zip(d[rr["item_code"]], d[rr["item_name"]]):
                    if pd.notna(k) and pd.notna(v):
                        names.setdefault(str(k).strip(), v)
        out["ten_hang"] = out["ma_hang"].astype("string").str.strip().map(names).values
    if out["ncc"].notna().any():
        for t, d in biz.tables.items():
            if not re.search(r"nha cung cap|ncc|supplier", _tn(t)) or t in cand:
                continue
            code = next((c for c in d.columns if re.search(r"^(ma )?(ncc|nha cung cap)|manhacungcap", fold(c))), None)
            name = next((c for c in d.columns if c != code and re.search(r"^(ten|tengoi|ten ncc|ten nha cung cap)", fold(c))), None)
            if code and name:
                m = dict(zip(d[code].astype(str).str.strip(), d[name]))
                k = out["ncc"].astype("string").str.strip()
                if k.dropna().isin(list(m)).mean() > 0.5:
                    out["ncc_ma"] = out["ncc"]
                    out["ncc"] = k.map(m).values
    out["ngay"] = pd.to_datetime(out["ngay"], errors="coerce").dt.normalize()
    out["huy"] = out["trang_thai_goc"].map(lambda v: status_one(v) == "huỷ" if pd.notna(v) else False).astype(bool)
    out["tinh"] = ~out["huy"]
    out["ten_hang"] = _canon_names(out["ten_hang"], out["ma_hang"], _item_key)
    out = out.drop(columns=["trang_thai_goc"])
    out.attrs["nguon"] = {"dong": lt}
    return out


# ============================================================================= CÔNG NỢ CHUẨN
# cong_no_don: mỗi dòng = một đơn.  ma_don, ngay (ngày đặt), ma_khach, ten_khach, kenh, pttt (hình thức thanh toán),
#   ghi_no (đơn bán nợ / mua chịu), cod_san (COD hoặc bán qua sàn: không phải nợ), huy, tong_don (khách phải trả cả đơn)
# thu_tien: mỗi dòng = một lần khách trả tiền cho một đơn.  ma_don, ngay_tra, so_tien, ngay_don, sau_ngay_mua
_PAY_M = re.compile(r"^hinh thuc thanh toan|^phuong thuc thanh toan")
_PAY_N = re.compile(r"^(phuong thuc thanh toan|phuong thuc tt|pttt|phuongthuctt|hinh thuc thanh toan|hinh thuc tt|thanh toan bang)$")
_TOT_M = re.compile(r"^so khach phai tra|^tong tien don|^khach phai tra|^tong thanh toan")
_TOT_N = re.compile(r"^(tong sau giam gia|tong bill|tong tien|tongthanhtoan|tong thanh toan|phai thu|tong don|tongtien|khach phai tra)$")
_AMT_M = re.compile(r"^so tien (tra|thu)|^tien (tra|thu)|so tien tra lan do")
_AMT_N = re.compile(r"^(gia tri|tien|so tien|so tien thu|sotienthu|so tien tra|tien thu)$")
_DEBT = re.compile(r"ghi no|\bdebt\b|cong no|mua chiu|\bno\b|tra sau|credit")
_CODSAN = re.compile(r"\bcod\b|thu ho|shopee|lazada|tiki|tiktok|\bsan\b|tmdt|thuong mai dien tu|marketplace")


def build_debt(biz):
    try:
        return _build_debt(biz)
    except Exception as e:
        biz.warnings.append(f"cong_no: lỗi {e!r}")
        return None, None


def _find_col(df, meanings, pm, pn, numeric=False):
    for by_m in (True, False):
        for c in df.columns:
            if c == "_nguon":
                continue
            key = fold(meanings.get(c, "")) if by_m else fold(c).replace("_", " ")
            if key and (pm if by_m else pn).search(key):
                if numeric and not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                return c
    return None


def _build_debt(biz):
    ban = getattr(biz, "ban", None)
    hname = ban.attrs.get("nguon", {}).get("don") if ban is not None else None
    if hname is None and ban is not None and ban.attrs.get("nguon", {}).get("moi_dong_la") == "một đơn":
        hname = ban.attrs["nguon"]["dong"]
    if hname is None:
        return None, None
    H = biz.tables[hname]
    hm = biz.col_meaning.get(hname, {})
    hr = column_roles(H, hm)
    pay = _find_col(H, hm, _PAY_M, _PAY_N)
    tot = _find_col(H, hm, _TOT_M, _TOT_N, numeric=True)
    if "id" not in hr or tot is None:
        return None, None
    don = pd.DataFrame({"ma_don": _norm_id(H[hr["id"]]).values})
    don["tong_don"] = pd.to_numeric(H[tot], errors="coerce").values
    don["pttt"] = H[pay].values if pay else np.nan
    # ngày, khách, kênh, trạng thái lấy từ bảng bán hàng chuẩn (đã chuẩn hoá) theo mã đơn
    if ban is not None:
        o = ban.dropna(subset=["ma_don"]).groupby("ma_don").agg(
            ngay=("ngay", "min"), ma_khach=("ma_khach", "first"), ten_khach=("ten_khach", "first"),
            kenh=("kenh", "first"), trang_thai=("trang_thai", "first"), tra_hang=("tra_hang", "max"))
        don = don.join(o, on="ma_don")
    don = don.dropna(subset=["ma_don"]).drop_duplicates("ma_don")
    fp = don["pttt"].map(lambda v: fold(v) if pd.notna(v) else "")
    fk = don["kenh"].map(lambda v: fold(v) if pd.notna(v) else "") if "kenh" in don else ""
    don["huy"] = don.get("trang_thai", pd.Series("", index=don.index)).eq("huỷ")
    don["cod_san"] = fp.str.contains(_CODSAN) | (fk.str.contains(_CODSAN) if isinstance(fk, pd.Series) else False)
    don["ghi_no"] = fp.str.contains(_DEBT) & ~don["cod_san"]
    # bảng các lần trả tiền
    thu = None
    for t, P in biz.tables.items():
        if t == hname or ban is not None and t == ban.attrs.get("nguon", {}).get("dong"):
            continue
        pm = biz.col_meaning.get(t, {})
        pr = column_roles(P, pm)
        amt = _find_col(P, pm, _AMT_M, _AMT_N, numeric=True)
        dcol = next((c for c in P.columns if pd.api.types.is_datetime64_any_dtype(P[c])), None)
        if amt is None or dcol is None or len(P) > 3 * len(don) + 1000:
            continue
        # cột mã đơn = cột có giá trị trùng mã đơn nhiều nhất (bảng thu còn có "mã phiếu thu" riêng)
        dset, best, ids = set(don["ma_don"]), 0.0, None
        for c in P.columns:
            if c == "_nguon" or pd.api.types.is_numeric_dtype(P[c]) or pd.api.types.is_datetime64_any_dtype(P[c]):
                continue
            v = _norm_id(P[c])
            frac = v.dropna().isin(dset).mean() if v.notna().any() else 0
            if frac > best:
                best, ids = frac, v
        if ids is None or best < 0.5:
            continue
        thu = pd.DataFrame({"ma_don": ids.values, "ngay_tra": pd.to_datetime(P[dcol], errors="coerce").dt.normalize().values,
                            "so_tien": pd.to_numeric(P[amt], errors="coerce").values})
        thu.attrs["nguon"] = t
        break
    if thu is not None:
        thu = thu.dropna(subset=["ma_don"])
        thu["ngay_don"] = thu["ma_don"].map(don.set_index("ma_don")["ngay"])
        thu["sau_ngay_mua"] = thu["ngay_tra"] > thu["ngay_don"]
        thu["ghi_no"] = thu["ma_don"].map(don.set_index("ma_don")["ghi_no"]).fillna(False).astype(bool)
        thu["huy"] = thu["ma_don"].map(don.set_index("ma_don")["huy"]).fillna(False).astype(bool)
        thu = thu.reset_index(drop=True)
    don = don.reset_index(drop=True)
    don.attrs["nguon"] = hname
    return don, thu


# ---------------------------------------------------------------- GIAO HÀNG NHIỀU ĐỢT (rulebook "Giao nhiều đợt")
def build_delivery(biz):
    try:
        return _build_delivery(biz)
    except Exception as e:
        biz.warnings.append(f"giao_hang: lỗi {e!r}")
        return None


def _build_delivery(biz):
    """Bảng sổ giao hàng: mỗi dòng là một mặt hàng giao trong một ĐỢT của một đơn.
    Nhận diện theo nghĩa cột (chung cho mọi shop): có cột đợt/lần/lượt giao, cột ngày giao, cột mã đơn khớp mã đơn bán."""
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban):
        return None
    don_ids = set(ban["ma_don"].dropna().astype(str).str.strip())
    used = set((ban.attrs.get("nguon") or {}).values())
    best = None
    for t, df in biz.tables.items():
        if t in used or len(df) == 0:
            continue
        mean = biz.col_meaning.get(t, {})
        key = lambda c: (fold(c).replace("_", " ") + " | " + fold(mean.get(c, "")))
        ctx = _tn(t) + " " + " ".join(key(c) for c in df.columns)
        if not re.search(r"\bgiao\b|\bcho\b|bien so|\bxe\b|van chuyen", ctx):
            continue
        dot = next((c for c in df.columns if re.search(r"\b(luot|lan|dot|chuyen)\b", key(c))
                    and pd.api.types.is_numeric_dtype(df[c])), None)
        ngay = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
                     and re.search(r"giao|cho|ngay", key(c))), None)
        if dot is None or ngay is None:
            continue
        idc, ov = None, 0.0
        for c in df.columns:
            if c in (dot, ngay, "_nguon") or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            v = df[c].dropna().astype(str).str.strip()
            if not len(v):
                continue
            r = v.isin(don_ids).mean()
            if r > ov:
                idc, ov = c, r
        if idc is None or ov < 0.5:
            continue
        if best is None or len(df) > len(best[1]):
            best = (t, df, idc, dot, ngay, key)
    if best is None:
        return None
    t, df, idc, dot, ngay, key = best
    out = pd.DataFrame({"ma_don": _norm_id(df[idc]).values, "dot": pd.to_numeric(df[dot], errors="coerce").values,
                        "ngay_giao": df[ngay].values})
    for name, pat, num in (("ten_hang", r"ten hang|hang hoa|san pham|mat hang", False),
                           ("so_luong", r"so luong|\bsl\b", True), ("don_vi", r"don vi|\bdv\b|dvt", False),
                           ("xe", r"bien so|\bxe\b", False), ("nguoi_giao", r"nguoi (giao|cho)|tai xe|nhan vien", False)):
        c = next((c for c in df.columns if c not in (idc, dot, ngay) and re.search(pat, key(c))
                  and (pd.api.types.is_numeric_dtype(df[c]) == num)), None)
        out[name] = df[c].values if c is not None else np.nan
    out.attrs["nguon"] = t
    return out
