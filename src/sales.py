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

import numpy as np
import pandas as pd

from loader import fold

SALES_VERSION = "s2"          # đổi số này khi sửa file để cache của loader tự dựng lại bảng chuẩn

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
    t = fold(tname)
    if re.search(r"nhap|ton kho|tonkho|lo hang|lohang|the kho|gia$|lich su gia|quy doi|don vi|chi phi|so quy|thanh toan|phieu thu", t):
        return True
    m = " ".join(fold(v) for v in meanings.values())
    return "ma phieu nhap" in m


# ----------------------------------------------------------------------------- dựng bảng
def build_sales(biz):
    try:
        return _build(biz)
    except Exception as e:                    # không bao giờ làm hỏng agent
        biz.warnings.append(f"ban_hang_chuan: lỗi {e!r}")
        return None


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
        cat = pdf.drop_duplicates(pr[key]).set_index(pdf.drop_duplicates(pr[key])[pr[key]].astype("string").str.strip())
        k = out[okey].astype("string").str.strip()
        if k.isin(cat.index).mean() < 0.5:
            continue
        for col, role in [("ten_hang", "item_name"), ("nhom_hang", "group"), ("gia_von", "cost"),
                          ("ke_don", "rx"), ("hoat_chat", "ingredient")]:
            if role in pr and (col not in out.columns or out[col].isna().all()):
                out[col] = k.map(cat[pr[role]]).values
    # 4b) danh mục khách: tên / nhóm khách theo mã khách
    for t, (cdf, cr, cm) in info.items():
        if t in used or len(cdf) > 20000 or "cust_code" not in cr or out["ma_khach"].isna().all():
            continue
        name_col = cr.get("cust_name") or (cr.get("item_name") if "khach" in fold(t) else None)
        if not name_col and "khach" in fold(t):
            name_col = next((c for c in cdf.columns if fold(c) in ("kh", "ten", "ho ten", "ho va ten", "tengoi", "khach hang")), None)
        if not name_col and "cust_group" not in cr:
            continue
        cat = cdf.drop_duplicates(cr["cust_code"])
        cat.index = cat[cr["cust_code"]].astype("string").str.strip()
        k = out["ma_khach"].astype("string").str.strip()
        if k.dropna().isin(cat.index).mean() < 0.5:
            continue
        if name_col and out["ten_khach"].isna().all():
            out["ten_khach"] = k.map(cat[name_col]).values
        if "cust_group" in cr and out["nhom_khach"].isna().all():
            out["nhom_khach"] = k.map(cat[cr["cust_group"]]).values
    # 4c) tên nhân viên theo mã nhân viên
    out["ten_nhan_vien"] = np.nan
    for t, (sdf, sr, sm) in info.items():
        if "staff" not in sr or "nhan vien" not in fold(t) or out["nhan_vien"].isna().all():
            continue
        name_col = next((c for c in sdf.columns if c != sr["staff"] and re.search(r"^(ten|ho ten|tengoi|ho va ten)", fold(c))), None)
        if name_col:
            cat = sdf.drop_duplicates(sr["staff"]).set_index(sr["staff"])[name_col]
            out["ten_nhan_vien"] = out["nhan_vien"].map(cat).values
    # 4d) mã hàng theo tên (khi bảng bán chỉ có tên) để xếp hạng "bằng nhau thì mã tăng dần"
    if out["ma_hang"].isna().all() and out["ten_hang"].notna().any():
        for t, (pdf, pr, pm) in info.items():
            if t in used or len(pdf) > 5000 or not {"item_code", "item_name"} <= set(pr) or "nhap" in fold(t):
                continue
            cat = pdf.drop_duplicates(pr["item_name"])
            cat.index = cat[pr["item_name"]].astype("string").str.strip()
            k = out["ten_hang"].astype("string").str.strip()
            if k.dropna().isin(cat.index).mean() >= 0.8:
                out["ma_hang"] = k.map(cat[pr["item_code"]]).values
                if out["nhom_hang"].isna().all() and "group" in pr:
                    out["nhom_hang"] = k.map(cat[pr["group"]]).values
                if out["gia_von"].isna().all() and "cost" in pr:
                    out["gia_von"] = k.map(cat[pr["cost"]]).values
                break
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
        elif re.search(r"tra hang|phieu tra", fold(t)):
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
    if out["ke_don"].notna().any():       # thuốc kê đơn: True/False (dữ liệu ghi True, 1, "Có", "x"...)
        out["ke_don"] = out["ke_don"].map(lambda v: (bool(v) if isinstance(v, (bool, np.bool_, int, float, np.integer))
                                                    else fold(v) in ("co", "x", "true", "yes", "ke don", "rx", "1"))
                                          if pd.notna(v) else False).astype(bool)
    # dòng phí (phí phục vụ, phí ship...) không phải mặt hàng: không xếp hạng, không đếm (vẫn tính doanh thu)
    out["la_phi"] = out["ten_hang"].astype("string").map(
        lambda v: bool(re.search(r"^(phi |phu thu|tien ship|ship\b)|phi phuc vu|phi van chuyen|phi giao hang", fold(v)))
        if pd.notna(v) else False).astype(bool)
    cols = ["ma_don", "ngay", "thoi_diem", "trang_thai", "tra_hang", "tinh_dt", "dong_tra_hang", "ma_hang", "ten_hang", "la_phi", "nhom_hang", "ke_don", "hoat_chat", "don_vi", "so_luong",
            "so_luong_co_ban", "don_gia", "thanh_tien", "gia_von", "ma_khach", "ten_khach", "nhom_khach", "kenh", "nhan_vien", "ten_nhan_vien"]
    out = out[cols].reset_index(drop=True)
    out.attrs["nguon"] = {"dong": base_t, "don": header if lines is not None else None,
                          "moi_dong_la": "một dòng hàng" if lines is not None else "một đơn"}
    return out


def describe(ban):
    """Đoạn mô tả ngắn đưa vào prompt."""
    src = ban.attrs.get("nguon", {})
    empty = [c for c in ban.columns if ban[c].isna().all()]
    return (f"Bảng `ban_hang_chuan` ({len(ban):,} dòng, mỗi dòng là {src.get('moi_dong_la')}; dựng từ bảng "
            f"`{src.get('dong')}`" + (f" + `{src.get('don')}`" if src.get("don") else "") + "). "
            + (f"Cột không có dữ liệu: {', '.join(empty)}." if empty else ""))
