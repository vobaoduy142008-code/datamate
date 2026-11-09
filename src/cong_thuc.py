"""Giải THẲNG bằng code (không cần LLM) một số loại câu có công thức cố định trong rulebook, đã kiểm trên đáp án train.
Chỉ nhận câu khi chắc chắn đúng loại; không chắc thì trả None để agent giải bằng LLM như bình thường.
Không dựa vào id câu hỏi hay business_id: chỉ dựa vào nội dung câu và bảng chuẩn của shop."""
import re

import pandas as pd

from loader import fold

_DV_BO = {"don", "khach", "mat", "loai", "ngay", "lan", "phan", "tien", "dong", "nguoi", "ca", "gio", "thang",
          "tuan", "nam", "nhom", "so", "%", "kg"}


def _range(p, ban):
    st = pd.Timestamp(p.start) if p.start is not None else ban["ngay"].min()
    return st.normalize(), pd.Timestamp(p.end).normalize()


def _in(s, p, ban):
    st, en = _range(p, ban)
    d = pd.to_datetime(s, errors="coerce").dt.normalize()
    return (d >= st) & (d <= en)


def _names_in_q(f, ban):
    """Câu có nhắc tên một mặt hàng cụ thể không (bỏ phần size '(M)' cuối tên)."""
    for n in ban["ten_hang"].dropna().unique():
        k = re.sub(r"\s*\(.*?\)\s*$", "", fold(n)).strip()
        if len(k) >= 4 and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", f):
            return True
    return False


# Mỗi công thức chỉ dùng khi rulebook của bộ ĐANG CHẠY vẫn ghi đúng định nghĩa đó (so chữ đã bỏ dấu, bỏ dấu câu).
# Rulebook private khác đi -> không dùng công thức, để model đọc rulebook mới và tự giải.
_LUAT = {
    "dot": "moi dot cua moi don tinh mot dot",
    "don_vi": "co don vi co ban la",
    "size": "ket thuc bang m hoac l",
    "e": "da tung ban truoc ky",
    "chay": "theo so luong quy ve don vi co ban",
    "khong_du": "du lieu phat hanh khong du",
    "gia": "cho mot don vi co ban truoc chiet khau dong",
    "tang_gia": "so so luong don vi co ban n ngay ke tu d",
}


def _co_luat(rulebook, key):
    if rulebook is None:
        return True
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    return _LUAT[key] in re.sub(r"\s+", " ", t)


def try_solve(question, periods, biz, rulebook=None):
    """-> (answer dict, mô tả công thức) hoặc None."""
    f = fold(question)
    ban = getattr(biz, "ban", None)
    if len(periods) != 1 or periods[0].kind == "snapshot":
        p = None
    else:
        p = periods[0]

    # 1) số ĐỢT giao hàng (rulebook: mỗi đợt của mỗi đơn tính một đợt) — train 16/16
    g = getattr(biz, "giao", None)
    if p is not None and g is not None and _co_luat(rulebook, "dot") and re.search(r"\bdot\b", f) and re.search(r"giao|\bcho\b|\bxe\b", f) \
            and re.search(r"bao nhieu|\bmay\b|dem|tong", f) and not re.search(r"\bdon\b.*\bnao\b|mat hang|trung binh", f):
        x = g[_in(g["ngay_giao"], p, ban if ban is not None else g.rename(columns={"ngay_giao": "ngay"}))]
        return {"type": "number", "answer": int(len(x.drop_duplicates(["ma_don", "dot"])))}, \
            "Số đợt giao = số cặp (mã đơn, đợt) khác nhau có ngày giao trong kỳ"

    # 0) shop KHÔNG ghi khách (mọi đơn là khách lẻ) mà câu hỏi về khách -> không trả lời được (train 8/8)
    if ban is not None and len(ban) and _co_luat(rulebook, "khong_du") \
            and ("khach" not in ban or ban["khach"].notna().sum() == 0) \
            and re.search(r"\bkhach\b", f) and not re.search(r"\bno\b|tra no|thu no|cong no|huy|tra hang|tra lai|\ble\b|vang lai", f):
        return {"type": "unanswerable", "reason": "Dữ liệu của shop không ghi thông tin khách hàng"}, \
            "Shop không có thông tin khách (mọi đơn là khách lẻ)"

    # 6) "doanh thu tháng sau có LUÔN cao hơn / thấp hơn tháng trước không" (các tháng liên tiếp trong kỳ) — train 21/21
    up = re.search(r"(luon|deu|lien tuc)\s*(cao hon|tang|lon hon|nhieu hon)|(cao hon|tang)\s*(lien tuc|deu)|thang sau .*cao hon thang truoc", f)
    down = re.search(r"(luon|deu|lien tuc)\s*(thap hon|giam|it hon)", f)
    if ban is not None and len(ban) and (up or down) and re.search(r"doanh thu|dthu", f) and re.search(r"khong\s*\??\s*$|phai khong|co phai|dung khong|hay khong", f):
        ms = sorted({m for q in periods for m in pd.period_range(pd.Timestamp(q.start), pd.Timestamp(q.end), freq="M")}) \
            if all(q.start is not None for q in periods) else []
        if len(ms) >= 2:
            x = ban[ban["tinh_dt"]]
            per = x["ngay"].dt.to_period("M")
            rev = [float(x.loc[per == m, "thanh_tien"].sum()) for m in ms]
            ok = all(rev[i + 1] > rev[i] for i in range(len(rev) - 1)) if up else \
                all(rev[i + 1] < rev[i] for i in range(len(rev) - 1))
            return {"type": "boolean", "answer": bool(ok)}, \
                "So doanh thu từng tháng liên tiếp: " + ", ".join(f"{m}: {r:,.0f}" for m, r in zip(ms, rev))

    if ban is None or not len(ban) or p is None:
        return None
    b = ban[ban["tinh_dt"] & _in(ban["ngay"], p, ban)]

    # 2) bán bao nhiêu <đơn vị> / size M-L, cộng MỌI mặt hàng (không nêu tên hàng) — train 33/33 + 10 câu "tính theo"
    m = (re.search(r"tinh theo (\w+)", f) if re.search(r"(cong|tat ca|moi|het) (cac )?(mat hang|mon)", f) else None) \
        or re.search(r"(?:tong cong|tat ca|cong lai)\s+(?:ban (?:ra|duoc) )?(?:duoc )?(?:may|bao nhieu) (\w+)", f) \
        or re.search(r"ban (?:ra|duoc|dc) (?:tong cong|total|tat ca) (?:may|bao nhieu) (\w+)", f)\
        or re.search(r"ban (?:ra|duoc|dc) (?:may|bao nhieu) (\w+) size [sml]\b", f) \
        or re.search(r"(?:so|may|bao nhieu) (\w+) size [sml]\b(?=.*(?:moi loai|tat ca|cong|tong|ca quan|ca cua hang))", f)
    if m and m[1] not in _DV_BO and "don_vi_co_ban" in b and _co_luat(rulebook, "don_vi") and not re.search(r"doanh thu|dthu|\btien\b|%|trung binh", f) \
            and not _names_in_q(f, ban):
        u = m[1]
        x = b[~b["la_phi"]] if "la_phi" in b else b
        size = re.search(r"\bsize ([sml])\b", f)
        if size:
            if not _co_luat(rulebook, "size"):
                return None
            x = x[x["ten_hang"].astype(str).str.strip().str.upper().str.endswith(f"({size[1].upper()})")]
        x = x[x["don_vi_co_ban"].map(lambda v: fold(v) if pd.notna(v) else "") == u]
        if len(x):
            v = float(x["so_luong_co_ban"].sum())
            return {"type": "number", "answer": int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)}, \
                f"Cộng so_luong_co_ban các dòng có đơn vị cơ bản '{u}'" + (f", tên kết thúc ({size[1].upper()})" if size else "")

    # 7) HIỆU ỨNG TĂNG GIÁ: số lượng (đơn vị cơ bản) N ngày kể từ D so với N ngày ngay trước — train 66/66
    cmp = p.meta.get("compare") if isinstance(getattr(p, "meta", None), dict) else None
    if cmp is not None and "ngay ke tu" in fold(p.label) and re.search(r"(tang|len|doi) gia", f) \
            and _co_luat(rulebook, "tang_gia") and re.search(r"%|phan tram|tang hay giam|thay doi", f):
        names = sorted({n for n in ban["ten_hang"].dropna().unique() if len(fold(n)) >= 4 and fold(n) in f},
                       key=len, reverse=True)
        if names:
            x = ban[ban["tinh_dt"] & (ban["ten_hang"] == names[0])]
            qty = x["so_luong_co_ban"].where(x["so_luong_co_ban"].notna(), x["so_luong"])
            d = x["ngay"].dt.normalize()
            after = float(qty[(d >= pd.Timestamp(p.start)) & (d <= pd.Timestamp(p.end))].sum())
            before = float(qty[(d >= pd.Timestamp(cmp.start)) & (d <= pd.Timestamp(cmp.end))].sum())
            if before:
                return {"type": "percent", "answer": round((after - before) / before * 100, 2)}, \
                    f"Số lượng '{names[0]}': {after:g} (N ngày kể từ ngày tăng giá) so với {before:g} (N ngày trước)"

    # 5) GIÁ BÁN mặt hàng X ngày D (đơn giá trên dòng bán ngày D, tính cả đơn huỷ/trả) — train 53/53
    mg = re.search(r"(?:gia|don gia)\b.*?(?:bao nhieu|may)\b.*?(?:mot|1|moi) (\w+)", f) \
        or re.search(r"(?:gia|don gia) (?:ban )?(?:mot|1) (\w+)", f)
    if mg and _co_luat(rulebook, "gia") and re.search(r"ngay \d", f) and p.start is not None and p.start == p.end \
            and not re.search(r"tang|giam|trung binh|so voi|nhap", f):
        names = sorted({(fold(n), n) for n in ban["ten_hang"].dropna().unique()}, key=lambda t: -len(t[0]))
        hit = [n for k, n in names if len(k) >= 4 and k in f]
        if hit:
            d = pd.Timestamp(p.end).normalize()
            x = ban[(ban["ten_hang"] == hit[0]) & (ban["ngay"].dt.normalize() == d)]
            xu = x[x["don_vi"].map(lambda v: fold(v) if pd.notna(v) else "") == mg[1]]
            if not len(xu):
                xu = x
            v = xu["don_gia"].mode()
            if len(v) and pd.notna(v.iloc[0]):
                return {"type": "number", "answer": int(round(float(v.iloc[0])))}, \
                    f"Đơn giá trên dòng bán '{hit[0]}' ngày {d:%Y-%m-%d} (giá hay gặp nhất)"

    # 4) MỘT mặt hàng bán chạy nhất (theo số lượng quy về đơn vị cơ bản, hoà thì mã nhỏ hơn) — train 24/24
    if _co_luat(rulebook, "chay") and re.search(
            r"(hang|san pham|sp|mat hang|mon)( nao)? (ban )?(chay nhat|ban (duoc|dc) nhieu nhat)|ban chay nhat la", f) \
            and not re.search(r"top|\d+ (mat hang|san pham|mon)|nhung|cac|nhom|khach|tien|doanh thu|it nhat|cham|"
                              r"\bca\b|kenh|zalo|dien thoai|thu [2-7]|chu nhat|cuoi tuan|ngay thuong|gio", f):
        x = b[~b["la_phi"]] if "la_phi" in b else b
        key = "ma_hang" if x["ma_hang"].notna().any() else "ten_hang"
        g = x.groupby(key).agg(sl=("so_luong_co_ban", "sum"), ten=("ten_hang", "first")).reset_index()
        g = g.sort_values(["sl", key], ascending=[False, True])
        if len(g):
            return {"type": "string", "answer": str(g.iloc[0]["ten"])}, \
                "Mặt hàng có tổng so_luong_co_ban lớn nhất trong kỳ (đơn tính doanh thu), hoà thì mã nhỏ hơn"

    # 3) mặt hàng ế / không bán được cái nào trong kỳ — train 21/22
    if _co_luat(rulebook, "e") and re.search(r"(khong|ko|chua) (ai )?(ban|mua) (?:(?:duoc|dc|ra) )*(cai|lan|mon|san pham)? ?nao|"
                                             r"\bban cham\b|hang e\b|\bbi e\b|\be han\b", f) \
            and re.search(r"bao nhieu|\bmay\b|dem|tong", f) and re.search(r"mat hang|\bmon\b|san pham", f):
        x = ban[~ban["la_phi"]] if "la_phi" in ban else ban
        key = "ma_hang" if x["ma_hang"].notna().any() else "ten_hang"
        _, en = _range(p, ban)
        tat_ca = set(x[x["ngay"].dt.normalize() <= en][key].dropna())
        da_ban = set(x[x["tinh_dt"] & _in(x["ngay"], p, ban)][key].dropna())
        return {"type": "number", "answer": len(tat_ca - da_ban)}, \
            "Mặt hàng đã từng bán tới hết kỳ nhưng trong kỳ không bán được (đơn tính doanh thu)"
    return None
