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

    if ban is None or not len(ban) or p is None:
        return None
    b = ban[ban["tinh_dt"] & _in(ban["ngay"], p, ban)]

    # 2) bán bao nhiêu <đơn vị> / size M-L, cộng MỌI mặt hàng (không nêu tên hàng) — train 33/33 + 10 câu "tính theo"
    m = (re.search(r"tinh theo (\w+)", f) if re.search(r"(cong|tat ca|moi|het) (cac )?(mat hang|mon)", f) else None) \
        or re.search(r"(?:tong cong|tat ca|cong lai)\s+(?:ban (?:ra|duoc) )?(?:duoc )?(?:may|bao nhieu) (\w+)", f) \
        or re.search(r"ban (?:ra|duoc|dc) (?:tong cong|total|tat ca) (?:may|bao nhieu) (\w+)", f)\
        or re.search(r"ban (?:ra|duoc|dc) (?:may|bao nhieu) (\w+) size [sml]\b", f)
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

    # 3) mặt hàng ế / không bán được cái nào trong kỳ — train 21/22
    if _co_luat(rulebook, "e") and re.search(r"(khong|ko|chua) (ai )?(ban|mua) (duoc|dc|ra)?\s*(cai|lan|mon)? ?nao|\bban cham\b|hang e\b|\be han\b", f) \
            and re.search(r"bao nhieu|\bmay\b|dem|tong", f) and re.search(r"mat hang|\bmon\b|san pham", f):
        x = ban[~ban["la_phi"]] if "la_phi" in ban else ban
        key = "ma_hang" if x["ma_hang"].notna().any() else "ten_hang"
        _, en = _range(p, ban)
        tat_ca = set(x[x["ngay"].dt.normalize() <= en][key].dropna())
        da_ban = set(x[x["tinh_dt"] & _in(x["ngay"], p, ban)][key].dropna())
        return {"type": "number", "answer": len(tat_ca - da_ban)}, \
            "Mặt hàng đã từng bán tới hết kỳ nhưng trong kỳ không bán được (đơn tính doanh thu)"
    return None
