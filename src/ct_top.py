"""Công thức cố định cho nhóm câu TOP / XẾP HẠNG và KHÁCH (không dùng LLM).
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.
Mỗi công thức chỉ bật khi rulebook của bộ đang chạy còn ghi đúng định nghĩa tương ứng (_co_luat)."""
import re
import pandas as pd
from loader import fold

_LUAT = {
    "chay": "theo so luong quy ve don vi co ban",
    "phi": "phi phuc vu khong phai mat hang",
    "khach": "moi ma khach la mot khach",
    "khach_tien": "khach mua nhieu tien nhat xep theo doanh thu",
    "khach_mua": "khach da mua trong ky la khach co it nhat mot don",
    "dt": "tong thanh tien cac dong hang",
    "hoa": "bang nhau o tieu chi cau hoi thi xep theo ma tang dan",
    "loi": "theo tong lai gop cua mat hang",
    "gv": "gia von cua moi dong ban bang so luong nhan voi gia von ghi tren dong ban",
    "nv": "nhan vien ban gioi nhat theo doanh thu",
}


def _co_luat(rulebook, *keys):
    if rulebook is None:
        return True
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook)))
    return all(_LUAT[k] in t for k in keys)


def _range(p, ban):
    st = pd.Timestamp(p.start) if p.start is not None else ban["ngay"].min()
    return st.normalize(), pd.Timestamp(p.end).normalize()


def _sel(ban, p):
    st, en = _range(p, ban)
    d = pd.to_datetime(ban["ngay"], errors="coerce").dt.normalize()
    return ban[ban["tinh_dt"] & (d >= st) & (d <= en)]


def _lo_gia_tri_la(f, ban):
    """Câu nhắc tên mặt hàng / nhóm hàng / kênh / nhóm khách cụ thể -> không xử lý."""
    for c in ("ten_hang", "nhom_hang", "kenh", "nhom_khach", "ten_khach"):
        if c not in ban:
            continue
        for n in ban[c].dropna().unique():
            k = re.sub(r"\s*\(.*?\)\s*$", "", fold(str(n))).strip()
            if len(k) >= 3 and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", f):
                return True
    return False


_LA = r"nhom|kenh|\bca\b|\bca (sang|chieu)|tru |ngoai tru|so voi|tang|giam|thu [2-7]|chu nhat|cuoi tuan|ngay thuong|\bgio\b|" \
      r"zalo|dien thoai|quay|\bno\b|cong no|chiu|nhap|\bton\b|han dung|\bhuy\b|tra hang|\bdon vi\b|gioi tinh|\bnu\b|\bnam\b (mua|chi)|" \
      r"duong |dia chi|\blan nhat\b.*\bmua\b|tre em|nguoi lon|\bmoi\b"


_SO_CHU = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
_DT = r"(?:khach|mat hang|san pham|sp|mon|hang|nguoi)"


def _N(f):
    m = re.search(r"\btop (\d{1,2})\b", f) or re.search(r"(?<![/\d])\b(\d{1,2}) " + _DT + r"\b", f)
    if m:
        return int(m[1])
    m = re.search(r"\btop (hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", f) \
        or re.search(r"\b(hai|ba|bon|sau|bay|tam|chin|muoi) " + _DT + r"\b", f) \
        or re.search(r"(?:liet ke|ke ten|ke ra|cho biet|xem) (nam) " + _DT + r"\b", f)
    return _SO_CHU[m[1]] if m else None


def _nhieu_ma_khong_ro_so(f):
    """Câu hỏi NHIỀU đối tượng (top / những / các ...) mà không đọc được N -> không trả lời một tên."""
    return _N(f) is None and re.search(r"\btop\b|\b(nhung|cac) " + _DT + r"\b|\bnhung (ai|gi)\b", f)


def _ten(g):
    return g


def solve(q, f, periods, biz, rulebook):
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban) or "tinh_dt" not in ban or "ngay" not in ban:
        return None
    if not _co_luat(rulebook, "hoa"):
        return None
    f = re.sub(r"^e ", "", re.sub(r"\s+", " ", f).strip())

    # ---------- THÁNG NÀO doanh thu cao nhất (nhiều kỳ tháng) ----------
    if len(periods) >= 2 and re.search(r"thang nao", f) and re.search(r"(doanh thu|dthu|revenue).{0,25}cao nhat|cao nhat.{0,25}(doanh thu|dthu)", f) \
            and _co_luat(rulebook, "dt") and all(p.kind == "month" and p.start is not None for p in periods) \
            and not re.search(_LA, f):
        rows = []
        for p in periods:
            b = _sel(ban, p)
            rows.append((float(b["thanh_tien"].sum()), pd.Timestamp(p.start)))
        rows.sort(key=lambda t: (-t[0], t[1]))
        return {"type": "date", "answer": rows[0][1].strftime("%Y-%m")}, "Tháng có doanh thu lớn nhất trong các tháng nêu (hoà: tháng sớm hơn)"
    if len(periods) == 1 and periods[0].kind == "range" and periods[0].start is not None and re.search(r"thang nao", f) \
            and re.search(r"tu thang \d+/\d{4} den (het )?thang \d+/\d{4}", f) and pd.Timestamp(periods[0].start).day == 1 \
            and (pd.Timestamp(periods[0].end) + pd.Timedelta(days=1)).day == 1 \
            and re.search(r"(doanh thu|dthu|revenue).{0,25}cao nhat|cao nhat.{0,25}(doanh thu|dthu)", f) \
            and _co_luat(rulebook, "dt") and not re.search(_LA, f):
        ms = pd.period_range(pd.Timestamp(periods[0].start), pd.Timestamp(periods[0].end), freq="M")
        rows = []
        for m in ms:
            st, en = m.start_time.normalize(), m.end_time.normalize()
            d = pd.to_datetime(ban["ngay"], errors="coerce").dt.normalize()
            rows.append((float(ban[ban["tinh_dt"] & (d >= st) & (d <= en)]["thanh_tien"].sum()), st))
        rows.sort(key=lambda t: (-t[0], t[1]))
        return {"type": "date", "answer": rows[0][1].strftime("%Y-%m")}, "Tháng có doanh thu lớn nhất trong dải tháng"
    if len(periods) != 1 or periods[0].kind == "snapshot" or periods[0].start is None:
        return None
    p = periods[0]
    b = _sel(ban, p)
    if not len(b):
        return None

    # ---------- NGÀY nào doanh thu cao nhất ----------
    if re.search(r"ngay nao", f) and re.search(r"(doanh thu|dthu|revenue).{0,25}(cao|nhieu|lon) nhat|ban dat nhat|(cao|nhieu|lon) nhat.{0,25}(doanh thu|dthu)", f) \
            and _co_luat(rulebook, "dt") and not re.search(_LA, f) and not _lo_gia_tri_la(f, ban):
        s = b.groupby(pd.to_datetime(b["ngay"]).dt.normalize())["thanh_tien"].sum()
        s = s.reset_index().sort_values(["thanh_tien", "ngay"], ascending=[False, True])
        return {"type": "date", "answer": s.iloc[0]["ngay"].strftime("%Y-%m-%d")}, "Ngày có tổng thành tiền (đơn tính doanh thu) lớn nhất trong kỳ; hoà: ngày sớm hơn"

    if re.search(_LA, f) or _lo_gia_tri_la(f, ban) or _nhieu_ma_khong_ro_so(f):
        return None
    n = _N(f)

    # ---------- MẶT HÀNG bán chạy ----------
    if _co_luat(rulebook, "chay", "phi") and re.search(r"\b(mat hang|hang|san pham|sp|mon)\b", f) \
            and re.search(r"ban chay|\bhot nhat|chay nhat|ban (duoc |dc )?nhieu nhat|ban nhieu nhat", f) \
            and not re.search(r"khach|doanh thu|tien|it nhat|cham|\be\b|lai|loi|nhan vien|nguoi ban|trung binh|\d+ don|bill", f):
        x = b[~b["la_phi"]] if "la_phi" in b else b
        if "so_luong_co_ban" in x and x["so_luong_co_ban"].notna().any():
            qcol = "so_luong_co_ban"
        else:
            return None
        key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if "ma_hang" in x else x["ten_hang"]
        x = x.assign(_k=key.astype(str))
        g = x.groupby("_k").agg(sl=(qcol, "sum"), ten=("ten_hang", "first")).reset_index()
        g = g.sort_values(["sl", "_k"], ascending=[False, True])
        if n and n >= 2:
            if len(g) < n:
                return None
            return {"type": "list", "answer": [str(t) for t in g["ten"].head(n)]}, f"Top {n} mặt hàng theo số lượng cơ bản (hoà: mã tăng dần)"
        if n == 1 or n is None:
            return {"type": "string" if n is None else "list", "answer": str(g.iloc[0]["ten"]) if n is None else [str(g.iloc[0]["ten"])]}, \
                "Mặt hàng bán chạy nhất theo số lượng cơ bản (hoà: mã nhỏ hơn)"

    # ---------- MẶT HÀNG lãi gộp cao nhất ----------
    if _co_luat(rulebook, "loi", "gv", "phi") and re.search(r"\b(mat hang|hang|san pham|sp|mon)\b", f) \
            and re.search(r"lai gop (ca thang )?(cao|lon|nhieu) nhat|(cao|lon|nhieu) nhat.{0,20}lai gop|loi nhat|lai nhat", f) \
            and "gia_von" in b and b["gia_von"].notna().any() and n in (None, 1):
        x = b[~b["la_phi"]] if "la_phi" in b else b
        if x["gia_von"].isna().any():
            return None
        key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if "ma_hang" in x else x["ten_hang"]
        x = x.assign(_k=key.astype(str), _lai=x["thanh_tien"] - x["so_luong"] * x["gia_von"])
        g = x.groupby("_k").agg(lai=("_lai", "sum"), ten=("ten_hang", "first")).reset_index()
        g = g.sort_values(["lai", "_k"], ascending=[False, True])
        return {"type": "string", "answer": str(g.iloc[0]["ten"])}, "Mặt hàng có tổng lãi gộp lớn nhất"

    # ---------- MẶT HÀNG được mua ở >= N đơn ----------
    m = re.search(r"(?:it nhat|tu|>=|≥|o|trong) (\d+) (?:don|bill|hoa don)(?: hang)?(?: khac nhau)?(?: tro len)?", f)
    if m and re.search(r"(bao nhieu|may|dem|so) (mat hang|san pham|sp|mon)", f) and re.search(r"it nhat|tro len|>=|≥", f) and _co_luat(rulebook, "phi") \
            and not re.search(r"khach|%|phan tram|ti le", f):
        x = b[~b["la_phi"]] if "la_phi" in b else b
        key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if "ma_hang" in x else x["ten_hang"]
        c = x.assign(_k=key.astype(str)).groupby("_k")["ma_don"].nunique()
        return {"type": "number", "answer": int((c >= int(m[1])).sum())}, "Số mặt hàng xuất hiện ở ít nhất N đơn khác nhau"

    # ---------- NHÂN VIÊN bán giỏi nhất ----------
    if _co_luat(rulebook, "nv", "dt") and re.search(r"(nhan vien|nguoi ban)", f) and not re.search(r"khach", f) \
            and re.search(r"doanh thu.{0,15}cao nhat|ban gioi nhat|(cao|nhieu) nhat.{0,15}doanh thu|doanh thu (lon|nhieu) nhat", f) and n in (None, 1) \
            and ("nhan_vien" in b or "ten_nhan_vien" in b):
        col = "nhan_vien" if "nhan_vien" in b and b["nhan_vien"].notna().any() else "ten_nhan_vien"
        s = b[b[col].notna()].groupby(col)["thanh_tien"].sum().reset_index().sort_values(["thanh_tien", col], ascending=[False, True])
        if len(s):
            return {"type": "string", "answer": str(s.iloc[0][col])}, "Nhân viên có doanh thu lớn nhất trong kỳ"

    # ---------- KHÁCH ----------
    if not re.search(r"\bkhach\b", f) or "khach" not in b or not b["khach"].notna().any() or not _co_luat(rulebook, "khach"):
        return None
    k = b[b["khach"].notna()].copy()
    k["_k"] = k["khach"].astype(str)
    g = k.groupby("_k").agg(tien=("thanh_tien", "sum"), so_don=("ma_don", "nunique"), ten=("ten_khach", "first")).reset_index()
    ten = lambda r: str(r["ten"]) if pd.notna(r["ten"]) else str(r["_k"])

    tien_re = r"nhieu tien nhat|chi (tieu )?(nhieu|cao) nhat|chi tieu nhieu nhat|tien mua cao nhat|mua cao nhat"
    don_re = r"nhieu (don|lan)( hang)? nhat|so don (mua )?nhieu nhat"
    top_k = (re.search(r"\b(khach|nguoi)\b", f) and (n is not None or re.search(r"\b(ai|khach nao|khach hang nao)\b", f)))
    if re.search(tien_re, f) != re.search(don_re, f) and top_k and _co_luat(rulebook, "khach_tien") \
            and not re.search(r"\d+ don|%|phan tram|ti le|bao nhieu khach|may khach", f):
        col = "tien" if re.search(tien_re, f) else "so_don"
        g = g.sort_values([col, "_k"], ascending=[False, True])
        if n and n >= 2:
            if len(g) < n:
                return None
            return {"type": "list", "answer": [ten(r) for _, r in g.head(n).iterrows()]}, f"Top {n} khách theo {col}"
        if n in (None, 1):
            return {"type": "string", "answer": ten(g.iloc[0])}, f"Khách đứng đầu theo {col}"
        return None

    # số khách mua / mua >= k đơn / tỉ lệ
    if not _co_luat(rulebook, "khach_mua"):
        return None
    pct = re.search(r"%|phan tram|ti le", f)
    ge = re.search(r"(?:tu|it nhat|>=|≥) (\d+) (?:don|lan|hoa don)|(\d+) (?:don|lan) tro len", f)
    gt = re.search(r"(?:hon|tren|qua|vuot) (\d+) (?:don|lan)", f)
    if ge or gt:
        if ge and gt:
            return None
        kk = int((ge or gt)[1] or (ge[2] if ge else 0))
        if ge and ge[1] is None:
            kk = int(ge[2])
        sel = (g["so_don"] >= kk) if ge else (g["so_don"] > kk)
        if pct:
            if not re.search(r"(so )?khach (da |co )?mua|khach hang trong|trong so khach|khach trong", f):
                return None
            return {"type": "percent", "answer": round(float(sel.sum()) / len(g) * 100, 2)}, "Số khách mua >= k đơn / số khách đã mua * 100"
        if re.search(r"(bao nhieu|may|dem|so) khach", f) or re.search(r"khach.{0,20}(bao nhieu|may)\b", f):
            return {"type": "number", "answer": int(sel.sum())}, "Số khách có số đơn thoả điều kiện"
        return None
    if re.search(r"(bao nhieu|nhieu|may|dem|tong so|so) khach( hang)? (da |co )?(mua|den mua)", f) and not pct \
            and not re.search(r"\d", re.sub(r"\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|thang \d+|nam \d+|tuan \d+", "", f)):
        return {"type": "number", "answer": int(len(g))}, "Số khách (mã khách) có đơn tính doanh thu trong kỳ"
    return None
