"""Công thức cố định (không LLM): TĂNG/GIẢM %, TỈ LỆ, TỈ TRỌNG, SO SÁNH HAI KỲ.
solve(q, f, periods, biz, rulebook) -> (answer_dict, mô tả) | None.  Không chắc -> None."""
import re
import pandas as pd
from loader import fold


def _norm_rb(rulebook):
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook or ""))
    return re.sub(r"\s+", " ", t)


# điều kiện bật: rulebook đang chạy phải có đúng định nghĩa
_LUAT = {
    "huy": "phan tram don huy trong moi don dat trong ky",
    "tt_nhom_khach": "ti trong cua nhom chia cho doanh thu ca shop cung ky",
    "pct": "phan tram thay doi tang truong so voi ky truoc",
    "ca": "ca sang 06 00 13 59 ca chieu 14 00 21 59",
    "ke_don": "ti le ke don tinh theo doanh thu",
    "khach": "khach da mua trong ky la khach co it nhat mot don",
    "unans": "hoan toan ngoai khoang ngay co du lieu",
}


def _ok(rb_n, key):
    return _LUAT[key] in rb_n


def _day(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _bounds(p, ban):
    st = pd.Timestamp(p.start) if p.start is not None else ban["ngay"].min()
    return st.normalize(), pd.Timestamp(p.end).normalize()


def _outside(p, ban):
    st, en = _bounds(p, ban)
    return en < ban["ngay"].min().normalize() or st > ban["ngay"].max().normalize()


def _sel(ban, p):
    st, en = _bounds(p, ban)
    d = _day(ban["ngay"])
    return ban[(d >= st) & (d <= en)]


def _rev(x):
    return float(x.loc[x["tinh_dt"], "thanh_tien"].sum())


def _norders(x):
    return int(x.loc[x["tinh_dt"], "ma_don"].nunique())


def _pct(num, den):
    if not den:
        return None
    return round(num / den * 100, 2)


def _unans(why):
    return {"type": "unanswerable", "reason": why}


def _find_name(f, values, prefix=None):
    """Tên (đã bỏ dấu) dài nhất xuất hiện nguyên từ trong câu. None nếu không có/nhập nhằng."""
    best = []
    for v in values:
        k = fold(str(v)).strip()
        if len(k) < 2:
            continue
        pat = (prefix or "") + re.escape(k) + r"(?!\w)"
        if re.search(r"(?<!\w)" + pat, f):
            best.append((len(k), v))
    if not best:
        return None
    best.sort(key=lambda t: -t[0])
    if len(best) > 1 and best[0][0] == best[1][0]:
        return None
    return best[0][1]


_REV = r"doanh thu|dthu|tien ban|tien hang|ban duoc bao nhieu tien|thu ve"
_BAD_COMMON = r"tang gia|doi gia|len gia|gia ban|trung binh|binh quan|moi don|\btop\b|xep hang|cao nhat|thap nhat|nhieu nhat|it nhat|\bno\b|cong no|ton kho|nhap|loi nhuan|\blai\b|\bvon\b|gia von"


def solve(q, f, periods, biz, rulebook):
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban) or not periods:
        return None
    rb = _norm_rb(rulebook)
    f1 = re.sub(r"\s+", " ", f.replace("revenue", "doanh thu"))
    bad = re.search(_BAD_COMMON, f1)
    for fn in (_khach_n_don, _ti_le_huy, _ca, _ti_trong, _doi_ky):
        if bad and fn is not _khach_n_don:
            continue
        try:
            r = fn(f1, periods, biz, ban, rb)
        except Exception:
            r = None
        if r:
            return r
    return None


# ---------------------------------------------------------------- tỉ lệ huỷ
def _ti_le_huy(f, periods, biz, ban, rb):
    if len(periods) != 1 or not _ok(rb, "huy"):
        return None
    if not re.search(r"(ti le (don )?huy|ti le huy don|huy don|don (bi )?huy)", f):
        return None
    if not re.search(r"%|phan tram|ti le", f) or re.search(r"so voi|cung ky|khach|tra hang|nhom|kenh", f):
        return None
    p = periods[0]
    if p.start is None or p.meta.get("compare") is not None:
        return None
    if _outside(p, ban):
        return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
    x = _sel(ban, p)
    n = int(x["ma_don"].nunique())
    if not n:
        return None
    h = int(x.loc[x["trang_thai"] == "huỷ", "ma_don"].nunique())
    return {"type": "percent", "answer": _pct(h, n)}, f"Đơn huỷ {h} / mọi đơn đặt trong kỳ {n}"


# ---------------------------------------------------------------- khách mua >= N đơn
def _khach_n_don(f, periods, biz, ban, rb):
    m = re.search(r"(?:tu|>=)\s*(\d+)\s*don tro len", f) or re.search(r"it nhat\s*(\d+)\s*don", f)
    if not m or len(periods) < 1 or not _ok(rb, "khach"):
        return None
    if not re.search(r"khach", f) or not re.search(r"%|phan tram", f) or re.search(r"so voi|cung ky|doanh thu|\bno\b", f):
        return None
    if not re.search(r"chiem|bao nhieu %|bn %|nhieu %|bnhiu %|phan tram|ti le", f):
        return None
    if not re.search(r"khach.{0,30}\bmua\b|so khach", f):
        return None
    p = periods[0]
    if p.start is None:
        return None
    if len(periods) > 1 and any(q.start is None or (q.start, q.end) != (p.start, p.end) for q in periods[1:]):
        return None
    if _outside(p, ban):
        return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
    if ban["khach"].notna().sum() == 0:
        return _unans("Dữ liệu của shop không ghi thông tin khách hàng"), "shop không có khách"
    x = _sel(ban, p)
    x = x[x["tinh_dt"] & x["khach"].notna()].drop_duplicates(["ma_don"])
    c = x.groupby("khach")["ma_don"].nunique()
    if not len(c):
        return None
    k = int(m.group(1))
    return {"type": "percent", "answer": _pct(int((c >= k).sum()), len(c))}, f"Khách >= {k} đơn / khách đã mua ({len(c)})"


# ---------------------------------------------------------------- ca sáng / ca chiều
def _ca(f, periods, biz, ban, rb):
    if not re.search(r"ca chieu", f) or not re.search(r"ca sang", f) or not _ok(rb, "ca"):
        return None
    if len(periods) != 1:
        return None
    if re.search(r"(so voi|so vs|\bvs\b|chenh) ca sang", f):
        pass
    elif f.find("ca chieu") > f.find("ca sang"):
        return None
    if not re.search(r"%|phan tram", f):
        return None
    p = periods[0]
    if p.start is None:
        return None
    if re.search(r"khach|nhom|mat hang|kenh", f):
        return None
    isrev = re.search(_REV, f)
    iso = re.search(r"so don|don hang|bao nhieu don|\bdon\b", f)
    if bool(isrev) == bool(iso):
        return None
    if _outside(p, ban):
        return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
    x = _sel(ban, p)
    if x["thoi_diem"].notna().sum() == 0 or x["thoi_diem"].isna().any():
        return None
    h = x["thoi_diem"].dt.hour
    sang = x[(h >= 6) & (h < 14)]
    chieu = x[(h >= 14) & (h < 22)]
    a, b = (_rev(chieu), _rev(sang)) if isrev else (_norders(chieu), _norders(sang))
    v = _pct(a - b, b)
    if v is None:
        return None
    return {"type": "percent", "answer": v}, f"(chiều {a:g} - sáng {b:g}) / sáng"


# ---------------------------------------------------------------- tỉ trọng
_PAYC = [("tm", r"tien mat|\btm\b|cash"), ("ck", r"chuyen khoan|\bck\b|transfer|bank"),
         ("vi", r"vi dien tu|e ?wallet|momo"), ("cod", r"\bcod\b|thu ho"), ("no", r"ghi no|debt|cong no")]


def _payc(v):
    t = fold(str(v))
    for k, pat in _PAYC:
        if re.search(pat, t):
            return k
    return None


def _no_gender(biz):
    for t, df in getattr(biz, "tables", {}).items():
        for c in df.columns:
            if re.search(r"gioi|gender|\bsex\b", fold(str(c))) and df[c].notna().any() and df[c].astype(str).str.strip().ne("").any():
                return False
    return True


def _term_in_raw(biz, term):
    """Cụm `term` (đã bỏ dấu) có xuất hiện trong ô nào của dữ liệu gốc (cột chữ ít giá trị khác nhau) không."""
    for t, df in getattr(biz, "tables", {}).items():
        for c in df.columns:
            col = df[c]
            if col.dtype != object and str(col.dtype) not in ("string", "category"):
                continue
            try:
                u = col.dropna().unique()
            except Exception:
                continue
            if len(u) > 80:
                continue
            for v in u:
                if term in fold(str(v)):
                    return True
    return False


def _ti_trong(f, periods, biz, ban, rb):
    if len(periods) != 1 or not re.search(r"chiem|ti trong|dong gop|phan cua|bang bao nhieu|bang bn|bang bnhiu|ti le|bang may|la bao nhieu|la bn", f):
        return None
    if not re.search(r"%|phan tram", f) or re.search(r"so voi|cung ky|tang hay giam|tang/giam|tang truong|hon hay kem|chenh|thay doi|\bhuy|khach da mua|tu \d+ don|it nhat", f):
        return None
    p = periods[0]
    if p.start is None:
        return None
    if not re.search(r"tong doanh thu|doanh thu|dthu", f):
        return None
    if re.search(r"so don|bao nhieu don", f):
        return None
    unans = None
    if re.search(r"khach (nu|nam)|gioi tinh", f):
        if _no_gender(biz):
            unans = "Dữ liệu không ghi giới tính khách"
        else:
            return None
    elif re.search(r"khach nhom|nhom khach", f):
        if ban["nhom_khach"].notna().sum() == 0:
            m = re.search(r"(?:khach nhom|nhom khach) ([a-z ]+?)(?: chiem| bang| la\b| trong| co\b|\?|,|$)", f)
            if not m or _term_in_raw(biz, m.group(1).strip()):
                return None
            unans = "Dữ liệu không ghi nhóm khách"
        elif _ok(rb, "tt_nhom_khach"):
            vals = [v for v in ban["nhom_khach"].dropna().unique() if not str(v).startswith("(")]
            v = _find_name(f, vals, prefix=r"(?:khach nhom |nhom khach |nhom )")
            if v is None:
                return None
            cond, what = (lambda x: x["nhom_khach"] == v), f"nhóm khách {v}"
        else:
            return None
    elif re.search(r"\bke don\b", f):
        if re.search(r"khong ke don", f) or not _ok(rb, "ke_don"):
            return None
        if ban["ke_don"].notna().sum() == 0:
            return None
        else:
            cond, what = (lambda x: x["ke_don"] == True), "thuốc kê đơn"  # noqa: E712
    elif re.search(r"\bnhom\b", f) and ban["nhom_hang"].notna().any() and not re.search(r"khach", f):
        vals = ban["nhom_hang"].dropna().unique()
        v = _find_name(f, vals, prefix=r"(?:nhom hang |nhom )")
        if v is None:
            return None
        cond, what = (lambda x: x["nhom_hang"] == v), f"nhóm hàng {v}"
    elif re.search(r"tra bang", f):
        if len(re.findall(r"tra bang", f)) != 1:
            return None
        nd = getattr(biz, "no_don", None)
        if nd is None or "pttt" not in nd or nd["pttt"].notna().sum() == 0:
            return None
        cls = {v: _payc(v) for v in nd["pttt"].dropna().unique()}
        if any(c is None for c in cls.values()) or len(set(cls.values())) != len(cls):
            return None
        tail = f.split("tra bang", 1)[1]
        want = None
        for k, pat in _PAYC:
            if re.match(r"\s*(?:" + pat + r")", tail):
                want = k
        if want is None or want not in cls.values():
            return None
        ids = set(nd.loc[nd["pttt"].map(lambda v: cls.get(v) == want), "ma_don"])
        cond, what = (lambda x: x["ma_don"].isin(ids)), f"đơn trả bằng {want}"
    elif re.search(r"ban qua|kenh|ban tai|\bqua (dien thoai|zalo)", f):
        if ban["kenh"].notna().sum() == 0:
            m = re.search(r"ban qua ([a-z ]+?)(?: chiem| bang| la\b| trong| co\b|\?|,|$)", f)
            if not m or _term_in_raw(biz, m.group(1).strip()):
                return None
            unans = "Dữ liệu không ghi kênh bán"
        else:
            vals = ban["kenh"].dropna().unique()
            v = _find_name(f, vals, prefix=r"(?:qua |kenh |ban tai |tai )?")
            if v is None:
                return None
            cond, what = (lambda x: x["kenh"] == v), f"kênh {v}"
    else:
        return None
    if _outside(p, ban):
        return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
    if unans:
        return _unans(unans), unans
    x = _sel(ban, p)
    den = _rev(x)
    num = _rev(x[cond(x)])
    v = _pct(num, den)
    if v is None:
        return None
    return {"type": "percent", "answer": v}, f"Tỉ trọng {what}: {num:,.0f} / {den:,.0f}"


# ---------------------------------------------------------------- so sánh hai kỳ
def _names_in_q(f, ban):
    for n in ban["ten_hang"].dropna().unique():
        k = re.sub(r"\s*\(.*?\)\s*$", "", fold(n)).strip()
        if len(k) >= 4 and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", f):
            return True
    return False


def _tok(p):
    if p.kind != "month":
        return None
    m, y = p.start.month, p.start.year
    return [rf"thang {m}/{y}", rf"thang {m} nam {y}", rf"thang {m:02d}/{y}"]


def _pos(f, toks):
    for t in toks:
        i = f.find(t)
        if i >= 0:
            return i
    return -1


def _doi_ky(f, periods, biz, ban, rb):
    if not re.search(r"%|phan tram", f) or not re.search(r"so voi|so vs|cung ky|hon hay kem|tang|giam|thay doi|tang truong|chenh", f):
        return None
    if not _ok(rb, "pct"):
        return None
    if len(periods) == 1:
        p = periods[0]
        c = p.meta.get("compare")
    elif len(periods) == 2:
        p, c = periods
        if p.meta.get("compare") is not None or c.meta.get("compare") is not None:
            return None
        if p.start is None or c.start is None or not p.start > c.start:
            return None
        if not re.search(r"so voi|so vs|hon hay kem", f):
            return None
        ok = False
        if p.kind == "tet" and c.kind in ("tet", "before_tet"):
            ok = True
        elif p.kind == "month" and c.kind == "month":
            i, j, k = _pos(f, _tok(p)), _pos(f, _tok(c)), max(f.find("so voi"), f.find("so vs"))
            ok = 0 <= i < k < j
        if not ok:
            return None
    else:
        return None
    if p.start is None or c is None or c.start is None:
        return None
    if re.search(r"nhom|mat hang|kenh|ban qua|khach|nhan vien|ke don|tien mat|chuyen khoan|ca sang|ca chieu|hoat chat|\bmon\b|size|\bly\b|chai|hop", f):
        return None
    isrev = re.search(_REV, f)
    iso = re.search(r"so don|don hang|bao nhieu don", f)
    if bool(isrev) == bool(iso):
        return None
    if _names_in_q(f, ban):
        return None
    if _outside(p, ban) or _outside(c, ban):
        return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
    xa, xb = _sel(ban, p), _sel(ban, c)
    a, b = (_rev(xa), _rev(xb)) if isrev else (_norders(xa), _norders(xb))
    v = _pct(a - b, b)
    if v is None:
        return None
    return {"type": "percent", "answer": v}, f"({p.label} {a:,.0f} - {c.label} {b:,.0f}) / {c.label}"
