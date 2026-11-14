"""Công thức cố định MỞ RỘNG (không LLM): doanh thu / số đơn có điều kiện phụ (nhóm hàng, hoạt chất, kê đơn, kênh,
nhóm khách, nhân viên, ca, hình thức thanh toán), danh sách nhiều giá trị, ngày/tháng doanh thu cao nhất, size M/L,
tiền nhập hàng, nhà cung cấp nhập nhiều nhất, lãi gộp, khách mua >= k đơn, số đơn huỷ.
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.  Mặc định TỪ CHỐI."""
import re
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from loader import fold
from ct_coban import _TIME_RE, _TIME_TOK, _OK, _norm, _tien_quy, _sl_fmt, _words, _ten_hang

_LUAT = {
    "dt": "tong thanh tien cac dong hang",
    "dt2": "don khong huy va khong tra hang",
    "ca": "ca sang 06 00 13 59 ca chieu 14 00 21 59",
    "hc": "doanh thu 1 cac thuoc co hoat chat ghi dung nhu vay",
    "ke": "thuoc ke don san pham ghi ke don trong danh muc thuoc",
    "pt": "theo hinh thuc ghi tren don",
    "nk": "nhom theo cot nhom khach",
    "nhom": "theo cot nhom hang",
    "size": "mon co ten ket thuc bang m hoac l",
    "nhap": "tong tien cac phieu nhap co ngay phieu trong ky bo phieu huy",
    "gv": "gia von cua moi dong ban bang so luong nhan voi gia von ghi tren dong ban",
    "lai": "doanh thu 1 tru gia von cua hang da ban",
    "khach": "khach da mua trong ky la khach co it nhat mot don nhu vay",
    "tb": "gia tri trung binh mot don",
}


def _co_luat(rulebook, *keys):
    if rulebook is None:
        return False
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    t = re.sub(r"\s+", " ", t)
    for k in keys:
        pat = _LUAT[k]
        if k in ("hc",):
            pat = "cac thuoc co hoat chat ghi dung nhu vay"
        if k == "lai":
            pat = "doanh thu 1 tru gia von cua hang da ban"
        if pat not in t:
            return False
    return True


WHY = []


def _no(tag=None):
    import sys
    WHY.append(tag or "L%d" % sys._getframe(1).f_lineno)
    return None


_W = set(_OK) | set(_TIME_TOK) | set("""rieng ban duoc dc ra tong cong tien thu ve bao nhieu bn bnhiu hang cac don do tu cua gia tri
bi trang thai qua tai may so luong la order doanh dthu nha thuoc quan tiem shop mat mon hom nay hnay co thi cho coi
mua ay hoi xem giup gium minh toi biet tinh dang hien alo ad oi anh em xin nho hen goi nghe nghen di rua ri wan mik mk""".split())
W_BASE = _W - {"ca", "tra", "kiem"}
_W_KH = {"khach", "hang", "mua", "da", "tro", "len", "nhieu", "bao", "so", "may", "nguoi"}
_MAP_PT = {"cod": ("cod", "thu ho"), "ck": ("chuyen khoan", "transfer", "banking"), "tm": ("tien mat", "cash"),
           "vi": ("vi dien tu", "ewallet", "e wallet", "momo"), "no": ("ghi no", "debt")}


def _ci(s):
    return re.sub(r"[^a-z0-9]+", " ", fold(str(s))).strip()


def _alt(vals):
    ks = sorted({_ci(v) for v in vals if _ci(v)}, key=len, reverse=True)
    if not ks:
        return None
    return "(" + "|".join(r"\s+".join(re.escape(t) for t in k.split()) for k in ks) + ")"


def _norm2(g):
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9 ]+", " ", g)).strip()


def _rev(b):
    return _tien_quy(b["thanh_tien"].sum())


def _gia_nhap_bq(gb, nh, nx, periods, rulebook):
    if "gia nhap binh quan mot mat hang" and not re.search(r"binh quan theo so luong nhap", re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook or "")))):
        return _no()
    if periods[0].kind != "month" or len(periods) != 1:
        return _no()
    m = re.search(r"\b(?:mot|1|moi) (\w+)\b(?: la)?(?: bao nhieu| bn| may)?(?: tien)?\s*(?:\w+\s*)?$", gb) or re.search(r"\b(?:mot|1|moi) (\w+)\b", gb)
    if not m:
        return _no()
    unit = m.group(1)
    names = nh["ten_hang"].dropna().unique()
    gk = " " + gb + " "
    hits = []
    for n in names:
        k = _ci(n)
        if len(k) >= 4 and re.search(r"(?<![a-z0-9])" + r"\s+".join(re.escape(t) for t in k.split()) + r"(?![a-z0-9])", gk):
            hits.append((len(k), n, k))
    if not hits:
        return _no()
    hits.sort(reverse=True)
    top = hits[0]
    for h in hits[1:]:
        if h[2] not in top[2]:
            return _no()
    item_keys = {_ci(n) for _, n, _ in hits if _ci(n) == top[2]}
    pat = r"(?<![a-z0-9])" + r"\s+".join(re.escape(t) for t in top[2].split()) + r"(?![a-z0-9])"
    gx = re.sub(pat, " ITEM ", gk, count=1)
    gx = re.sub(r"\b(?:mot|1|moi) " + re.escape(unit) + r"\b", " ", gx)
    extra = {"nhap", "vao", "gia", "binh", "quan", "bao", "nhieu", "tien", "z", "ha", "ay", "nhi", "bn", "may", "la", "tinh", "cua", "hang", "trung", "ITEM"}
    for t in re.sub(r"\b(?:TIME|ITEM)\b", " ", gx).split():
        if t not in W_BASE and t not in extra:
            return _no()
    x = nx[nx["ten_hang"].map(lambda v: isinstance(v, str) and _ci(v) == top[2])]
    if not len(x) or x["don_vi"].isna().any() or x["so_luong"].isna().any() or x["thanh_tien"].isna().any():
        return _no()
    us = {_ci(v) for v in x["don_vi"]}
    if us != {unit}:
        return _no()
    if "ma_hang" in x and x["ma_hang"].nunique() > 1:
        return _no()
    sl = float(x["so_luong"].sum())
    if sl <= 0:
        return _no()
    return {"type": "number", "answer": _tien_quy(float(x["thanh_tien"].sum()) / sl)}, \
        f"Giá nhập bình quân '{top[1]}' = tổng tiền nhập / tổng số lượng nhập (đơn vị {unit}), bỏ phiếu huỷ"


class _P:
    def __init__(self, a, b, label=""):
        self.start, self.end, self.kind, self.label, self.meta = a.date(), b.date(), "month", label, {}


def _parse_months(f):
    s = re.sub(r"[^a-z0-9/\- ,]", " ", f)
    dn = re.search(r"\bnam (\d{4})\b", s)
    items = []
    for m in re.finditer(r"(?<![\d/\-.])(?:(thang|th|t) ?)?(\d{1,2})(?: ?[/-] ?(\d{4}|\d{2}))?(?![\d/\-.])", s):
        if m.group(1) is None and m.group(3) is None:
            continue
        mo = int(m.group(2))
        if not 1 <= mo <= 12:
            return None
        y = m.group(3)
        y = (int(y) + (2000 if len(y) == 2 else 0)) if y else None
        items.append([mo, y])
    if not items:
        return None
    if dn:
        for it in items:
            if it[1] is None:
                it[1] = int(dn.group(1))
    nxt = None
    for it in reversed(items):
        if it[1] is None:
            it[1] = nxt
        else:
            nxt = it[1]
    if any(it[1] is None for it in items):
        return None
    return [(it[1], it[0]) for it in items]


def solve(q, f, periods, biz, rulebook):
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban) or not periods:
        return _no()
    for c in ("ma_don", "ngay", "tinh_dt", "thanh_tien"):
        if c not in ban:
            return _no()
    if not _co_luat(rulebook, "dt", "dt2"):
        return _no()
    ov = False
    mm_ = _parse_months(f)
    if mm_ and len(mm_) == 2 and re.search(r"\btu (?:thang |th |t)?\d{1,2}[^a-z]{0,10}(?:den|toi|qua)\b", f) \
            and not (len(periods) == 1 and periods[0].kind == "range") and mm_[0] <= mm_[1]:
        try:
            ms_ = pd.period_range(pd.Timestamp(*mm_[0], 1), pd.Timestamp(*mm_[1], 1), freq="M")
            if 2 <= len(ms_) <= 12:
                periods = [_P(m_.start_time.normalize(), m_.end_time.normalize(), f"tháng {m_.month}/{m_.year}") for m_ in ms_]
                ov = True
        except Exception:
            return _no()
    elif mm_:
        if len(mm_) >= 2 and not re.search(r"\btu (?:thang |th |t)?\d{1,2}[^a-z]{0,10}(?:den|toi|qua)\b", f):
            cur = [(p.start.year, p.start.month) for p in periods if p.start is not None and p.kind == "month"]
            if cur != mm_ or len(periods) != len(mm_):
                try:
                    periods = [_P(pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0), f"tháng {m}/{y}") for y, m in mm_]
                    ov = True
                except Exception:
                    return _no()
    for p in periods:
        if p.kind == "snapshot" or p.start is None:
            return _no()
    nmin, nmax = ban["ngay"].min().normalize(), ban["ngay"].max().normalize()
    st0, en0 = pd.Timestamp(periods[0].start).normalize(), pd.Timestamp(periods[-1].end).normalize()
    if en0 < nmin or st0 > nmax:
        return _no()

    if re.search(r"(?:nhieu|may|bn|bnhiu|bao nhieu) ?%|phan tram|ti le|ty le|so voi|\bvs\b", f):
        return _no()
    g = _norm(f)
    g = re.sub(r"\b(?:nhut|nhta|nhat)\b", "nhat", g)
    g = re.sub(r"\b(?:bua|hom|ngay) (?:nao|mo)\b", "ngay nao", g)
    g = re.sub(r"\borders?\b", " don ", g)
    g = re.sub(r"\brevenue\b", " doanh thu ", g)
    gb = " " + g + " "
    # --- "N tháng liền nhau kết thúc ở tháng M/Y, từ tháng cũ đến tháng mới"
    mlist = None
    mm = re.search(r"\b(\d+) thang lien (?:nhau )?(?:ket thuc|toi|den|cuoi) (?:o |vao |tai )?(?:thang )?(\d{1,2}) ?/ ?(\d{4})", gb)
    if mm:
        n, m_, y_ = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        if not (2 <= n <= 6 and 1 <= m_ <= 12) or "tu thang cu den thang moi" not in gb:
            return _no()
        mlist = []
        yy, mo = y_, m_
        for _ in range(n):
            mlist.append((yy, mo))
            mo -= 1
            if mo == 0:
                mo, yy = 12, yy - 1
        mlist.reverse()
        gb = gb.replace(mm.group(0), " TIME ").replace("tu thang cu den thang moi", " ")
        gb = re.sub(r"\b(?:moi|tung) thang\b", " ", gb)
    for pat in _TIME_RE:
        gb = re.sub(pat, " TIME ", gb)
    gb = re.sub(r"\b(?:th|t) ?\d{1,2}\b|\bq ?\d(?: ?/ ?\d{4})?\b|\b(?:th|thang) (?:nay|ni|truoc|roi)\b|\bthang (?:\d{1,2}|nay|ni)\b", " TIME ", gb)
    gb = re.sub(r"\s+", " ", gb)
    gb = _norm2(gb)
    gb = re.sub(r"\bcong lai\b", " ", gb)
    if mlist is not None:
        pers = [(pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)) for y, m in mlist]
        if pers[-1][1] > pd.Timestamp(periods[0].end) + pd.Timedelta(days=0) and False:
            return _no()
    else:
        pers = [(pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()) for p in periods]
    if mlist is not None and len(periods) != 1:
        return _no()
    expanded = False
    if len(periods) > 1 and mlist is None and not ov and not (mm_ and len(mm_) == len(periods)) and not re.search(r"xu huong|(?:cao|thap) hon", f):
        if len(re.findall(r"\b\d{1,2} ?/ ?\d{4}\b", f)) != len(periods):
            return _no("SO_THANG")
    if len(periods) > 1:
        gb = re.sub(r"\b(?:xet |cua )?\d+ (?:thang\b|(?=TIME))|\bxet\b|\bghi \d+ so\b|\blan luot\b|\btheo thu tu (?:thoi gian|do|nhu tren|nhu cau hoi)\b|\bghi\b|\b(?:moi|tung) thang\b", " ", gb)
        gb = re.sub(r"\s+", " ", gb)
    gb = re.sub(r"\blan luot\b", " ", gb)
    if mlist is None and len(periods) == 1 and re.search(r"\b(?:moi|tung) thang\b|\b\d+ thang lien tiep\b", gb):
        a_, b_ = pers[0]
        if len(periods) != 1 or periods[0].kind != "range" or a_.day != 1 or (b_ + pd.Timedelta(days=1)).day != 1:
            return _no()
        mons = pd.period_range(a_, b_, freq="M")
        if not (2 <= len(mons) <= 12):
            return _no()
        pers = [(m_.start_time.normalize(), m_.end_time.normalize()) for m_ in mons]
        gb = re.sub(r"\b(?:moi|tung) thang\b|\b\d+ thang lien tiep\b|\bliet ke\b|\bghi \d+ so theo thu tu thoi gian\b|\bghi\b", " ", gb)
        gb = re.sub(r"\b(?:xet |cua )?\d+ so\b|\btheo thu tu thoi gian\b", " ", gb)
        expanded = True

    d = ban["ngay"].dt.normalize()
    conds = []   # (loai, mask-fn, thứ tự xuất hiện)

    def take(pat, kind, key, mask, order_pos=None):
        nonlocal gb
        ms = list(re.finditer(pat, gb))
        for m_ in ms:
            conds.append((kind, key, mask(m_), m_.start()))
        if ms:
            gb = re.sub(pat, " COND ", gb)
        return len(ms)

    # ---- size M / L
    size = None
    ms = re.findall(r"\bsize (m|l)\b", gb)
    if ms:
        if len(set(ms)) != 1 or not _co_luat(rulebook, "size"):
            return _no()
        size = ms[0]
        gb = re.sub(r"\bsize (?:m|l)\b", " SIZE ", gb)

    # ---- hoạt chất
    if "hoat_chat" in ban and ban["hoat_chat"].notna().any() and re.search(r"hoat chat", gb):
        if not _co_luat(rulebook, "hc"):
            return _no()
        hk = ban["hoat_chat"].map(lambda v: _ci(v) if pd.notna(v) else None)
        a = _alt(ban["hoat_chat"].dropna().unique())
        pat = r"(?:(?:cac )?thuoc )?(?:co )?hoat chat " + a + r"(?=\s|$)"
        take(pat, "hc", "hc", lambda m_, hk=hk: (hk == m_.group(1)))
        if not any(c[0] == "hc" for c in conds):
            return _no()
    # ---- kê đơn
    if re.search(r"ke don|bac si", gb):
        if "ke_don" not in ban or not ban["ke_don"].notna().any() or not _co_luat(rulebook, "ke"):
            return _no()
        kd = ban["ke_don"].fillna(False).astype(bool)
        neg = r"(?:cac )?(?:mat hang |san pham |thuoc |hang )?(?:khong|k|ko|hok|hong|chang) (?:can )?(?:ke don|don bac si)"
        neg2 = r"tru (?:cac )?(?:mat hang |san pham |thuoc |hang )?(?:ke don|can don bac si) ra"
        pos = r"(?:cac )?thuoc (?:phai |can )?(?:ke don|don bac si)"
        n1 = take(neg, "kd", "kd", lambda m_, kd=kd: ~kd)
        n2 = take(neg2, "kd", "kd", lambda m_, kd=kd: ~kd)
        n3 = take(pos, "kd", "kd", lambda m_, kd=kd: kd)
        if n1 + n2 + n3 != 1 or re.search(r"ke don|bac si", gb):
            return _no()
        if n2:
            gb = re.sub(r"\b(?:cac )?(?:mat hang |san pham |hang )?con lai\b", " ", gb)
    # ---- nhóm khách
    if "nhom_khach" in ban and ban["nhom_khach"].notna().any() and re.search(r"\bnhom\b", gb):
        vals = ban["nhom_khach"].dropna().unique()
        a = _alt(vals)
        if a:
            nk = ban["nhom_khach"].map(lambda v: _ci(v) if pd.notna(v) else None)
            p1 = r"(?:do |tu |cua )?(?:cac )?khach(?: hang)? (?:thuoc )?nhom " + a + r"(?: mua)?(?=\s|$)"
            p2 = r"nhom khach(?: hang)? " + a + r"(?=\s|$)"
            c0 = len(conds)
            take(p1, "nk", "nk", lambda m_, nk=nk: (nk == m_.group(1)))
            take(p2, "nk", "nk", lambda m_, nk=nk: (nk == m_.group(1)))
            if len(conds) > c0 and not _co_luat(rulebook, "nk"):
                return _no()
    # ---- nhóm hàng
    if "nhom_hang" in ban and ban["nhom_hang"].notna().any() and re.search(r"\bnhom\b", gb):
        a = _alt(ban["nhom_hang"].dropna().unique())
        if a:
            nh = ban["nhom_hang"].map(lambda v: _ci(v) if pd.notna(v) else None)
            p = r"(?:(?:cac )?(?:mat hang |san pham |mon |hang )?)nhom (?:hang )?" + a + r"(?=\s|$)"
            c0 = len(conds)
            take(p, "nh", "nh", lambda m_, nh=nh: (nh == m_.group(1)))
            if len(conds) > c0 and not _co_luat(rulebook, "nhom"):
                return _no()
    # ---- nhân viên
    _vals = list(ban["nhan_vien"].dropna().unique()) if "nhan_vien" in ban else []
    _codes = [v for v in _vals if re.search(r"\d", str(v))]
    _a3 = _alt(_codes)
    if re.search(r"\bnhan vien\b|\bnv\b", gb) or (_a3 and re.search(_a3, gb)):
        vals = list(_vals)
        if "ten_nhan_vien" in ban:
            vals += list(ban["ten_nhan_vien"].dropna().unique())
        a = _alt(vals)
        if not a:
            return _no()
        nv1 = ban["nhan_vien"].map(lambda v: _ci(v) if pd.notna(v) else None) if "nhan_vien" in ban else pd.Series(None, index=ban.index)
        nv2 = ban["ten_nhan_vien"].map(lambda v: _ci(v) if pd.notna(v) else None) if "ten_nhan_vien" in ban else pd.Series(None, index=ban.index)
        p = r"(?:do |cua )?(?:nhan vien|nv)(?: ma)? " + a + r"(?=\s|$)"
        take(p, "nv", "nv", lambda m_: (nv1 == m_.group(1)) | (nv2 == m_.group(1)))
        if _a3 and not any(c[0] == "nv" for c in conds):
            take(r"(?:do |cua )?(?:ma )?" + _a3 + r"(?=\s|$)", "nv", "nv", lambda m_: (nv1 == m_.group(1)))
        if re.search(r"\bnhan vien\b|\bnv\b", gb):
            return _no()
    # ---- kênh
    if "kenh" in ban and ban["kenh"].notna().any() and re.search(r"\bqua\b|\btai\b|\bkenh\b", gb):
        a = _alt(ban["kenh"].dropna().unique())
        if a:
            kk = ban["kenh"].map(lambda v: _ci(v) if pd.notna(v) else None)
            p = r"(?:(?:cac )?don (?:hang )?(?:duoc )?)?(?:(?:ban|dat|mua) )?(?:qua|tai|bang|tren) (?:kenh )?" + a + r"(?=\s|$)"
            take(p, "kenh", "kenh", lambda m_, kk=kk: (kk == m_.group(1)))
            a2 = _alt([v for v in ban["kenh"].dropna().unique() if _ci(v) not in ("dien thoai", "phone")])
            if a2:
                take(r"(?:cac )?don (?:hang )?(?:duoc )?(?:(?:ban|dat|mua) )?" + a2 + r"(?=\s|$)", "kenh", "kenh", lambda m_, kk=kk: (kk == m_.group(1)))
            take(r"\bkenh " + a + r"(?=\s|$)", "kenh", "kenh", lambda m_, kk=kk: (kk == m_.group(1)))
    # ---- hình thức thanh toán
    if re.search(r"\b(?:tra|thanh toan|thu)\b (?:bang|qua|theo)", gb):
        nd = getattr(biz, "no_don", None)
        if nd is None or "pttt" not in nd or not _co_luat(rulebook, "pt"):
            return _no()
        pm = {"cod": r"cod|thu ho", "ck": r"chuyen khoan|ck|banking", "tm": r"tien mat|tien tuoi|tm", "vi": r"vi dien tu|vi|momo", "no": r"ghi no|cong no|mua chiu"}
        allp = "|".join(pm.values())
        p = r"(?:(?:cac )?don (?:hang )?(?:duoc )?)?(?:tra|thanh toan|thu) (?:bang|qua|theo) (?:hinh thuc )?(" + allp + r")(?=\s|$)"
        pk = nd.drop_duplicates("ma_don").set_index("ma_don")["pttt"].map(_ci)
        ptt = ban["ma_don"].map(pk)

        def mk(m_, ptt=ptt, pm=pm):
            w = m_.group(1)
            key = next(k for k, v in pm.items() if re.fullmatch(v, w))
            keys = _MAP_PT[key]
            return ptt.map(lambda s: isinstance(s, str) and any(re.search(r"(^| )" + re.escape(x) + r"( |$)", s) for x in keys)
                           and not (key == "tm" and re.search(r"cod|thu ho", s)))
        n = take(p, "pt", "pt", mk)
        if not n or re.search(r"\b(?:tra|thanh toan) (?:bang|qua|theo)\b", gb):
            return _no()
        if not ptt.notna().any():
            return _no()
    # ---- ca
    mca = re.findall(r"\bca (sang|chieu)\b", gb)
    if mca:
        if len(mca) != 1 or "thoi_diem" not in ban or not _co_luat(rulebook, "ca"):
            return _no()
        ts = pd.to_datetime(ban["thoi_diem"], errors="coerce")
        if ts.notna().mean() < 0.9 or ts.dt.hour.nunique() < 6:
            return _no()
        h = ts.dt.hour
        mk_ = ((h >= 6) & (h <= 13)) if mca[0] == "sang" else ((h >= 14) & (h <= 21))
        conds.append(("ca", "ca", mk_, 0))
        gb = re.sub(r"\bca (?:sang|chieu)\b", " COND ", gb)

    gb = re.sub(r"\s+", " ", gb).strip()
    gb = " " + gb + " "
    # kết hợp điều kiện: cùng cột lặp lại -> chế độ nhiều giá trị
    kinds = [c[0] for c in conds]
    multi = None
    if len(set(kinds)) < len(kinds):
        if len(set(kinds)) == 1 and kinds[0] in ("kenh", "pt") and len(kinds) in (2, 3):
            multi = sorted(conds, key=lambda c: c[3])
        else:
            return _no()
    if multi is None and len({c[0] for c in conds}) != len(conds):
        return _no()
    base_mask = pd.Series(True, index=ban.index)
    if multi is None:
        for c in conds:
            base_mask &= c[2].fillna(False).astype(bool) if hasattr(c[2], "fillna") else c[2]
    stc = [x for x in conds if x[0] == "x"]

    def inp(a, b):
        return (d >= a) & (d <= b)

    _mx = {"voi", "ben", "moi", "loai", "nhau"} if multi is not None else set()
    rest_tok = lambda s, extra=(): [t for t in re.sub(r"\b(?:TIME|COND|SIZE|INTENT)\b", " ", s).split()
                                    if t not in W_BASE and t not in extra and t not in _mx]
    REVI = r"doanh thu|dthu|tien hang|tien ban|bao nhieu tien|bn tien|nhieu tien|tong tien|thu (?:duoc|ve)|tong gia tri"
    sub = ban[ban["tinh_dt"]]

    # ============ SIZE M/L
    if size is not None:
        if conds or len(pers) != 1:
            return _no()
        mu = re.search(r"(?:bao nhieu|may|bn|nhieu|so|tong so|total) (?:ly|coc)\b|(?:ly|coc) size", gb)
        if "ly" not in gb:
            return _no()
        if rest_tok(gb, {"ly", "coc", "quan", "cong", "moi", "mon", "tat", "ca", "loai", "do", "uong", "ta", "nhi", "z", "ha", "ban", "mua",
                         "tinh", "bao", "nhieu", "total", "tong", "so", "bn", "ra"}) and True:
            return _no()
        if "ca" in gb.split() and not re.search(r"tat ca|cong tat ca|moi", gb):
            return _no()
        if "don_vi_co_ban" not in ban or "ten_hang" not in ban:
            return _no()
        x = sub[inp(*pers[0])[ban["tinh_dt"]]]
        x = x[x["ten_hang"].astype("string").str.contains(r"\(" + size + r"\)\s*$", case=False, regex=True, na=False)]
        if (x["don_vi_co_ban"].map(lambda v: fold(str(v))) != "ly").any() or not len(x) or x["so_luong_co_ban"].isna().any():
            return _no()
        return {"type": "number", "answer": _sl_fmt(x["so_luong_co_ban"].sum())}, f"Số ly size {size.upper()} (tên món kết thúc ({size.upper()})) kỳ {periods[0].label}"

    def sel(a, b, mask=None):
        m = ban["tinh_dt"] & inp(a, b)
        if mask is not None:
            m &= mask
        return ban[m]

    # ============ TĂNG / GIẢM LIÊN TỤC (boolean) qua các tháng nêu
    mt = re.search(r"\b(tang|giam) (?:lien tuc|deu|dan deu)\b|\bluon (tang|giam)\b", gb)
    if mt and not conds:
        if len(pers) < 2 or mlist is not None or any(p.kind != "month" for p in periods) or expanded:
            return _no()
        if not re.search(r"doanh thu|dthu", gb):
            return _no()
        gx = re.sub(r"doanh thu|dthu|\b(?:tang|giam) (?:lien tuc|deu|dan deu)\b|\bluon (?:tang|giam)\b|\bco\b|\bphai\b|\bxu huong\b|\bhok\b|\bkhong\b|\bkg\b|\bko\b|\bk\b|\bhong\b|\bvay\b|\bnhi\b|\bkhong\b", " ", gb)
        if rest_tok(gx, {"qua", "r", "roi", "xu", "huong"}):
            return _no()
        if [a for a, b in pers] != sorted(a for a, b in pers):
            return _no()
        v = [sel(a, b)["thanh_tien"].sum() for a, b in pers]
        sg = mt.group(1) or mt.group(2)
        ok = all(v[i + 1] > v[i] for i in range(len(v) - 1)) if sg == "tang" else all(v[i + 1] < v[i] for i in range(len(v) - 1))
        return {"type": "boolean", "answer": bool(ok)}, "Doanh thu các tháng nêu tăng/giảm liên tục (so liền kề, nghiêm ngặt)"

    # ============ SỐ LƯỢNG theo đơn vị cơ bản (một mặt hàng hoặc tất cả hàng có đơn vị đó)
    if re.search(r"\bban\b", gb) and not re.search(REVI, gb) and all(c[0] == "nh" for c in conds) and len(conds) <= 1 and len(pers) == 1 and "don_vi_co_ban" in ban and "so_luong_co_ban" in ban:
        gz = gb.replace("TIME", "ztime")
        g1, item = _ten_hang(gz, q, ban)
        if item == "AMBIG":
            return _no()
        g1 = " " + g1 + " "
        un = {fold(str(v)) for v in ban["don_vi_co_ban"].dropna().unique()}
        stop = {"ITEM", "ztime", "ban", "mua", "duoc", "dc", "ra", "tong", "cong", "luong", "la", "bao", "nhieu"}
        mu = None
        for m_ in re.finditer(r"(?:bao nhieu|bnhiu|bn|nhieu|may|tong so|so) (\w+)(?= |$)", g1):
            if m_.group(1) not in stop:
                mu = m_
                break
        if mu is not None and (mu.group(1) in un or item):
            unit = mu.group(1)
            g2 = g1[:mu.start()] + " INTENT " + g1[mu.end():]
            g2 = re.sub(r"\b(?:cong |tinh )?(?:moi|tat ca|cac) (?:mon|mat hang|loai)(?: do uong)?\b|\btat ca\b", " ", g2)
            g2 = g2.replace("ztime", "TIME")
            bad = [t for t in re.sub(r"\b(?:TIME|ITEM|INTENT)\b", " ", g2).split() if t not in W_BASE and t not in {"ta", "nhi", "z", "ha", "ay", "cong", "tinh", "loai", "do", "uong"}]
            if bad:
                return _no("QTYREST:" + repr(bad))
            x = sel(*pers[0], mask=base_mask if conds else None)
            if item and conds:
                return _no()
            if item:
                sb = ban[ban["ten_hang"] == item]
                if "la_phi" in sb and sb["la_phi"].any():
                    return _no()
                if "ma_hang" in ban and sb["ma_hang"].nunique() > 1:
                    return _no()
                if sb["don_vi_co_ban"].dropna().nunique() != 1 or fold(str(sb["don_vi_co_ban"].dropna().iloc[0])) != unit:
                    return _no()
                x = x[x["ten_hang"] == item]
            else:
                if unit not in un or unit in ("cai", "con", "chiec") and False:
                    return _no()
                x = x[x["don_vi_co_ban"].map(lambda v: isinstance(v, str) and fold(v) == unit) & ~x["la_phi"]]
                if (ban["don_vi_co_ban"].isna() & ban["tinh_dt"]).any():
                    return _no()
            if x["so_luong_co_ban"].isna().any():
                return _no()
            return {"type": "number", "answer": _sl_fmt(x["so_luong_co_ban"].sum())}, \
                "Số lượng quy về đơn vị cơ bản " + (f"của {item}" if item else f"mọi mặt hàng có đơn vị cơ bản là {unit}")


    # ============ MẶT HÀNG LÃI GỘP CAO NHẤT
    if re.search(r"lai gop|loi nhuan gop|loi gop", gb) and re.search(r"\bnhat\b", gb) and re.search(r"\b(?:hang nao|mat hang nao|san pham nao|mon nao|san pham|mat hang|hang)\b", gb):
        if conds or len(pers) != 1 or not _co_luat(rulebook, "gv", "lai") or "gia_von" not in ban:
            return _no()
        if not re.search(r"(?:mat hang|san pham|hang|mon) nao (?:co )?(?:lai|loi nhuan|loi) gop (?:ca thang )?(?:cao|lon|nhieu) nhat|(?:mat hang|san pham|hang|mon) (?:mang lai )?(?:lai|loi nhuan|loi) gop (?:cao|lon|nhieu) nhat", gb):
            return _no()
        gx = re.sub(r"(?:mat hang|san pham|hang|mon) nao (?:co )?(?:lai|loi nhuan|loi) gop (?:ca thang )?(?:cao|lon|nhieu) nhat|(?:mat hang|san pham|hang|mon) (?:mang lai )?(?:lai|loi nhuan|loi) gop (?:cao|lon|nhieu) nhat", " INTENT ", gb)
        if rest_tok(gx, {"la", "gi", "nao", "z", "vay", "ay", "nhi", "ha", "mang", "lai", "nhat", "ca"}):
            return _no()
        x = sel(*pers[0])
        x = x[~x["la_phi"]] if "la_phi" in x else x
        if not len(x) or x["gia_von"].isna().any() or x["so_luong"].isna().any() or x["ten_hang"].isna().any():
            return _no()
        x = x.assign(_l=x["thanh_tien"] - pd.to_numeric(x["so_luong"]) * pd.to_numeric(x["gia_von"]))
        key = "ma_hang" if "ma_hang" in x and x["ma_hang"].notna().all() else "ten_hang"
        s_ = x.groupby(key)["_l"].sum().round(6)
        top = s_[s_ == s_.max()]
        if len(top) != 1:
            return _no()
        nm = x.loc[x[key] == top.index[0], "ten_hang"]
        return {"type": "string", "answer": str(nm.mode().iloc[0])}, "Mặt hàng có tổng lãi gộp cao nhất trong kỳ"

    # ============ "tháng A cao hơn tháng B và tháng C cao hơn tháng A" (boolean)
    if re.search(r"\b(?:cao|thap) hon\b", gb) and re.search(r"doanh thu|dthu", gb) and not conds:
        _pp = r"(?:thang |th |t)?(\d{1,2}) ?/ ?(\d{4}|\d{2}) (cao|thap) hon (?:thang |th |t)?(\d{1,2}) ?/ ?(\d{4}|\d{2})"
        nf = _norm(f)
        raw = re.findall(_pp, nf)
        if not raw:
            return _no()
        gx = re.sub(_pp, " ", nf)
        gx = re.sub(r"\bxu huong\b|\bdoanh thu\b|\bdthu\b|\bco phai\b|\bphai\b|\bco\b|\bva\b|\broi\b|\blai\b|\bcung\b|\bdung\b|\bvay\b|\bkhong\b|\bko\b|\bk\b|\bhok\b|\bthang\b|\b\d+\b", " ", gx)
        gx = re.sub(r"[^a-z ]+", " ", gx)
        if [t for t in gx.split() if t not in {"nhi", "ha", "ay", "a", "ah", "ak", "nha", "nhe", "u", "z", "dung", "ro", "hen"}]:
            return _no()
        res = []
        for m1, y1, cmp_, m2, y2 in raw:
            y1 = int(y1) + (2000 if len(y1) == 2 else 0)
            y2 = int(y2) + (2000 if len(y2) == 2 else 0)
            if not (1 <= int(m1) <= 12 and 1 <= int(m2) <= 12):
                return _no()
            a1 = pd.Timestamp(y1, int(m1), 1); a2 = pd.Timestamp(y2, int(m2), 1)
            if a1 > nmax or a2 > nmax or a1 + pd.offsets.MonthEnd(0) < nmin or a2 + pd.offsets.MonthEnd(0) < nmin:
                return _no()
            v1 = sel(a1, a1 + pd.offsets.MonthEnd(0))["thanh_tien"].sum()
            v2 = sel(a2, a2 + pd.offsets.MonthEnd(0))["thanh_tien"].sum()
            res.append(v1 > v2 if cmp_ == "cao" else v1 < v2)
        return {"type": "boolean", "answer": bool(all(res))}, "Các so sánh doanh thu tháng đều đúng"

    # ============ AI MUA NHIỀU TIỀN NHẤT (khách top theo doanh thu)
    if re.search(r"\bai\b", gb) and re.search(r"\bmua\b", gb) and re.search(r"nhieu|cao|lon", gb):
        if conds or len(pers) != 1 or not _co_luat(rulebook, "khach") or "khach" not in ban or ban["khach"].notna().sum() == 0 or "ten_khach" not in ban:
            return _no()
        if not re.search(r"\bai (?:la )?(?:nguoi )?(?:da )?mua (?:hang )?(?:nhieu|cao) (?:tien )?nhat\b|\bai (?:la )?(?:nguoi )?(?:da )?mua (?:hang )?nhieu tien nhat\b", gb):
            return _no()
        gx = re.sub(r"\bai (?:la )?(?:nguoi )?(?:da )?mua (?:hang )?(?:nhieu|cao) (?:tien )?nhat\b", " INTENT ", gb)
        if rest_tok(gx, {"nhat", "vay", "ay", "nhi", "z"}):
            return _no()
        x = sel(*pers[0]).dropna(subset=["khach"])
        if not len(x):
            return _no()
        s_ = x.groupby("khach")["thanh_tien"].sum().round(6)
        top = s_[s_ == s_.max()]
        if len(top) != 1:
            return _no()
        nm = x.loc[x["khach"] == top.index[0], "ten_khach"].dropna()
        if not len(nm):
            return _no()
        return {"type": "string", "answer": str(nm.mode().iloc[0])}, "Khách có doanh thu cao nhất trong kỳ"

    # ============ NGÀY / THÁNG doanh thu cao nhất
    if re.search(r"\b(?:ngay|thang) nao\b", gb):
        mday = re.search(r"\bngay nao\b", gb)
        mth = re.search(r"\bthang nao\b", gb)
        if conds or (len(pers) != 1 and not (mth and len(pers) >= 2 and all(p.kind == "month" for p in periods) and mlist is None)):
            return _no()
        if bool(mday) == bool(mth):
            return _no()
        if not re.search(r"(?:ban|thu) (?:duoc|dc|ra|ve)? ?(?:nhieu|cao) (?:tien |doanh thu )?nhat|doanh thu (?:thang )?(?:nao )?(?:cao|nhieu|lon) nhat|nhieu tien nhat|nhieu nhat", gb):
            return _no()
        if rest_tok(gb, {"nao", "nhat", "nhieu", "cao", "lon", "xet", "ve", "tien", "tong", "cong", "nhat", "doanh", "thu", "ban", "duoc", "dc", "ra", "nhi", "ha", "ay"}):
            return _no()
        if re.search(r"\b(?:thap|it|khong|tru|ca|chi)\b", gb):
            return _no()
        if len(pers) >= 2:
            s_ = pd.Series({str(a.to_period("M")): sel(a, b)["thanh_tien"].sum() for a, b in pers}).round(6)
            top = s_[s_ == s_.max()]
            if len(top) != 1 or s_.max() <= 0:
                return _no()
            return {"type": "date", "answer": top.index[0]}, "Tháng có doanh thu cao nhất trong các tháng nêu"
        x = sel(*pers[0])
        if not len(x):
            return _no()
        if mday:
            s_ = x.groupby(x["ngay"].dt.normalize())["thanh_tien"].sum().round(6)
            mx = s_.max()
            top = s_[s_ == mx]
            if len(top) != 1:
                return _no()
            return {"type": "date", "answer": top.index[0].strftime("%Y-%m-%d")}, "Ngày có doanh thu cao nhất trong kỳ"
        a, b = pers[0]
        if periods[0].kind != "range" or b < a + pd.Timedelta(days=27):
            return _no()
        if a.day != 1 or (b + pd.Timedelta(days=1)).day != 1:
            return _no()
        s_ = x.groupby(x["ngay"].dt.to_period("M"))["thanh_tien"].sum().round(6)
        mx = s_.max()
        top = s_[s_ == mx]
        if len(top) != 1:
            return _no()
        return {"type": "date", "answer": str(top.index[0])}, "Tháng có doanh thu cao nhất trong khoảng"

    # ============ NHẬP HÀNG
    if re.search(r"\bnhap\b|phieu nhap", gb):
        nh = getattr(biz, "nhap", None)
        if nh is None or conds or len(pers) != 1 or not _co_luat(rulebook, "nhap"):
            return _no()
        if "ban" in gb.split() or re.search(r"\b(tru|ton|so voi|tang|giam|nhieu hon|it hon)\b", gb) or (re.search(r"\b(binh quan|trung binh)\b", gb) and "nhap" not in gb.split()):
            if not re.search(r"nha cung cap", gb):
                return _no()
        nx = nh[nh["tinh"] & (nh["ngay"] >= pers[0][0]) & (nh["ngay"] <= pers[0][1])]
        if re.search(r"binh quan|trung binh", gb):
            return _gia_nhap_bq(gb, nh, nx, periods, rulebook)
        if nh["ngay"].isna().mean() > 0.01 or ban["ngay"].isna().all():
            return _no()
        if re.search(r"nha cung cap", gb):
            if not re.search(r"nao|ai", gb) or not re.search(r"nhieu (?:tien )?(?:hang )?nhat|cao nhat|nhieu nhat", gb) or "ncc" not in nh:
                return _no()
            if rest_tok(gb, {"nha", "cung", "cap", "nao", "nhap", "nhieu", "nhat", "tien", "hang", "duoc", "nhap", "vay", "ai", "cua", "phieu", "tong", "tu", "cao", "gia", "tri"}):
                return _no()
            if not len(nx) or nx["ncc"].isna().any() or nx["thanh_tien"].isna().any():
                return _no()
            s_ = nx.groupby("ncc")["thanh_tien"].sum().round(6)
            top = s_[s_ == s_.max()]
            if len(top) != 1:
                return _no()
            return {"type": "string", "answer": str(top.index[0])}, "Nhà cung cấp có tiền nhập cao nhất trong kỳ"
        if not re.search(r"tien|gia tri", gb):
            return _no()
        if rest_tok(gb, {"nhap", "phieu", "gia", "tri", "het", "bo", "ra", "vao", "tien", "hang", "tong", "cong", "bao", "nhieu", "mat", "cac", "ve", "tong", "so", "ca", "cua", "hang", "shop"}):
            return _no()
        if not len(nx):
            return _no()
        if nx["thanh_tien"].isna().any():
            return _no()
        # tổng phiếu: ưu tiên tổng các dòng; nếu có tổng phiếu thì phải khớp
        v = nx["thanh_tien"].sum()
        return {"type": "number", "answer": _tien_quy(v)}, "Tiền nhập hàng = tổng thành tiền các dòng của phiếu nhập không huỷ có ngày phiếu trong kỳ"

    # ============ LÃI GỘP
    if re.search(r"\blai gop\b|\bloi gop\b|\bloi nhuan gop\b", gb):
        if conds or len(pers) != 1 or not _co_luat(rulebook, "gv", "lai"):
            return _no()
        if rest_tok(gb, {"lai", "gop", "loi", "nhuan", "la", "bao", "nhieu", "tinh", "cua", "hang", "tong", "cong", "bao", "so", "tien", "duoc"}):
            return _no()
        if "gia_von" not in ban or "so_luong" not in ban:
            return _no()
        x = sel(*pers[0])
        if not len(x) or x["gia_von"].isna().any() or x["so_luong"].isna().any():
            return _no()
        v = x["thanh_tien"].sum() - (pd.to_numeric(x["so_luong"]) * pd.to_numeric(x["gia_von"])).sum()
        return {"type": "number", "answer": _tien_quy(v)}, "Lãi gộp = doanh thu - tổng(số lượng x giá vốn dòng bán)"

    # ============ KHÁCH mua từ k đơn trở lên
    mk_ = re.search(r"(?:bao nhieu|may|so|nhieu) khach(?: hang)? (?:da )?mua (?:tu )?(\d+) don (?:hang )?tro len", gb) or \
        re.search(r"(?:bao nhieu|may|so|nhieu) khach(?: hang)? (?:da )?mua (?:hang )?(?:tu )?(\d+) don (?:hang )?tro len", gb)
    if mk_:
        if conds or len(pers) != 1 or not _co_luat(rulebook, "khach") or "khach" not in ban or ban["khach"].notna().sum() == 0:
            return _no()
        gx = gb.replace(mk_.group(0), " INTENT ")
        if rest_tok(gx, {"nhieu"}):
            return _no()
        k = int(mk_.group(1))
        x = sel(*pers[0])
        c = x.dropna(subset=["khach"]).groupby("khach")["ma_don"].nunique()
        return {"type": "number", "answer": int((c >= k).sum())}, f"Số khách có >= {k} đơn tính doanh thu trong kỳ"

    # ============ SỐ ĐƠN HUỶ
    if re.search(r"huy", gb):
        if conds or len(pers) != 1 or "trang_thai" not in ban:
            return _no()
        if not re.search(r"(?:bao nhieu|may|so) (?:luong )?don (?:hang )?(?:bi |o trang thai |da |trang thai )?huy|huy (?:bao nhieu|may) don|bi huy (?:bao nhieu|may) don|don (?:hang )?(?:bi |da |o trang thai )?huy (?:la )?(?:bao nhieu|may)", gb):
            return _no()
        gx = re.sub(r"(?:bao nhieu|may|so) (?:luong )?don (?:hang )?(?:bi |o trang thai |da |trang thai )?huy|huy (?:bao nhieu|may) don|bi huy (?:bao nhieu|may) don|don (?:hang )?(?:bi |da |o trang thai )?huy (?:la )?(?:bao nhieu|may)", " INTENT ", gb)
        if rest_tok(gx, {"trang", "thai", "co", "huy"}) or re.search(r"\b(?:ti le|ty le|phan tram)\b", gx):
            return _no()
        a = ban[inp(*pers[0])]
        return {"type": "number", "answer": int(a.loc[a["trang_thai"] == "huỷ", "ma_don"].nunique())}, "Số đơn huỷ trong kỳ"

    # ============ SỐ ĐƠN / DOANH THU / TRUNG BÌNH
    DON = r"(?:don hang|don|bill|hoa don)"
    avg = re.search(r"(?:doanh thu |dthu |gia tri |tien )?(?:trung binh|binh quan)(?: (?:cua|tren|moi|mot|1))* ?(?:moi |mot |1 )?" + DON, gb)
    cnt = re.search(r"(?:tong )?(?:co )?(?:bao nhieu|may|so)(?: luong)? " + DON + r"\b|\b" + DON + r" (?:ban duoc|da ban)", gb)
    rev = re.search(REVI, gb)
    intent = None
    if avg:
        if multi or len(pers) != 1 or not _co_luat(rulebook, "tb") or any(c[0] not in ("nk", "kenh", "nv", "ca", "pt") for c in conds):
            return _no()
        intent = "AVG"
        gx = re.sub(r"(?:doanh thu |dthu |gia tri |tien )?(?:trung binh|binh quan)(?: (?:cua|tren|moi|mot|1))* ?(?:moi |mot |1 )?" + DON, " INTENT ", gb)
        gx = re.sub(REVI, " ", gx)
        if rest_tok(gx, {"nhieu"}):
            return _no()
    elif cnt and not rev:
        intent = "CNT"
        gx = re.sub(r"(?:tong )?(?:co )?(?:bao nhieu|may|so)(?: luong)? " + DON + r"\b(?: hang)?|\b" + DON + r" (?:ban duoc|da ban)", " INTENT ", gb)
        if rest_tok(gx, {"nhieu", "ban", "duoc"}):
            return _no()
    elif rev and not cnt:
        intent = "REV"
        gx = re.sub(REVI, " INTENT ", gb)
        if rest_tok(gx, {"nhieu"}):
            return _no("REVREST:" + repr(rest_tok(gx, {"nhieu"})) + gx)
        if re.search(r"\b(?:khach|ti le|ty le|phan tram|so voi|tang|giam|trung binh|nhat|top|moi)\b", gx) and not (multi is not None and not re.search(r"\b(?:khach|ti le|ty le|phan tram|so voi|tang|giam|trung binh|nhat|top)\b", gx)):
            return _no()
    else:
        return _no()
    if len(periods) > 1 and (conds or multi):
        return _no()
    if len(periods) > 1 and intent != "REV":
        return _no()
    # cond "pt"/"kenh" lặp: danh sách; các nhãn kỳ lặp: danh sách theo kỳ
    if multi is not None:
        if intent not in ("REV", "CNT") or len(pers) != 1 or gx.count("INTENT") < 1:
            return _no()
        out = []
        for c in multi:
            x = sel(*pers[0], mask=c[2].fillna(False).astype(bool))
            out.append(_rev(x) if intent == "REV" else int(x["ma_don"].nunique()))
        return {"type": "list", "answer": out}, f"Danh sách {intent} theo từng điều kiện, kỳ {periods[0].label}"
    if len(pers) > 1:
        if any(p.kind != "month" for p in periods) and mlist is None and not expanded:
            return _no()
        out = [_rev(sel(a, b)) for a, b in pers]
        return {"type": "list", "answer": out}, "Doanh thu từng tháng theo thứ tự nêu"
    if len(pers) == 1 and mlist is not None:
        pass
    x = sel(*pers[0], mask=base_mask if conds else None)
    if mlist is not None:
        return {"type": "list", "answer": [_rev(sel(a, b)) for a, b in pers]}, "Doanh thu từng tháng, từ cũ đến mới"
    if not conds and intent != "AVG" and False:
        return _no()
    if intent == "REV":
        return {"type": "number", "answer": _rev(x)}, f"Doanh thu có điều kiện {kinds} kỳ {periods[0].label}"
    if intent == "CNT":
        return {"type": "number", "answer": int(x["ma_don"].nunique())}, f"Số đơn có điều kiện {kinds} kỳ {periods[0].label}"
    if intent == "AVG":
        n = x["ma_don"].nunique()
        if n == 0:
            return _no()
        v = (Decimal(repr(round(float(x["thanh_tien"].sum()), 6))) / Decimal(int(n))).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return {"type": "number", "answer": int(v)}, f"Doanh thu trung bình mỗi đơn kỳ {periods[0].label}"
    return _no()
