"""Công thức cho các luật CHỈ có ở rulebook public (so với train) + nhập hàng (không LLM).
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.  Mặc định TỪ CHỐI.
Mỗi công thức chỉ bật khi rulebook đang chạy có đúng câu định nghĩa (kiểm chữ bỏ dấu như _co_luat); rulebook None -> tắt hết."""
import re
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from loader import fold

_LUAT = {
    "khach": ["xac dinh khach"],
    "tt": ["khach trung thanh nhat", "nhieu ngay co mua nhat"],
    "sop": ["khach sop nhat", "gia tri trung binh mot don cao nhat trong ky", "tu 3 don tro len"],
    "doi": ["khach hay doi y nhat", "nhieu don huy nhat"],
    "uc": ["mat hang duoc ua chuong nhat", "co mat trong nhieu don nhat"],
    "cc": ["cham chi nhat", "nhieu ngay lam viec nhat"],
    "ngay": ["ngay dat khach nhat", "ngay co nhieu don nhat"],
    "gio": ["gio cao diem", "gio bat dau khung"],
    "tb": ["trung binh moi ngay thu x cua ky", "chia cho so ngay thu x trong ky"],
    "ncc_chinh": ["nha cung cap chinh", "nhieu phieu nhap nhat"],
    "nhap": ["tong tien cac phieu nhap co ngay phieu trong ky bo phieu huy"],
    "ncc_tien": ["nha cung cap nhap nhieu nhat nha cung cap co tien nhap cao nhat"],
    "phi": ["phi phuc vu khong phai mat hang"],
    "dt": ["tong thanh tien cac dong hang"],
}


def _gon(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(s or "")))


_CACHE = {}


def _co(rulebook, *keys):
    if rulebook is None:
        return False
    if id(rulebook) not in _CACHE:
        _CACHE.clear()
        _CACHE[id(rulebook)] = _gon(rulebook)
    t = _CACHE[id(rulebook)]
    return all(p in t for k in keys for p in _LUAT[k])


# ----------------------------------------------------------------------------- tiện ích
def _tien(v):
    return int(Decimal(repr(float(v))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _r2(v):
    return float(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


_SO_CHU = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
_DT = r"(?:khach|mat hang|san pham|sp|mon|hang|nguoi|nhan vien|tho|nha cung cap|ncc)"


def _N(f):
    m = re.search(r"\btop (\d{1,2})\b", f) or re.search(r"(?<![/\d])\b(\d{1,2}) " + _DT + r"\b", f)
    if m:
        return int(m[1])
    m = re.search(r"\btop (hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", f) \
        or re.search(r"\b(hai|ba|bon|sau|bay|tam|chin|muoi) " + _DT + r"\b", f)
    return _SO_CHU[m[1]] if m else None


_OK = set("""cho toi hoi xem giup gium minh mk mik biet cua hang tiem shop quan sop tiem cuahang la ai nao gi vay z ha nhi nha a oi ad em anh chi
do day thi co trong tat ca cac thang th t quy q nam tuan truoc roi nay qua vua hom cuoi dau mot nhat tren ten khach nguoi ban nhan vien
mat san pham mon tho duoc bao nhieu may tinh tong cong so don hoa bill gio ngay luot xe phieu nhap cung cap ncc tien
va voi ve de hen nghen nghe dung khong vui long lam on nhe ne hay xin top nhung gom ke ra
cho minh nhe a ha ne tui t toi nhan cua gui nhat la gi nay nao day hok ko nhi z
coi check ta bn bnhiu nhiu nhieu tu jum kiem ho hen nhut nhat bao ton wan kua zo vo""".split())
_OK_GENERAL = _OK


def _sach(f, intents, extra=()):
    """Sau khi bỏ cụm ý định + số + cụm 'cả tháng' còn lại chỉ là chữ đệm / chữ chỉ kỳ -> True."""
    g = f
    for r in intents:
        g = re.sub(r, " ", g)
    g = re.sub(r"\bca (thang|quy|nam|ngay)\b", " ", g)
    g = re.sub(r"\d+", " ", g)
    toks = re.findall(r"[a-z]+", g)
    ok = _OK | set(extra)
    return all(t in ok for t in toks)


def _rank(items):
    """items: list (key, value) -> sắp giảm dần theo value, hoà theo key tăng dần."""
    return sorted(items, key=lambda t: (-t[1], str(t[0])))


def _bang_nhau_dau(r, tol=1e-9):
    return len(r) >= 2 and abs(r[0][1] - r[1][1]) <= tol


def _d(ban):
    return pd.to_datetime(ban["ngay"], errors="coerce").dt.normalize()


def _inp(ban, p):
    d = _d(ban)
    return (d >= pd.Timestamp(p.start).normalize()) & (d <= pd.Timestamp(p.end).normalize())


def _co_topping_cot(biz):
    for t, df in getattr(biz, "tables", {}).items():
        if any("topping" in fold(str(c)) for c in df.columns):
            return True
    return False


def _la_suaxe(biz):
    return any(re.search(r"sua ?xe|tho", fold(str(t)).replace("_", " ")) for t in getattr(biz, "tables", {}))


def _tra_loi_xep(r, n, ten, loai, mo_ta, tie_none=False):
    """r đã sắp xếp; n = số lượng phần tử hỏi."""
    if not r:
        return None
    if n is None or n == 1:
        if tie_none and _bang_nhau_dau(r):
            return None
        a = ten(r[0][0])
        return ({"type": "string", "answer": a} if n is None else {"type": "list", "answer": [a]}), mo_ta
    if len(r) < n:
        return None
    if tie_none and any(abs(r[i][1] - r[i + 1][1]) <= 1e-9 for i in range(n - 1 if len(r) == n else n)):
        return None
    return {"type": "list", "answer": [ten(k) for k, _ in r[:n]]}, mo_ta + f" (top {n})"


_NHIEU_TUONG = r"\bnhieu\b|\bcao\b"


def solve(q, f, periods, biz, rulebook):
    if rulebook is None:
        return None
    f = re.sub(r"\bnhut\b", "nhat", re.sub(r"\s+", " ", f).strip())
    if len(periods) != 1 or periods[0].kind == "snapshot" or periods[0].start is None:
        return None
    p = periods[0]
    ban = getattr(biz, "ban", None)
    nh = getattr(biz, "nhap", None)
    n = _N(f)
    if n is None and re.search(r"\btop\b|(?<!trong )(?<!ca )\b(nhung|cac) (khach|mat hang|san pham|mon|nhan vien|tho|nha cung cap|ncc)\b", f):
        return None
    tn = r"(?:\btop \d+\b|\b\d+ (?:khach|mat hang|san pham|nhan vien|tho|ncc|nha cung cap)\b)"

    # ======================= NHẬP HÀNG
    if re.search(r"\bnhap\b|nha cung cap|\bncc\b|phieu nhap", f) and nh is not None and ban is not None and len(ban):
        return _nhap(f, p, nh, ban, rulebook, n)

    if ban is None or not len(ban) or "tinh_dt" not in ban or "ngay" not in ban or "ma_don" not in ban:
        return None
    if ban["ma_don"].isna().any():
        return None
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    allb = ban[_inp(ban, p)]
    b = allb[allb["tinh_dt"]]
    if not len(b):
        return None
    dmin, dmax = _d(ban).min(), _d(ban).max()

    # ======================= KHÁCH TRUNG THÀNH / SỘP / ĐỔI Ý
    has_k = "khach" in ban and ban["khach"].notna().any()
    kw = None
    if re.search(r"\btrung thanh\b", f):
        kw = "tt"
    elif re.search(r"\bsop\b", f):
        kw = "sop"
    elif re.search(r"\bdoi y\b", f):
        kw = "doi"
    if kw:
        if not has_k or not _co(rulebook, "khach", kw) or not re.search(r"\bnhat\b", f):
            return None
        intent = {"tt": r"trung thanh", "sop": r"\bsop\b", "doi": r"hay doi y|doi y"}[kw]
        if not _sach(f, [intent, tn, r"\bkhach (hang)?\b"]):
            return None
        if re.search(r"mat hang|san pham|\bmon\b", f):
            return None
        ten = lambda k: _ten_khach(ban, k)
        if kw == "tt":
            x = b[b["khach"].notna()]
            s = x.groupby("khach")["ngay"].nunique()
            mt = "Khách có nhiều NGÀY có mua nhất trong kỳ (đơn tính doanh thu); hoà: mã tăng dần"
        elif kw == "sop":
            x = b[b["khach"].notna()]
            g = x.groupby("khach").agg(tien=("thanh_tien", "sum"), don=("ma_don", "nunique"))
            g = g[g["don"] >= 3]
            s = g["tien"] / g["don"]
            r0 = _rank(list(s.items()))
            if len(r0) >= 2 and _tien(r0[0][1]) == _tien(r0[1][1]) and r0[0][1] != r0[1][1]:
                return None  # xếp theo số chưa tròn hay đã tròn đồng không rõ
            mt = "Khách có giá trị trung bình một đơn cao nhất, chỉ khách >= 3 đơn; hoà: mã tăng dần"
        else:
            x = allb[(allb["trang_thai"] == "huỷ") & allb["khach"].notna()]
            s = x.groupby("khach")["ma_don"].nunique()
            mt = "Khách có nhiều đơn huỷ nhất trong kỳ; hoà: mã tăng dần"
        return _tra_loi_xep(_rank([(k, float(v)) for k, v in s.items()]), n, ten, "string", mt)

    # ======================= MẶT HÀNG ƯA CHUỘNG
    if re.search(r"\bua chuong\b", f):
        if not _co(rulebook, "uc", "phi") or not re.search(r"\bnhat\b", f) or "ten_hang" not in ban or _co_topping_cot(biz):
            return None
        if not _sach(f, [r"\bua chuong\b", tn, r"\b(mat hang|san pham|mon|hang)\b"]):
            return None
        x = b[~b["la_phi"]] if "la_phi" in b else b
        if x["ten_hang"].isna().any():
            return None
        key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if "ma_hang" in x else x["ten_hang"]
        x = x.assign(_k=key.astype(str))
        g = x.groupby("_k").agg(c=("ma_don", "nunique"), ten=("ten_hang", "first"))
        names = g["ten"].to_dict()
        return _tra_loi_xep(_rank([(k, float(v)) for k, v in g["c"].items()]), n, lambda k: str(names[k]), "string",
                            "Mặt hàng có mặt trong nhiều ĐƠN nhất; hoà: mã tăng dần")

    # ======================= NHÂN VIÊN / THỢ CHĂM CHỈ
    if re.search(r"\bcham chi\b", f):
        if not _co(rulebook, "cc") or not re.search(r"\bnhat\b", f) or _la_suaxe(biz):
            return None
        if not _sach(f, [r"\bcham chi\b", tn, r"\b(nhan vien|nguoi|tho)\b"]):
            return None
        col = "nhan_vien" if "nhan_vien" in b and b["nhan_vien"].notna().any() else None
        if col is None or b["nhan_vien"].isna().any():
            return None
        s = b.groupby("nhan_vien")["ngay"].nunique()
        tn_map = {}
        if "ten_nhan_vien" in b and b["ten_nhan_vien"].notna().any():
            tn_map = b.dropna(subset=["ten_nhan_vien"]).drop_duplicates("nhan_vien").set_index("nhan_vien")["ten_nhan_vien"].to_dict()
        return _tra_loi_xep(_rank([(k, float(v)) for k, v in s.items()]), n, lambda k: str(tn_map.get(k, k)), "string",
                            "Nhân viên có nhiều ngày làm việc (ngày có đơn tính doanh thu) nhất; hoà: mã tăng dần")

    # ======================= NGÀY ĐẮT KHÁCH NHẤT
    if re.search(r"\bdat khach\b", f):
        if not _co(rulebook, "ngay") or not re.search(r"\bnhat\b", f) or n is not None:
            return None
        if not _sach(f, [r"\bdat khach\b", r"\bngay (nao)?\b"], extra=("nao",)):
            return None
        s = b.groupby(_d(b))["ma_don"].nunique()
        r = _rank([(k, float(v)) for k, v in s.items()])
        if _bang_nhau_dau(r):
            return None
        return {"type": "date", "answer": r[0][0].strftime("%Y-%m-%d")}, "Ngày có nhiều đơn nhất trong kỳ"

    # ======================= GIỜ CAO ĐIỂM
    if re.search(r"\bcao diem\b", f):
        if not _co(rulebook, "gio") or n is not None:
            return None
        if not _sach(f, [r"\bgio cao diem\b", r"\bcao diem\b", r"\bgio\b", r"\bmay\b", r"\btrong ngay\b"]):
            return None
        if "thoi_diem" not in b or b["thoi_diem"].isna().mean() > 0.01:
            return None
        t = pd.to_datetime(b["thoi_diem"], errors="coerce")
        if t.dt.hour.nunique() < 4:
            return None
        tt = t.groupby(b["ma_don"]).min().dropna()
        s = tt.dt.hour.value_counts()
        r = _rank([(int(k), float(v)) for k, v in s.items()])
        if _bang_nhau_dau(r):
            return None
        return {"type": "number", "answer": int(r[0][0])}, "Giờ bắt đầu khung 1 giờ có nhiều đơn nhất (giờ tạo đơn)"

    # ======================= TRUNG BÌNH MỖI NGÀY THỨ X
    m = re.search(r"\bmoi (?:ngay )?(thu (hai|ba|tu|nam|sau|bay|[2-7])|chu nhat|cn)\b", f)
    if m and _co(rulebook, "tb", "dt"):
        wk = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tu": 2, "4": 2, "nam": 3, "5": 3, "sau": 4, "6": 4, "bay": 5, "7": 5}
        w = 6 if m[1] in ("chu nhat", "cn") else wk[m[2]]
        if p.kind not in ("month", "quarter", "week", "year", "range", "month_to_date", "year_to_date", "last_n_days"):
            return None
        if st < dmin or en > dmax:
            return None
        sd = int((pd.date_range(st, en).weekday == w).sum())
        if sd == 0:
            return None
        rest = f.replace(m[0], " ")
        if re.search(r"doanh thu|dthu|tien ban|tien hang|ban (duoc )?(bao nhieu )?tien|thu ve", rest) and not re.search(r"don|bill|hoa don|luot", rest):
            if not _sach(rest, [r"doanh thu|dthu|tien ban|tien hang|thu ve|trung binh|\btb\b|ban duoc|tien"]):
                return None
            tong = b[_d(b).dt.weekday == w]["thanh_tien"].sum()
            return {"type": "number", "answer": _tien(tong / sd)}, "Trung bình mỗi ngày thứ X = doanh thu các ngày thứ X / số ngày thứ X trong kỳ"
        if re.search(r"\b(don|bill|hoa don)\b", rest) and not re.search(r"doanh thu|dthu|tien|gia tri", rest):
            if not _sach(rest, [r"\b(don|bill|hoa don)\b", r"trung binh|\btb\b|ban duoc|\bhang\b"]):
                return None
            c = b[_d(b).dt.weekday == w]["ma_don"].nunique()
            return {"type": "number", "answer": _r2(c / sd)}, "Trung bình mỗi ngày thứ X = số đơn các ngày thứ X / số ngày thứ X trong kỳ"
    return None


def _ten_khach(ban, k):
    if "ten_khach" in ban:
        s = ban.loc[ban["khach"] == k, "ten_khach"].dropna()
        if len(s):
            return str(s.iloc[0])
    return str(k)


# ----------------------------------------------------------------------------- NHẬP HÀNG
def _nhap(f, p, nh, ban, rulebook, n):
    need = {"ma_phieu", "ngay", "tinh", "thanh_tien"}
    if not need <= set(nh.columns):
        return None
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    if nh["ngay"].isna().mean() > 0.01 or not len(nh):
        return None
    dmin, dmax = nh["ngay"].min(), nh["ngay"].max()
    if en < dmin or st > dmax:
        return None
    nx = nh[nh["tinh"] & (nh["ngay"] >= st) & (nh["ngay"] <= en)]
    if not len(nx):
        return None
    co_ncc = "ncc" in nh and nx["ncc"].notna().all()
    kcol = "ncc_ma" if "ncc_ma" in nx and nx["ncc_ma"].notna().all() else "ncc"

    def ten_ncc(k):
        return str(k)

    if re.search(r"nha cung cap|\bncc\b", f):
        # NCC CHÍNH: nhiều phiếu nhất
        if re.search(r"\bchinh\b", f) and not re.search(r"nhap nhieu|nhieu tien|tien nhieu", f):
            if not _co(rulebook, "ncc_chinh") or not co_ncc or n is not None:
                return None
            if not _sach(f, [r"nha cung cap|\bncc\b", r"\bchinh\b"]):
                return None
            g = nx.drop_duplicates(["ma_phieu", kcol]).groupby(kcol)["ma_phieu"].nunique()
            if nx.groupby("ma_phieu")[kcol].nunique().max() > 1:
                return None
            r = _rank([(k, float(v)) for k, v in g.items()])
            name = _ncc_ten(nx, kcol)
            return {"type": "string", "answer": name[r[0][0]]}, "Nhà cung cấp có nhiều PHIẾU nhập nhất (bỏ phiếu huỷ); hoà: mã/tên tăng dần"
        # NCC nhập nhiều tiền nhất
        if re.search(r"nhieu (?:tien )?(?:hang )?nhat|cao nhat|lon nhat|nhieu tien", f) and not re.search(r"\bchinh\b|phieu", f):
            if not _co(rulebook, "ncc_tien") or not co_ncc or n is not None or nx["thanh_tien"].isna().any():
                return None
            if not _sach(f, [r"nha cung cap|\bncc\b", r"nhieu (?:tien )?(?:hang )?nhat|cao nhat|lon nhat|nhieu tien", r"\bnhap\b",
                             r"\btien\b|\bhang\b|\bgia tri\b|\bphieu\b"], extra=("nhap", "tien", "gia", "tri", "hang", "nhieu", "lon", "cao", "cua", "vo", "zo", "vao", "nhut", "nhat", "lon")):
                return None
            g = nx.groupby(kcol)["thanh_tien"].sum().round(6)
            r = _rank([(k, float(v)) for k, v in g.items()])
            name = _ncc_ten(nx, kcol)
            return {"type": "string", "answer": name[r[0][0]]}, "Nhà cung cấp có tiền nhập cao nhất trong kỳ; hoà: mã/tên tăng dần"
        return None
    # TIỀN NHẬP HÀNG
    if re.search(r"\btien\b|gia tri|bao nhieu|bnhiu|nhieu tien", f) and re.search(r"\bnhap\b", f):
        if not _co(rulebook, "nhap"):
            return None
        if re.search(r"\bban\b|\btru\b|\bton (?:kho|cuoi|dau)|so voi|tang|giam|nhieu hon|it hon|binh quan|trung binh|gia nhap|\bsl\b|so luong|\d+ (?:vien|goi|hop|cai|ly)", f):
            return None
        if not _sach(f, [r"\bnhap\b( hang| vao| ve| het)?", r"\bnhap hang\b", r"tien hang nhap", r"tien nhap", r"\btien\b", r"\bgia tri\b",
                         r"\bbao nhieu\b|\bbnhiu\b|\bnhieu\b"],
                     extra=("het", "roi", "tong", "tn", "chi", "ton", "vo", "vao", "zo", "ve", "hang", "tien", "gia", "tri", "nhap", "mua", "tat", "ca", "bo", "ra", "cac", "phieu", "tiem", "quan", "nhieu", "z", "ay", "nhe")):
            return None
        if nx["thanh_tien"].isna().any():
            return None
        v = nx["thanh_tien"].sum()
        if "tong_phieu" in nx and nx["tong_phieu"].notna().any():
            tp = nx.drop_duplicates("ma_phieu")["tong_phieu"]
            if tp.isna().any() or abs(tp.sum() - v) > 1:
                return None
        return {"type": "number", "answer": _tien(v)}, "Tiền nhập hàng = tổng thành tiền các dòng của phiếu nhập không huỷ có ngày phiếu trong kỳ"
    return None


def _ncc_ten(nx, kcol):
    """Tên trả lời cho từng khoá NCC: tên nếu có (cột ncc), khoá là mã thì tên theo mã."""
    if kcol == "ncc":
        return {k: str(k) for k in nx["ncc"].unique()}
    return nx.drop_duplicates(kcol).set_index(kcol)["ncc"].astype(str).to_dict()
