"""Công nợ (§4) mở rộng + tồn kho / lô / hạn dùng (§5). Không dùng LLM, không logic theo id / shop.
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None."""
import os, re, sys, types
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from loader import fold
import cong_thuc
import sales

_LUAT = {
    "no": "cac khoan tra sau ngay mua co ngay tra trong ky",
    "no_con": "con no cong cac don con no",
    "tuoi": "so ngay tinh tu ngay dat",
    "khach": "moi ma khach la mot khach",
    "kho": "ton het hom nay cua tung mat hang nhan gia von mot don vi co ban",
    "lo": "lo con ton co han dung tu d den het ngay d n",
    "lo_gan": "lo co han som nhat",
}


def _co_luat(rulebook, *keys):
    if rulebook is None:
        return True
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook)))
    return all(_LUAT[k] in t for k in keys)


def _nd(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


# ------------------------------------------------------------------ công nợ: bảng O (đơn) + T (các lần trả)
def _get_debt(biz):
    """(O, T) từ biz.no_don/thu_tien; shop gộp đơn+dòng hàng trong một sheet thì dựng lại bằng sales._build_debt."""
    if getattr(biz, "_nt_debt", None) is not None:
        return biz._nt_debt
    O, T = getattr(biz, "no_don", None), getattr(biz, "thu_tien", None)
    if O is None:
        ban = getattr(biz, "ban", None)
        try:
            nguon = ban.attrs.get("nguon", {}) if ban is not None else {}
            if nguon.get("don") is None and nguon.get("dong") is not None:
                old = ban.attrs["nguon"]
                ban.attrs["nguon"] = dict(old, don=old["dong"])
                try:
                    O, T = sales._build_debt(biz)
                finally:
                    ban.attrs["nguon"] = old
        except Exception:
            O, T = None, None
    try:
        biz._nt_debt = (O, T)
    except Exception:
        pass
    return O, T


def _proxy(biz, O, T):
    return types.SimpleNamespace(no_don=O, thu_tien=T, ban=getattr(biz, "ban", None))


def _con_no_theo_don(O, T, X):
    O2 = O[O["ghi_no"] & ~O["huy"]].copy()
    O2 = O2[_nd(O2["ngay"]) <= X]
    ids = set(O2["ma_don"])
    T2 = T[T["ma_don"].isin(ids)]
    da = T2[_nd(T2["ngay_tra"]) <= X].groupby("ma_don")["so_tien"].sum()
    O2["no"] = O2["tong_don"] - O2["ma_don"].map(da).fillna(0)
    O2["tuoi"] = (X - _nd(O2["ngay"])).dt.days
    return O2


def _khach_key(O2):
    if "ma_khach" in O2 and O2["ma_khach"].notna().any():
        key = O2["ma_khach"]
    else:
        key = O2["ten_khach"].map(lambda v: re.sub(r"\s+", " ", fold(str(v))).strip() if pd.notna(v) else np.nan)
    return key.where(~key.astype(str).str.fullmatch(r"(?i)nan|none|kl|khach le|le|vang lai|"), np.nan)


_SO = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def _giai_no_them(f, periods, biz, rulebook):
    if not _co_luat(rulebook, "no", "no_con", "tuoi"):
        return None
    if len(periods) != 1:
        return None
    p = periods[0]
    X = pd.Timestamp(p.end).normalize()
    hoi_con = re.search(r"\bno\b|cong no|chua tra|chiu", f)
    if not hoi_con or re.search(r"no them|chiu them|tra (bot|duoc|dc)|thu (hoi|duoc|dc|ve)|doi duoc|tang|giam|so voi|trung binh|%", f):
        return None
    if p.start is not None and p.kind != "snapshot" and not re.search(r"den (het )?(ngay|hom|nay)|tinh den|toi (ngay|hom|het)|tinh toi|hien (tai|gio|nay)|cuoi ngay|het ngay|tai ngay|\bhet \d", f):
        return None
    O2 = _state(biz, X)
    if O2 is None:
        return None
    pos = O2[O2["no"] > 0]
    nkhach = re.search(r"\btop (\d{1,2})\b", f) or re.search(r"(?<![/\d])\b(\d{1,2}) khach\b", f)
    nchu = re.search(r"\b(hai|ba|bon|nam|sau|bay|tam|chin|muoi) khach\b", f) or re.search(r"\btop (hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", f)
    if nkhach or nchu:
        n = int(nkhach[1]) if nkhach else _SO[nchu[1]]
        if re.search(r"tuoi|qua \d+ ngay|hon \d+ ngay", f):
            return None
        key = _khach_key(pos)
        x = pos[key.notna()].assign(_k=key[key.notna()])
        if not len(x):
            return None
        g = x.groupby("_k").agg(ten=("ten_khach", "first"), no=("no", "sum")).reset_index()
        g = g[g["no"] > 0].sort_values(["no", "_k"], ascending=[False, True]).head(n)
        rows = [[str(r.ten), int(round(r.no))] for r in g.itertuples()]
        return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": rows}}, f"Top {n} khách nợ lớn nhất đến hết {X:%Y-%m-%d}"
    if re.search(r"(bao nhieu|\bmay|\bso) khach|so luong khach", f) and not re.search(r"tien|\bnhieu nhat", f):
        key = _khach_key(pos)
        g = pos[key.notna()].assign(_k=key[key.notna()]).groupby("_k")["no"].sum()
        return {"type": "number", "answer": int((g > 0).sum())}, f"Số khách còn nợ đến hết {X:%Y-%m-%d}"
    mb = re.search(r"(?<!\d)0\D{1,3}7(?!\d)\D{1,40}(?<!\d)8\D{1,3}15(?!\d)\D{1,40}(tren|hon|qua|>) ?15", f)
    if mb and re.search(r"tuoi", f):
        a = pos["tuoi"]
        res = [pos.loc[(a >= 0) & (a <= 7), "no"].sum(), pos.loc[(a >= 8) & (a <= 15), "no"].sum(), pos.loc[a > 15, "no"].sum()]
        return {"type": "list", "answer": [int(round(v)) for v in res]}, "Nợ còn lại theo tuổi nợ 0–7 / 8–15 / >15 ngày (tuổi từ ngày đặt)"
    if re.search(r"\btuoi\b", f) and re.search(r"(?<!\d)0\D{1,3}7(?!\d)", f):
        return None
    mt = re.search(r"(?:tuoi(?: no)?|dat|ke tu ngay dat|no)\s*(?:da )?(?:cach do )?(qua|hon|tren|lau hon) (\d+) ngay", f) \
        or re.search(r"(qua|hon|tren) (\d+) ngay.{0,40}(no|chua tra)", f)
    if mt:
        a = pos["tuoi"]
        return {"type": "number", "answer": int(round(float(pos.loc[a > int(mt[2]), "no"].sum())))}, f"Tổng nợ còn của đơn có tuổi > {mt[2]} ngày"
    if re.search(r"\btuoi\b|qua \d+ ngay|hon \d+ ngay|\d+ ngay", f):
        return None
    if re.search(r"bao nhieu|tong|\bbn\b|nhieu", f):
        return {"type": "number", "answer": int(round(float(pos["no"].sum())))}, f"Tổng nợ còn đến hết {X:%Y-%m-%d}"
    return None


def _giai_no_cu(f, periods, biz, rulebook):
    O, T = _get_debt(biz)
    if O is None or T is None:
        return None
    g = f.replace("doi duoc", "thu duoc").replace("doi lai", "thu hoi")
    return cong_thuc.giai_no(g, periods, _proxy(biz, O, T), rulebook)


def _giai_no(f, periods, biz, rulebook):
    if not re.search(r"\bno\b|cong no|chiu|ghi so|chua tra|chua thu|thieu", f):
        return None
    r = _giai_no_them(f, periods, biz, rulebook)
    if r:
        return r
    g = re.sub(r"tien khach no chua thu|khach no chua thu|no chua thu", "con no", f)
    return _giai_no_cu(g, periods, biz, rulebook)




# ------------------------------------------------------------------ sổ ghi chú nợ dạng chữ (không có bảng các lần trả)
def _tien(s):
    """Các số tiền trong một đoạn ghi chú (đã bỏ dấu): 695k, 298 nghin, 294.500d, 1tr267 ..."""
    out = []
    for m in re.finditer(r"(\d+)\s*tr\s*(\d{1,3})?|(\d[\d.]*)\s*(k|nghin|ngan)\b|(\d[\d.]*)\s*(d|dong)\b", s):
        if m.group(1):
            v = int(m.group(1)) * 1_000_000 + (int(m.group(2).ljust(3, "0")) * 1000 if m.group(2) else 0)
        elif m.group(3):
            v = float(m.group(3).replace(".", "")) * 1000
        else:
            v = float(m.group(5).replace(".", ""))
        out.append(int(round(v)))
    return out


def _parse_note(note, total):
    n = fold(str(note))
    if re.search(r"da tra du|tra het|ck du|tra xong|da thu du|thanh toan du|tra du", n):
        return 0.0
    a = _tien(n)
    if re.search(r"(dua|tra|ck|chuyen khoan|coc) truoc|da dua|da ck", n) and re.search(r"con lai|con thieu|thieu|ghi so|\bno\b", n):
        if len(a) >= 2:
            return float(a[-1])
        if len(a) == 1 and total is not None and pd.notna(total):
            return float(total) - a[0]
        return np.nan
    if a and re.search(r"\bno\b|thieu|ghi so", n):
        return float(a[-1])
    if not a and re.search(r"no ca don|chua tra|ghi no|ghi so", n) and total is not None and pd.notna(total):
        return float(total)
    return np.nan


def _note_no(biz, O):
    ban = getattr(biz, "ban", None)
    if ban is None:
        return None
    hn = ban.attrs.get("nguon", {}).get("dong")
    if hn is None or hn not in biz.tables:
        return None
    H = biz.tables[hn]
    try:
        hr = sales.column_roles(H, biz.col_meaning.get(hn, {}))
        idc = hr["id"]
    except Exception:
        return None
    best = None
    for c in H.columns:
        if c == "_nguon" or H[c].dtype != object:
            continue
        u = H[c].dropna().astype(str).unique()
        if len(u) < 10:
            continue
        k = sum(1 for v in u if re.search(r"da tra du|tra het|ck du|khach ck", fold(v)) and re.search(r"\bno\b|ghi so", fold(v)))
        if k >= 5 and (best is None or k > best[1]):
            best = (c, k)
    if best is None:
        return None
    ids = sales._norm_id(H[idc])
    d = pd.DataFrame({"ma_don": ids.values, "note": H[best[0]].values}).dropna(subset=["ma_don", "note"]).drop_duplicates("ma_don")
    tong = O.set_index("ma_don")["tong_don"]
    d["no"] = [_parse_note(a, tong.get(b)) for a, b in zip(d["note"], d["ma_don"])]
    return d


def _data_end(biz):
    de = getattr(biz, "data_end", None)
    try:
        return pd.Timestamp(de).normalize()
    except Exception:
        return None


def _state(biz, X):
    """DataFrame đơn ghi nợ (không huỷ) với 'no' (còn nợ đến hết ngày X) và 'tuoi'; None nếu không dựng được chắc chắn."""
    O, T = _get_debt(biz)
    if O is None:
        return None
    if T is not None:
        return _con_no_theo_don(O, T, X)
    de = _data_end(biz)
    if de is None or X < de:
        return None
    d = _note_no(biz, O)
    if d is None:
        return None
    O2 = O[O["ghi_no"] & ~O["huy"]].copy()
    O2 = O2[_nd(O2["ngay"]) <= X]
    mp = d.set_index("ma_don")["no"]
    O2["no"] = O2["ma_don"].map(mp)
    if O2["no"].isna().any():
        # đơn ghi nợ không có ghi chú rõ -> không dựng được chắc chắn
        return None
    O2["tuoi"] = (X - _nd(O2["ngay"])).dt.days
    return O2


# ------------------------------------------------------------------ đơn bán chịu theo "ngày phải trả"
def _hom_nay(periods):
    for p in periods:
        if "hom nay" in fold(getattr(p, "label", "")) and p.end is not None:
            return pd.Timestamp(p.end).normalize()
    return None


def _sophaitra(f, periods, biz, rulebook):
    if not re.search(r"ngay phai tra|han phai tra|han thanh toan", f) or not re.search(r"chiu|ghi no|\bno\b", f):
        return None
    if not re.search(r"bao nhieu|\bmay\b|dem|so don", f):
        return None
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban):
        return None
    D = _hom_nay(periods)
    if D is None:
        return None
    # bảng ngày phải trả: cột ngày có tên/ý nghĩa "phải trả", cột mã đơn trùng mã đơn bán
    ma = set(ban["ma_don"].dropna())
    tb = None
    for t, df in biz.tables.items():
        mm = biz.col_meaning.get(t, {})
        dc = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and re.search(r"phai tra|han (thanh toan|no|tra)", fold(c).replace("_", " ") + " " + fold(mm.get(c, "")))]
        if not dc or len(df) > 200000:
            continue
        best = None
        for c in df.columns:
            if c == "_nguon" or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            v = sales._norm_id(df[c])
            fr = v.dropna().isin(ma).mean() if v.notna().any() else 0
            if fr > 0.9 and (best is None or fr > best[1]):
                best = (v, fr)
        if best is not None:
            tb = pd.DataFrame({"ma_don": best[0].values, "han": pd.to_datetime(df[dc[0]], errors="coerce").dt.normalize().values})
            break
    if tb is None:
        return None
    tb = tb.dropna(subset=["ma_don"]).drop_duplicates("ma_don")
    o = ban.dropna(subset=["ma_don"]).groupby("ma_don").agg(ngay=("ngay", "min"), tt=("trang_thai", "first"))
    tb = tb.join(o, on="ma_don")
    tb = tb[(tb["tt"] != "huỷ") & (_nd(tb["ngay"]) <= D)]
    mon = [p for p in periods if p.kind == "month" and p.start is not None]
    if re.search(r"sau hom nay|muon hon hom nay|con sau|sau ngay hom nay", f) and not mon:
        x = tb[tb["han"] > D]
        return {"type": "number", "answer": int(len(x))}, f"Số đơn bán chịu đặt đến {D:%Y-%m-%d} có ngày phải trả sau hôm nay"
    if len(mon) == 1 and re.search(r"trong thang|roi vao thang|thang \d|\bth ?\d", f):
        st, en = pd.Timestamp(mon[0].start).normalize(), pd.Timestamp(mon[0].end).normalize()
        x = tb[(tb["han"] >= st) & (tb["han"] <= en)]
        return {"type": "number", "answer": int(len(x))}, f"Số đơn bán chịu (đặt đến {D:%Y-%m-%d}) có ngày phải trả trong {st:%m/%Y}"
    return None


# ------------------------------------------------------------------ tồn kho cuối ngày cuối dữ liệu x giá vốn danh mục
def _pairs(biz):
    ban = biz.ban
    P = ban[["ma_hang", "ten_hang"]].drop_duplicates()
    return P


def _key_of(df, biz):
    """cột khoá hàng (mã hoặc tên) của df -> (tên cột, 'ma'|'ten')"""
    P = _pairs(biz)
    codes, names = set(P["ma_hang"].dropna()), set(P["ten_hang"].dropna())
    best = None
    for c in df.columns:
        if c == "_nguon" or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        v = df[c].dropna()
        if not len(v):
            continue
        for kind, S in (("ma", codes), ("ten", names)):
            fr = v.isin(S).mean()
            if fr > 0.6 and (best is None or fr > best[2]):
                best = (c, kind, fr)
    return best


def _to_code(keys, kind, biz):
    if kind == "ma":
        return keys
    P = _pairs(biz).dropna(subset=["ma_hang"]).drop_duplicates("ten_hang").set_index("ten_hang")["ma_hang"]
    return keys.map(P)


def _ton_hien_tai(biz):
    """Series mã hàng -> tồn cuối ngày cuối dữ liệu (đơn vị cơ bản) hoặc None."""
    T = biz.tables
    tk = T.get("the_kho")
    if tk is not None and {"MaHangHoa", "TonCuoi", "NgayGioPhatSinh"} <= set(tk.columns):
        x = tk.sort_values(["NgayGioPhatSinh", "MaGiaoDichKho"] if "MaGiaoDichKho" in tk else ["NgayGioPhatSinh"], kind="stable")
        return x.groupby("MaHangHoa")["TonCuoi"].last().astype(float)
    cands = []
    for t, df in T.items():
        if len(df) > 5000:
            continue
        mm = biz.col_meaning.get(t, {})
        for c in df.columns:
            if c == "_nguon" or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            nm = fold(c).replace("_", " ")
            me = fold(mm.get(c, ""))
            if re.search(r"\bton\b|ton ?kho|ton ?cuoi|ton ?thuc", nm) and not re.search(r"nho nhat|lon nhat|toi thieu|toi da|dinh muc|min|max", nm):
                kc = _key_of(df, biz)
                if kc:
                    cands.append((df, c, kc, 1 if re.search(r"cuoi ngay cuoi|co ban", me) else 0))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[3])
    df, c, kc, _ = cands[0]
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=_to_code(df[kc[0]], kc[1], biz).values)
    s = s[s.index.notna()]
    return s.groupby(level=0).last()


def _gia_von_dm(biz):
    cands = []
    for t, df in biz.tables.items():
        if len(df) > 2000:
            continue
        for c in df.columns:
            nm = fold(c).replace("_", " ")
            if c != "_nguon" and re.search(r"gia ?von", nm) and not re.search(r"luc ?ban", nm) and pd.api.types.is_numeric_dtype(df[c]):
                kc = _key_of(df, biz)
                if kc:
                    cands.append((df, c, kc))
    if len(cands) != 1:
        return None
    df, c, kc = cands[0]
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=_to_code(df[kc[0]], kc[1], biz).values)
    s = s[s.index.notna()]
    return s.groupby(level=0).last()


def _gia_tri_ton(f, periods, biz, rulebook):
    if not re.search(r"gia von|von", f) or not re.search(r"\bkho\b|\bton\b|con lai|con du|\bdu\b", f):
        return None
    if re.search(r"\blo\b|\bhan\b|\bhsd\b|\bnhap\b|ban ra|doanh thu|%|trung binh|\bnhom\b", f):
        return None
    if not re.search(r"bao nhieu|dang gia|\bbn\b|tong|nhieu tien", f):
        return None
    if _names_hang(f, biz):
        return None
    if not _co_luat(rulebook, "kho"):
        return None
    if len(periods) != 1 or periods[0].end is None:
        return None
    D = pd.Timestamp(periods[0].end).normalize()
    de = _data_end(biz)
    if de is None or D < de:
        return None
    ton, gv = _ton_hien_tai(biz), _gia_von_dm(biz)
    if ton is None or gv is None:
        return None
    j = pd.concat([ton.rename("ton"), gv.rename("gv")], axis=1)
    j = j[j["ton"].notna() & (j["ton"] > 0)]
    if j["gv"].isna().any():
        return None
    return {"type": "number", "answer": int(round(float((j["ton"] * j["gv"]).sum())))}, "Tồn cuối ngày cuối dữ liệu x giá vốn danh mục (hàng còn tồn)"


def _names_hang(f, biz):
    P = _pairs(biz)
    for n in P["ten_hang"].dropna().unique():
        k = re.sub(r"\s*\(.*?\)\s*$", "", fold(n)).strip()
        if len(k) >= 4 and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", f):
            return True
    return False


# ------------------------------------------------------------------ lô / hạn dùng (họ bảng lo_hang + chi_tiet_don_lo)
def _lots(biz, D):
    """DataFrame lô: ma_hang, han, ton (đơn vị cơ bản còn lại hết ngày D), gia (giá nhập một đơn vị cơ bản). None nếu không có đủ bảng."""
    T = biz.tables
    need = ("lo_hang", "chi_tiet_don_lo", "phieu_nhap", "don_hang")
    if not all(t in T for t in need):
        return None
    L, S, PN, DH = T["lo_hang"], T["chi_tiet_don_lo"], T["phieu_nhap"], T["don_hang"]
    try:
        L = L[["MaLo", "MaHangHoa", "MaPhieuNhap", "HanSuDung", "SLNhapQuyDoi", "DonGiaNhap"]].copy()
        S = S[["MaHoaDon", "MaLo", "SLQuyDoi", "SLTraQuyDoi"]].copy()
        pn = PN[["MaPhieuNhap", "NgayNhapKho", "TinhTrang"]].copy()
        dh = DH[["MaHoaDon", "NgayLap", "TinhTrang"]].copy()
    except Exception:
        return None
    pn_ok = pn[~pn["TinhTrang"].astype(str).str.contains(r"cancel|huy|hủy|huỷ", case=False)]
    pn_ok = pn_ok[_nd(pn_ok["NgayNhapKho"]) <= D]
    L = L[L["MaPhieuNhap"].isin(set(pn_ok["MaPhieuNhap"]))]
    dh = dh[~dh["TinhTrang"].astype(str).str.contains(r"cancel|huy|hủy|huỷ", case=False)].copy()
    dh["nd"] = _nd(dh["NgayLap"])
    S = S.merge(dh[["MaHoaDon", "nd"]], on="MaHoaDon", how="inner")
    S = S[S["nd"] <= D]
    rt = None
    if "tra_hang" in T and {"MaHoaDon", "NgayGioTra"} <= set(T["tra_hang"].columns):
        rt = T["tra_hang"].assign(nr=_nd(T["tra_hang"]["NgayGioTra"])).groupby("MaHoaDon")["nr"].min()
    S["nr"] = S["MaHoaDon"].map(rt) if rt is not None else pd.NaT
    S["nr"] = S["nr"].fillna(S["nd"])
    S["tra"] = np.where(S["nr"] <= D, S["SLTraQuyDoi"], 0)
    g = S.groupby("MaLo").agg(b=("SLQuyDoi", "sum"), r=("tra", "sum"))
    L = L.join(g, on="MaLo").fillna({"b": 0, "r": 0})
    L["ton"] = L["SLNhapQuyDoi"] - L["b"] + L["r"]
    return pd.DataFrame({"ma_hang": L["MaHangHoa"].values, "han": _nd(L["HanSuDung"]).values, "ton": L["ton"].values,
                         "gia": pd.to_numeric(L["DonGiaNhap"], errors="coerce").values, "lo": L["MaLo"].values})


def _ten_hang_map(biz):
    P = _pairs(biz).dropna(subset=["ma_hang"]).drop_duplicates("ma_hang")
    return P.set_index("ma_hang")["ten_hang"]


def _ten_trong_cau(f, biz):
    P = _pairs(biz).dropna(subset=["ma_hang", "ten_hang"]).drop_duplicates("ma_hang")
    hit = [(fold(n), c) for c, n in zip(P["ma_hang"], P["ten_hang"]) if len(fold(n)) >= 4 and re.search(rf"(?<!\w){re.escape(fold(n))}(?!\w)", f)]
    if not hit:
        return None
    hit.sort(key=lambda t: -len(t[0]))
    if len(hit) > 1 and len(hit[0][0]) == len(hit[1][0]):
        return None
    return hit[0][1]


def _giai_lo(f, periods, biz, rulebook):
    if not re.search(r"\blo\b|han (su )?dung|\bhsd\b|het han|can date|cận date|can date|gan het han|\bdate\b", f):
        return None
    if re.search(r"\bno\b|chiu|phai tra|nhap hang|ban ra", f):
        return None
    if getattr(biz, "ban", None) is None or not len(biz.ban):
        return None
    if not _co_luat(rulebook, "lo"):
        return None
    de = _data_end(biz)
    win = [p for p in periods if p.kind in ("range", "next_n_days") and p.start is not None and p.end is not None]
    gan = re.search(r"gan het han|can date|het han som nhat|han som nhat|sap het han nhat|can han nhat|gan nhat", f)
    if gan and not win:
        if not _co_luat(rulebook, "lo_gan"):
            return None
        ma = _ten_trong_cau(f, biz)
        if ma is None:
            return None
        sn = [p for p in periods if p.kind in ("snapshot", "day") and p.end is not None]
        D = pd.Timestamp(sn[0].end).normalize() if len(sn) == 1 else (de if not periods else None)
        if D is None:
            return None
        Lt = _lots(biz, D)
        if Lt is None:
            return None
        x = Lt[(Lt["ma_hang"] == ma) & (Lt["ton"] > 0) & (Lt["han"] >= D)]
        if not len(x):
            return None
        return {"type": "date", "answer": x["han"].min().strftime("%Y-%m-%d")}, "Lô còn tồn chưa quá hạn có hạn dùng sớm nhất của mặt hàng"
    if len(win) != 1 or len(periods) > 2:
        return None
    p = win[0]
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    Lt = _lots(biz, st)
    if Lt is None:
        return None
    x = Lt[(Lt["ton"] > 0) & (Lt["han"] >= st) & (Lt["han"] <= en)]
    if re.search(r"tien von|von\b|dong|dong bao nhieu|tien|gia tri", f):
        if x["gia"].isna().any():
            return None
        return {"type": "number", "answer": int(round(float((x["ton"] * x["gia"]).sum())))}, "Giá trị phần còn tồn của lô (giá nhập của lô) có hạn trong cửa sổ"
    if re.search(r"bao nhieu lo|may lo|\bbn lo\b|so lo|\bbn\b.{0,6}\blo\b", f) or re.search(r"\blo\b.{0,40}(bao nhieu|\bbn\b)", f):
        return {"type": "number", "answer": int(len(x))}, "Số lô còn tồn có hạn trong cửa sổ"
    if re.search(r"\b(sp|san pham|mat hang|hang|thuoc)\b.{0,25}(nao|gi)|ke |liet ke|nhung (sp|san pham|mat hang|hang)", f):
        nm = _ten_hang_map(biz)
        g = x.groupby("ma_hang")["han"].min().reset_index().sort_values(["han", "ma_hang"])
        names = [str(nm.get(c)) for c in g["ma_hang"]]
        if any(n == "None" or n == "nan" for n in names):
            return None
        return {"type": "list", "answer": names}, "Mặt hàng còn lô tồn có hạn trong cửa sổ, xếp theo hạn sớm nhất rồi mã"
    return None


def solve(q, f, periods, biz, rulebook):
    f = re.sub(r"\s+", " ", f).strip()
    for fn in (_sophaitra, _giai_no, _giai_lo, _gia_tri_ton):
        try:
            r = fn(f, periods, biz, rulebook)
        except Exception:
            raise
        if r:
            return r
    return None
