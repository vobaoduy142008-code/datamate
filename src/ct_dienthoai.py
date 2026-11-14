"""Công thức cố định (pandas, không LLM) cho ngành CỬA HÀNG ĐIỆN THOẠI + công nợ gồm cả trả góp.
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.  Không chắc thì None.
Nhận diện shop bằng ĐẶC ĐIỂM BẢNG (cột IMEI + Khách trả + STT ở sổ bán; sổ nợ / sổ góp / bảo hành / thu máy cũ theo tên cột),
không dùng business_id, tên shop, id câu. Chỉ bật khi rulebook có đúng các luật liên quan."""
import math
import re

import numpy as np
import pandas as pd

from loader import fold

try:
    from sales import status_one
except Exception:                                   # pragma: no cover
    def status_one(v):
        t = fold(v) if v is not None else ""
        if re.search(r"\bhuy\b", t):
            return "huỷ"
        if re.search(r"\btra hang|\btra lai|khach tra\b|mang tra", t):
            return "trả hàng"
        return "hoàn thành"


# ------------------------------------------------------------------ tiện ích
def fz(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", fold(s)).strip()


def _cols(df):
    return {fz(c): c for c in df.columns}


def _col(c, *pats):
    for p in pats:
        for k, v in c.items():
            if re.search(p, k):
                return v
    return None


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _dt(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _money(x):
    return int(math.floor(float(x) + 0.5))


def _pct(a, b):
    return round(float(a) / float(b) * 100.0, 2)


_LUAT = {
    "mc": "tien thu may cu khong tru vao doanh thu",
    "tra": "khach phai tra la tien cua don sau bot va sau tru may cu",
    "ky": "chi tinh don khong huy khong tra",
    "gop": "phan gop chua tra la tien khach con no",
    "han": "ky gop co han truoc ngay hoi ma chua gop du la qua han",
    "bh": "luot bao hanh co ngay nhan trong quy",
    "dong": "dong may phan ten hang dung truoc dung luong",
    "no": "cac khoan tra sau ngay mua co ngay tra trong ky",
    "hang": "hang chu dau ten hang",
}


def _co_luat(rulebook, *keys):
    if rulebook is None:
        return True
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook)))
    return all(_LUAT[k] in t for k in keys)


# ------------------------------------------------------------------ dựng dữ liệu
def _dong_may(ten):
    return re.split(r"\s+\d+\s*gb\b", str(ten).strip(), maxsplit=1, flags=re.I)[0].strip()


def _hang_may(ten):
    w = str(ten).strip().split()
    if not w:
        return ""
    return "Apple" if fold(w[0]) == "iphone" else w[0]


def _imeis(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return [d for d in (re.sub(r"\D", "", x) for x in re.split(r"[/,;]", str(v))) if len(d) >= 10]


def _stt_of(v):
    m = re.search(r"(\d+)", str(v)) if v is not None else None
    return int(m[1]) if m else None


def _ckey(name, phone):
    d = re.sub(r"\D", "", str(phone)) if phone is not None and not (isinstance(phone, float) and math.isnan(phone)) else ""
    return d if len(d) >= 8 else fz(name)



# ---- đọc lại bảng phụ từ FILE GỐC: loader đoán sai thứ tự ngày/tháng của các ô ngày mơ hồ ('04/07/2026') ở sổ nợ / góp / bảo hành
import datetime as _dtm
import os as _os

import loader as _ld


def _isnull(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT


def _col_order(vals):
    """'dm' (ngày/tháng, kiểu Việt Nam) trừ khi có bằng chứng tháng/ngày."""
    dm = md = False
    for v in vals:
        if isinstance(v, str):
            m = re.match(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", v)
            if m:
                a, b = int(m[1]), int(m[2])
                dm |= a > 12
                md |= b > 12
    return "md" if (md and not dm) else "dm"


def _pdate(v, order="dm"):
    if _isnull(v):
        return pd.NaT
    try:
        if isinstance(v, (pd.Timestamp, _dtm.datetime, _dtm.date)):
            return pd.Timestamp(v).normalize()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))).normalize() if 20000 <= float(v) <= 80000 else pd.NaT
        t = str(v).strip().lower()
        if re.fullmatch(r"\d{5}(\.\d+)?", t):
            return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(t))).normalize()
        m = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", t)
        if m:
            return pd.Timestamp(int(m[1]), int(m[2]), int(m[3]))
        m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:[ t].*)?$", t)
        if m:
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            y = y + 2000 if y < 100 else y
            d, mo = (a, b) if order == "dm" else (b, a)
            return pd.Timestamp(y, mo, d)
        d1 = _ld._parse_date_value(v, None)[0]
        return pd.Timestamp(d1).normalize() if d1 is not None else pd.NaT
    except Exception:
        return pd.NaT


def _dser(s):
    vals = list(s)
    o = _col_order(vals)
    return pd.Series([_pdate(v, o) for v in vals], index=s.index, dtype="datetime64[ns]")


def _mser(s):
    def one(v):
        if _isnull(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return float(v)
        x = _ld.to_number(v)
        return float(x) if x is not None else np.nan
    return pd.Series([one(v) for v in s], index=s.index, dtype=float)


def _sser(s):
    return pd.Series([None if _isnull(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()) for v in s], index=s.index, dtype=object)


def _raw_of(biz, tname, cache):
    """Bảng thô (chưa ép kiểu) của bảng `tname`, ghép từ các file|sheet nguồn. None nếu không đọc được."""
    path = getattr(biz, "path", None)
    srcs = getattr(biz, "sources", {}).get(tname)
    if not path or not srcs or not _os.path.isdir(path):
        return None
    frames = []
    for sr in srcs:
        fname, _, sh = sr.partition("|")
        fp = _os.path.join(path, fname)
        try:
            if fname.lower().endswith(".csv"):
                fr = _ld._frame_from_raw(_ld._read_csv_raw(fp))
            else:
                if fp not in cache:
                    cache[fp] = _ld._read_excel_raw(fp)
                shs = cache[fp]
                key = next((k for k in shs if str(k) == sh), None)
                if key is None:
                    return None
                fr = _ld._frame_from_raw(shs[key])
        except Exception:
            return None
        if not fr.empty:
            frames.append(fr)
    if not frames:
        return None
    base = {fz(c): c for c in frames[0].columns}
    out = []
    for fr in frames:
        m = {fz(c): c for c in fr.columns}
        if set(m) != set(base):
            return None
        out.append(fr[[m[k] for k in base]].rename(columns={m[k]: base[k] for k in base}))
    return pd.concat(out, ignore_index=True)


class _D:
    pass


def _build(biz):
    T = biz.tables
    sb, gt, tm, no, gop, bh = [], None, None, None, None, None
    tn = {}
    for name, df in T.items():
        c = _cols(df)
        ks = set(c)
        if {"ngay", "stt", "hang", "imei", "sl", "thanh tien", "khach tra"} <= ks:
            sb.append(df)
        elif {"ghi tat", "ten hang", "loai"} <= ks:
            gt = df
        elif {"so ky", "da gop", "ngay gop"} <= ks:
            gop, tn["gop"] = df, name
        elif {"no", "da tra", "ngay tra"} <= ks:
            no, tn["no"] = df, name
        elif {"ngay nhan", "imei"} <= ks:
            bh, tn["bh"] = df, name
        elif "gia thu" in ks and any(k.startswith("tru vao") for k in ks):
            tm, tn["tm"] = df, name
    if not sb:
        return None
    D = _D()
    cache = {}
    S = pd.concat(sb, ignore_index=True)
    c = _cols(S)
    R = pd.DataFrame({
        "ngay": _dt(S[c["ngay"]]), "stt": _num(S[c["stt"]]), "khach": S[c["khach"]] if "khach" in c else None,
        "hang": S[c["hang"]], "imei": S[c["imei"]], "sl": _num(S[c["sl"]]), "gia": _num(S[c["gia"]]) if "gia" in c else np.nan,
        "tt": _num(S[c["thanh tien"]]), "bot": _num(S[c["bot"]]) if "bot" in c else np.nan,
        "kt": _num(S[c["khach tra"]]),
        "mc": _num(S[c["thu may cu"]]) if "thu may cu" in c else np.nan,
        "gc": S[c["ghi chu"]] if "ghi chu" in c else None,
    })
    R = R[R["ngay"].notna() & R["stt"].notna()].copy()
    R["mid"] = R["ngay"].dt.strftime("%Y-%m-%d") + "#" + R["stt"].astype(int).astype(str)
    R["tt"] = R["tt"].where(R["tt"].notna(), R["sl"].fillna(1) * R["gia"])
    R["sl"] = R["sl"].fillna(1)
    g = R.groupby("mid", sort=False)
    O = pd.DataFrame({"ngay": g["ngay"].min(), "tt": g["tt"].sum(), "kt": g["kt"].first(), "bot": g["bot"].first().fillna(0),
                      "mc_col": g["mc"].first()})
    O["stt"] = g["stt"].first().astype(int)
    st = g["gc"].apply(lambda s: {status_one(v) for v in s.dropna().unique()})
    O["huy"] = st.map(lambda x: "huỷ" in x)
    O["tra"] = st.map(lambda x: "trả hàng" in x) & ~O["huy"]
    O["tinh"] = ~O["huy"] & ~O["tra"]
    # STT -> mã đơn (chỉ khi STT chỉ thuộc một ngày): để nối sổ phụ kể cả khi ngày ở sổ phụ ghi lệch
    cnt = O.groupby("stt").size()
    D.stt_mid = {int(k): v for k, v in O.reset_index().drop_duplicates("stt").set_index("stt")["mid"].items() if cnt[k] == 1}

    def link(d, stt):
        """mã đơn ở sổ bán của (ngày, STT) ở sổ phụ; None nếu không nối được."""
        if stt is None or (isinstance(stt, float) and math.isnan(stt)):
            return None
        mid = (d.strftime("%Y-%m-%d") + "#" + str(int(stt))) if pd.notna(d) else None
        if mid in O.index:
            return mid
        return D.stt_mid.get(int(stt))

    # thu máy cũ: cột ở sổ bán, hoặc sổ riêng (Giá thu, Trừ vào STT)
    mc = pd.Series(0.0, index=O.index)
    D.raw_ok = True
    if O["mc_col"].notna().any():
        mc = O["mc_col"].fillna(0)
    elif tm is not None:
        rt = _raw_of(biz, tn["tm"], cache)
        if rt is None:
            D.raw_ok = False
        else:
            ct = _cols(rt)
            tv = _col(ct, r"^tru vao")
            if "ngay" in ct and tv:
                dts, stts, vals = _dser(rt[ct["ngay"]]), _num(rt[tv]), _mser(rt[ct["gia thu"]])
                acc = {}
                for d, sn, v in zip(dts, stts, vals):
                    mid = link(d, sn)
                    if mid is not None and pd.notna(v):
                        acc[mid] = acc.get(mid, 0.0) + v
                mc = pd.Series(acc).reindex(O.index).fillna(0.0)
    O["mc"] = mc
    O["kt"] = O["kt"].where(O["kt"].notna(), O["tt"] - O["bot"] - O["mc"])
    D.O = O
    D.dmin, D.dmax = O["ngay"].min(), O["ngay"].max()
    # bảng ghi tắt
    amap = {}
    if gt is not None:
        cg = _cols(gt)
        for _, r in gt.iterrows():
            ten = str(r[cg["ten hang"]]).strip()
            loai = fz(r[cg["loai"]])
            amap[fz(r[cg["ghi tat"]])] = (ten, loai)
            amap[fz(ten)] = (ten, loai)
    D.amap = amap
    rows = R[R["hang"].notna()]
    ten, loai = [], []
    for h, im in zip(rows["hang"], rows["imei"]):
        k = fz(h)
        if k in amap:
            t, l = amap[k]
        else:
            has = bool(_imeis(im))
            t, l = str(h).strip(), ("may cu" if re.search(r"\bcu\b", k) else "may moi") if has else "phu kien"
        ten.append(t)
        loai.append(l)
    L = pd.DataFrame({"mid": rows["mid"].values, "ngay": rows["ngay"].values, "ten": ten, "loai": loai,
                      "sl": rows["sl"].values, "tt": rows["tt"].values, "imei": [_imeis(x) for x in rows["imei"]]})
    L["tinh"] = L["mid"].map(O["tinh"]).fillna(False).astype(bool)
    L["phone"] = L["loai"].isin(["may moi", "may cu"])
    L["dong"] = [_dong_may(t) for t in L["ten"]]
    L["hangm"] = [_hang_may(t) for t in L["ten"]]
    D.L = L
    # bảo hành (đọc từ file gốc)
    D.B = None
    if bh is not None:
        rb = _raw_of(biz, tn["bh"], cache)
        if rb is None:
            D.raw_ok = False
        else:
            cb = _cols(rb)
            B = pd.DataFrame({"ngay": _dser(rb[cb["ngay nhan"]]), "imei": [(_imeis(_sser(pd.Series([x]))[0]) or [None])[0] for x in rb[cb["imei"]]],
                              "may": rb[cb["may"]] if "may" in cb else None})
            D.B = B[B["ngay"].notna()].copy()
    # sổ nợ (đọc từ file gốc; ô trống ở dòng sau của cùng đơn = giống dòng đầu)
    D.no = []
    if no is not None:
        rn = _raw_of(biz, tn["no"], cache)
        if rn is None:
            D.raw_ok = False
        else:
            cn = _cols(rn)
            n = pd.DataFrame({"ngay": _dser(rn[cn["ngay"]]), "khach": _sser(rn[cn["khach"]]) if "khach" in cn else None,
                              "sdt": _sser(rn[cn["sdt"]]) if "sdt" in cn else None, "no": _mser(rn[cn["no"]]),
                              "tra": _mser(rn[cn["da tra"]]), "nt": _dser(rn[cn["ngay tra"]]),
                              "stt": [_stt_of(v) for v in rn[cn["ghi chu"]]] if "ghi chu" in cn else None})
            for col in ("ngay", "khach", "sdt", "no"):
                n[col] = n[col].ffill()
            n = n[n["stt"].notna() & n["ngay"].notna()].copy()
            n["mid"] = [link(d, s_) for d, s_ in zip(n["ngay"], n["stt"])]
            n["ord"] = n["mid"].where(n["mid"].notna(), n["ngay"].dt.strftime("%Y-%m-%d") + "#" + n["stt"].astype(int).astype(str))
            for mid, x in n.groupby("ord", sort=False):
                r0 = x.iloc[0]
                pays = [(a, b) for a, b in zip(x["nt"], x["tra"]) if pd.notna(a) and pd.notna(b)]
                known = mid in O.index
                D.no.append(dict(mid=mid, ngay=O.at[mid, "ngay"] if known else r0["ngay"], n0=float(r0["no"]), pays=pays,
                                 name=r0["khach"], key=_ckey(r0["khach"], r0["sdt"]), huy=bool(O.at[mid, "huy"]) if known else False,
                                 known=known))
    # sổ góp
    D.gop = []
    if gop is not None:
        rg = _raw_of(biz, tn["gop"], cache)
        if rg is None:
            D.raw_ok = False
        else:
            cg = _cols(rg)
            hc = _col(cg, r"^han gop")
            gdf = pd.DataFrame({"ngay": _dser(rg[cg["ngay"]]), "stt": _num(rg[cg["stt"]]) if "stt" in cg else np.nan,
                                "khach": _sser(rg[cg["khach"]]) if "khach" in cg else None, "sdt": _sser(rg[cg["sdt"]]) if "sdt" in cg else None,
                                "gia": _mser(rg[cg["gia"]]), "tt": _mser(rg[cg["tra truoc"]]), "k": _num(rg[cg["so ky"]]),
                                "gm": _mser(rg[_col(cg, r"^gop thang")]), "h": _num(rg[hc]) if hc else np.nan,
                                "ng": _dser(rg[cg["ngay gop"]]), "dg": _mser(rg[cg["da gop"]])})
            for col in ("ngay", "khach", "sdt", "gia", "tt", "k", "gm", "h"):
                gdf[col] = gdf[col].ffill()
            gdf = gdf[gdf["stt"].notna() & gdf["ngay"].notna()].copy()
            gdf["mid"] = [link(d, s_) for d, s_ in zip(gdf["ngay"], gdf["stt"])]
            gdf["ord"] = gdf["mid"].where(gdf["mid"].notna(), gdf["ngay"].dt.strftime("%Y-%m-%d") + "#" + gdf["stt"].astype(int).astype(str))
            for mid, x in gdf.groupby("ord", sort=False):
                r0 = x.iloc[0]
                n_, gm, rest = int(r0["k"]), float(r0["gm"]), float(r0["gia"] - r0["tt"])
                if n_ <= 0 or not np.isfinite(r0["h"]):
                    D.gop = None
                    break
                known = mid in O.index
                nd = O.at[mid, "ngay"] if known else r0["ngay"]
                amts = [gm] * (n_ - 1) + [rest - gm * (n_ - 1) if rest - gm * (n_ - 1) > 0 else gm]
                dues = []
                for i in range(1, n_ + 1):
                    m = nd.to_period("M") + i
                    dues.append(pd.Timestamp(m.year, m.month, min(int(r0["h"]), m.days_in_month)))
                pays = [(a, b) for a, b in zip(x["ng"], x["dg"]) if pd.notna(a) and pd.notna(b)]
                D.gop.append(dict(mid=mid, ngay=nd, gia=float(r0["gia"]), tt=float(r0["tt"]), rest=rest, n=n_, amts=amts, dues=dues,
                                  pays=pays, name=r0["khach"], key=_ckey(r0["khach"], r0["sdt"]),
                                  huy=bool(O.at[mid, "huy"]) if known else False, known=known))
    D.has_no = no is not None and D.raw_ok
    D.has_gop = gop is not None and D.gop is not None and D.raw_ok
    D.has_bh = D.B is not None and D.raw_ok
    D.has_mc = bool((O["mc"] > 0).any()) and D.raw_ok
    return D


def _get(biz):
    D = getattr(biz, "_dt_cache", "none")
    if D == "none":
        try:
            D = _build(biz)
        except Exception:
            D = None
        try:
            biz._dt_cache = D
        except Exception:
            pass
    return D


# ------------------------------------------------------------------ trạng thái nợ / góp
def _paid(pays, X, after=None):
    return float(sum(b for a, b in pays if a <= X and (after is None or a >= after)))


def _no_rest(d, X):
    if d["huy"] or d["ngay"] > X:
        return None
    return max(0.0, d["n0"] - _paid(d["pays"], X))


def _gop_rest(c, X):
    if c["huy"] or c["ngay"] > X:
        return None
    return max(0.0, c["rest"] - _paid(c["pays"], X))


def _gop_over(c, X):
    """(số tiền quá hạn, số kỳ quá hạn) tại ngày X: kỳ có hạn TRƯỚC X mà chưa góp đủ (góp cộng dồn theo ngày)."""
    if c["huy"] or c["ngay"] > X:
        return 0.0, 0
    paid = _paid(c["pays"], X)
    cum, amt, cnt = 0.0, 0.0, 0
    for d, a in zip(c["dues"], c["amts"]):
        if d < X:
            cum += a
            if cum > paid + 0.5:
                cnt += 1
    due = sum(a for d, a in zip(c["dues"], c["amts"]) if d < X)
    return max(0.0, due - paid), cnt


# ------------------------------------------------------------------ nhận câu hỏi
_STOP_KHACH = r"(hang|con|kon|chua|tra|no|qua|phai|co|nao|dang|thieu|mua|dem|duoc|moi|cu|gop|nao|nhieu|cham|nay|le|vang|mang|toi|den|do|ma|ban|va|hay)"


def _find_names(f, ents):
    t = " " + fz(f) + " "
    hits = {}
    for e in ents:
        nm = fz(e["name"])
        if len(nm.split()) >= 2 and len(nm) >= 5 and (" " + nm + " ") in t:
            hits.setdefault(nm, set()).add(e["key"])
    hits = {k: v for k, v in hits.items() if not any(k != o and k in o for o in hits)}
    return hits


def _ngay_cuoi(periods):
    """Ngày X cho câu 'đến hết ngày X' / hôm nay."""
    if not periods:
        return None
    for p in periods:
        if p.kind == "snapshot":
            return pd.Timestamp(p.end)
    if len(periods) == 1 and periods[0].kind in ("day", "range", "month_to_date", "year_to_date"):
        return pd.Timestamp(periods[0].end)
    return None


def _debt(q, f, periods, D, rulebook):
    if not (D.has_no or D.has_gop):
        return None
    gop_w = re.search(r"\bgop\b|tra gop|ky gop", f)
    no_w = re.search(r"\bno\b|cong no|tuoi no|mua chiu|ghi no|chiu\b", f)
    if not (gop_w or no_w):
        return None
    if re.search(r"doanh thu|dthu|\btop\b|nhieu nhat|it nhat|lon nhat|cao nhat|trung binh|%|phan tram|ty le|ti le|tien mat|chuyen khoan|\bca \d|\bdon\b.{0,12}(bao nhieu|may)|(bao nhieu|may) don", f):
        return None
    if gop_w and not D.has_gop:
        return None
    if re.search(r"\bno\b", f) and not D.has_no and not D.has_gop:
        return None
    ents_all = [dict(e, kind="no") for e in D.no] + [dict(e, kind="gop") for e in (D.gop or [])]
    names = _find_names(f, ents_all)
    if len(names) > 1 or any(len(v) > 1 for v in names.values()):
        return None
    sel = None
    if names:
        k = next(iter(names))
        sel = next(iter(names[k]))
    elif re.search(rf"\bkhach (?!{_STOP_KHACH}\b)\w+ \w+", f):
        return None                                  # nhắc một khách cụ thể mà không tìm ra tên -> không chắc
    NO = [e for e in D.no if sel is None or e["key"] == sel]
    CG = [e for e in (D.gop or []) if sel is None or e["key"] == sel]
    ask_cust = re.search(r"\b(may|mi|bao nhieu|so|nhieu) (khach|ngui|nguoi)\b", f)
    ask_ky = re.search(r"\b(may|bao nhieu|so) ky\b", f)
    X = _ngay_cuoi(periods)
    p1 = periods[0] if len(periods) == 1 else None
    qua = re.search(r"qua han|cham gop|cham tra|tre han", f)
    toi = re.search(r"toi han|den han|da toi han", f)
    cu_toi = re.search(r"chua toi han|chua den han", f)

    # 1) tuổi nợ theo nhóm
    if re.search(r"tuoi no", f) and X is not None:
        bk = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", f)]
        tr = re.findall(r"(?:tren|hon|qua) (\d+) ngay", f)
        if len(bk) == 2 and len(tr) == 1 and not names and re.search(r"moi nhom|tung nhom|nhom", f):
            tot = [0.0, 0.0, 0.0]
            items = []
            for d in D.no:
                r = _no_rest(d, X)
                if r and r > 0:
                    items.append((d["ngay"], r))
            for c in (D.gop or []):
                r = _gop_rest(c, X)
                if r and r > 0:
                    items.append((c["ngay"], r))
            for ng, r in items:
                age = (X - ng).days
                if bk[0][0] <= age <= bk[0][1]:
                    tot[0] += r
                elif bk[1][0] <= age <= bk[1][1]:
                    tot[1] += r
                elif age > int(tr[0]):
                    tot[2] += r
            return {"type": "list", "answer": [_money(v) for v in tot]}, "Tuổi nợ (từ ngày đặt) chia nhóm: nợ + phần góp chưa trả đến hết ngày X"
        return None

    # 2) trả góp
    if gop_w:
        if not _co_luat(rulebook, "gop"):
            return None
        if re.search(r"(tien gop|gop) (thu|duoc|da thu)|thu (duoc |ve )?(tien )?gop|khach gop (duoc )?(bao nhieu|tong)", f) \
                and not re.search(r"chua|con |qua han|toi han|cham", f):
            if p1 is None or p1.start is None or p1.kind == "snapshot" or sel is not None or not _co_luat(rulebook, "no"):
                return None
            s_, e_ = pd.Timestamp(p1.start), pd.Timestamp(p1.end)
            tot = sum(b for c in (D.gop or []) if not c["huy"] for a, b in c["pays"] if s_ <= a <= e_ and a > c["ngay"])
            return {"type": "number", "answer": _money(tot)}, "Tiền góp thu trong kỳ: các lần góp có ngày góp trong kỳ (sau ngày mua)"
        if X is None:
            return None
        # số kỳ đã tới hạn của một khách
        if toi and (re.search(r"may ky|bao nhieu ky|so ky", f)) and sel is not None and not qua:
            if len(CG) != 1 or re.search(r"khong (tinh|ke)", f):
                return None
            c = CG[0]
            if c["huy"] or c["ngay"] > X:
                return None
            if re.search(r"khong (tinh|ke)", f):
                return None
            if not re.search(r"(tinh|ke) (ca|luon)", f):
                return None                           # chưa nói rõ có tính kỳ hạn đúng ngày hỏi hay không
            return {"type": "number", "answer": int(sum(1 for d in c["dues"] if d <= X))}, "Số kỳ góp có hạn <= ngày X (tính cả kỳ hạn đúng ngày X)"
        if qua:
            if not _co_luat(rulebook, "han"):
                return None
            over = [(c, _gop_over(c, X)) for c in CG]
            if ask_ky:
                return {"type": "number", "answer": int(sum(o[1] for _, o in over))}, "Số kỳ góp quá hạn (hạn trước ngày X, chưa góp đủ)"
            if re.search(r"\btien\b", f) and not ask_cust:
                return {"type": "number", "answer": _money(sum(o[0] for _, o in over))}, "Tiền kỳ góp quá hạn chưa góp (hạn trước ngày X)"
            if ask_cust or (not re.search(r"\btien\b", f) and sel is None):
                keys = {c["key"] for c, o in over if o[0] > 0.5}
                if re.search(r"hop dong", f):
                    return {"type": "number", "answer": int(sum(1 for c, o in over if o[0] > 0.5))}, "Số hợp đồng góp có kỳ quá hạn"
                return {"type": "number", "answer": len(keys)}, "Số khách có kỳ góp quá hạn (hạn trước ngày X, chưa góp đủ)"
            return None
        if re.search(r"chua gop|con thieu|con lai|chua tra|con no|kon no", f) and (cu_toi or re.search(r"ke ca|tinh ca|bao gom", f)):
            return {"type": "number", "answer": _money(sum(r for r in (_gop_rest(c, X) for c in CG) if r))}, \
                "Tiền góp còn lại chưa góp (gồm cả kỳ chưa tới hạn) đến hết ngày X"
        return None

    # 3) công nợ nói chung
    if not _co_luat(rulebook, "gop", "no"):
        return None
    if re.search(r"thu no|tra no|thu hoi no|tra bot|no them|ghi no them|mua chiu them|chiu them|\bthu\b.{0,20}\bno\b", f):
        if p1 is None or p1.start is None or p1.kind == "snapshot" or sel is not None:
            return None
        s, e = pd.Timestamp(p1.start), pd.Timestamp(p1.end)
        thu = re.search(r"thu no|tra no|thu hoi no|tra bot|\bthu\b.{0,20}\bno\b", f)
        them = re.search(r"no them|ghi no them|mua chiu them|chiu them", f)
        if thu and them:
            return None
        if thu:
            tot = 0.0
            for d in D.no:
                if not d["huy"]:
                    tot += sum(b for a, b in d["pays"] if s <= a <= e and a > d["ngay"])
            for c in (D.gop or []):
                if not c["huy"]:
                    tot += sum(b for a, b in c["pays"] if s <= a <= e and a > c["ngay"])
            return {"type": "number", "answer": _money(tot)}, "Thu nợ trong kỳ: các khoản trả SAU ngày mua (nợ + tiền góp), ngày trả trong kỳ"
        tot = sum(d["n0"] for d in D.no if not d["huy"] and s <= d["ngay"] <= e) \
            + sum(c["rest"] for c in (D.gop or []) if not c["huy"] and s <= c["ngay"] <= e)
        return {"type": "number", "answer": _money(tot)}, "Nợ thêm trong kỳ: phần chưa trả ngay lúc mua của đơn ghi nợ + đơn trả góp đặt trong kỳ"
    if X is None:
        return None
    if re.search(r"\bno\b|cong no|chiu", f) and not re.search(r"thu no|tra no", f):
        keys = {}
        for d in NO:
            r = _no_rest(d, X)
            if r and r > 0:
                keys[d["key"]] = keys.get(d["key"], 0) + r
        for c in CG:
            r = _gop_rest(c, X)
            if r and r > 0:
                keys[c["key"]] = keys.get(c["key"], 0) + r
        if ask_cust:
            return {"type": "number", "answer": len(keys)}, "Số khách còn nợ (nợ + phần góp chưa trả) đến hết ngày X"
        if re.search(r"\btien\b|bao nhieu|tong", f) and not re.search(r"khach nao|\bai\b", f):
            return {"type": "number", "answer": _money(sum(keys.values()))}, "Tổng còn nợ (nợ + phần góp chưa trả, gồm kỳ chưa tới hạn) đến hết ngày X"
    return None


# ------------------------------------------------------------------ bán hàng, thu máy cũ, bảo hành
def _in(s, p):
    return (s >= pd.Timestamp(p.start)) & (s <= pd.Timestamp(p.end))


def _loai_filter(f):
    m, c = re.search(r"may moi|dien thoai moi|\bdt moi", f), re.search(r"may cu|dien thoai cu|\bdt cu", f)
    if m and c:
        return None
    return "may moi" if m else ("may cu" if c else None)


def _match_target(f, D):
    """Tên mặt hàng / dòng máy nêu trong câu: -> (cột, giá trị, nhãn) hoặc None. Lấy khớp DÀI NHẤT."""
    t = " " + fz(f).replace(" coc sac", " cu sac").replace(" ke dien thoai", " gia do dien thoai") + " "
    best = None
    L = D.L
    cand = {}
    for ten in L["ten"].unique():
        cand.setdefault(fz(ten), ("ten", ten))
        dm = _dong_may(ten)
        cand.setdefault(fz(dm), ("dong", dm))
    for k, v in cand.items():
        if len(k) >= 4 and (" " + k + " ") in t:
            if best is None or len(k) > len(best[0]):
                best = (k, v)
    if not best:
        return None
    rest = t[t.find(" " + best[0] + " ") + len(best[0]) + 2:].split()
    if rest and (rest[0] in {"pro", "max", "plus", "ultra", "mini", "fe", "lite", "se", "5g", "4g", "flip", "fold", "note", "gb"} or re.fullmatch(r"\d+gb", rest[0])) \
            and best[1][0] == "dong":
        return False                                  # tên máy trong câu dài hơn tên dòng máy có trong dữ liệu -> không chắc
    return best[1]


def _hang_trong_cau(f, D):
    t = " " + fz(f) + " "
    ph = D.L[D.L["phone"]]
    hs = {}
    for h in ph["hangm"].unique():
        hs[fz(h)] = h
    out = []
    for k, h in hs.items():
        if (" " + k + " ") in t or (h == "Apple" and (" apple " in t or " iphone " in t)):
            out.append(h)
    return out


def _ban(q, f, periods, D, rulebook):
    if len(periods) != 1 or periods[0].start is None or periods[0].kind == "snapshot":
        return None
    p = periods[0]
    if pd.Timestamp(p.end) < D.dmin or pd.Timestamp(p.start) > D.dmax:
        return None
    O, L = D.O, D.L
    Lp = L[L["tinh"] & _in(L["ngay"], p)]
    Op = O[O["tinh"] & _in(O["ngay"], p)]
    ph = Lp[Lp["phone"]]

    # ---- tổng khách phải trả / trung bình mỗi đơn khách phải trả
    if re.search(r"khach phai tra", f) and not re.search(r"\bno\b|con thieu|gop|doanh thu", f):
        if not _co_luat(rulebook, "tra", "ky"):
            return None
        if re.search(r"trung binh|binh quan|moi don", f) and re.search(r"trung binh|binh quan", f):
            if not len(Op):
                return None
            return {"type": "number", "answer": _money(Op["kt"].sum() / len(Op))}, "Trung bình mỗi đơn = tổng khách phải trả / số đơn (không huỷ, không trả)"
        if re.search(r"\bdon\b.{0,8}(bao nhieu|may)|%|phan tram", f):
            return None
        return {"type": "number", "answer": _money(Op["kt"].sum())}, "Tổng khách phải trả = tổng 'Khách trả' của đơn không huỷ, không trả trong kỳ"

    # ---- tiền thu máy cũ
    if re.search(r"thu (lai |mua )?may cu|may cu thu|tien thu may", f) and D.has_mc:
        if re.search(r"%|phan tram|chiem|ty le|ti le", f) and re.search(r"\bdon\b|\bdh\b", f):
            if not len(Op):
                return None
            return {"type": "percent", "answer": _pct((Op["mc"] > 0).sum(), len(Op))}, "Tỉ lệ đơn có thu máy cũ / tổng đơn (không huỷ, không trả)"
        if re.search(r"bao nhieu (don|may|chiec)|\bmay chiec|trung binh|binh quan|%|phan tram|chiem", f):
            return None
        if _co_luat(rulebook, "mc", "ky"):
            return {"type": "number", "answer": _money(Op["mc"].sum())}, "Tổng tiền thu máy cũ của đơn không huỷ, không trả trong kỳ (không trừ vào doanh thu)"
        return None

    # ---- bảo hành
    if re.search(r"bao hanh", f) and D.has_bh:
        B = D.B
        if not _co_luat(rulebook, "bh"):
            return None
        Bp = B[_in(B["ngay"], p)]
        if re.search(r"%|phan tram|ty le|ti le|chiem", f) and re.search(r"may (ban|duoc)|so may|ban duoc|ban ra", f):
            if p.kind not in ("quarter", "month"):
                return None
            n = float(ph["sl"].sum())
            if n <= 0:
                return None
            return {"type": "percent", "answer": _pct(len(Bp), n)}, "Tỉ lệ bảo hành = số lượt bảo hành (ngày nhận trong kỳ) / số máy bán trong kỳ"
        if re.search(r"dong may|\bmay nao", f) and re.search(r"nhieu|nhut|nhat", f) and not re.search(r"\btop\b|\d+ dong", f):
            cnt = {}
            for v in Bp["may"].dropna():
                cnt.setdefault(fz(v), [0, {}])
                cnt[fz(v)][0] += 1
                cnt[fz(v)][1][str(v).strip()] = cnt[fz(v)][1].get(str(v).strip(), 0) + 1
            if not cnt:
                return None
            k = sorted(cnt, key=lambda x: (-cnt[x][0], x))
            if len(k) > 1 and cnt[k[0]][0] == cnt[k[1]][0]:
                return None
            return {"type": "string", "answer": max(cnt[k[0]][1], key=cnt[k[0]][1].get)}, "Dòng máy có nhiều lượt bảo hành nhất trong kỳ"
        if re.search(r"may ban|ban ra|da ban", f) and re.search(r"da (duoc )?(mang|dem)|mang (toi|lai|den)|it nhat", f):
            return None
        if re.search(r"luot|bao nhieu|may", f) and not re.search(r"ban ra|may ban|%|phan tram|dong may|top", f):
            return {"type": "number", "answer": int(len(Bp))}, "Số lượt bảo hành có ngày nhận trong kỳ"
        return None

    # ---- doanh thu / số máy / hãng / dòng máy
    if re.search(r"\bkhach\b|nhan vien|\bthu ngan\b", f):
        return None
    if not ("may" in f or "dien thoai" in f or "phu kien" in f or "hang" in f):
        tgt0 = _match_target(f, D)
        if not tgt0:
            return None
    if (p.meta.get("compare") is not None and not re.search(r"chiem|ty le|ti le", f)) or re.search(r"tang gia|doi gia|giam gia|tang hay giam|thay doi|so voi", f):
        return None
    flt = _loai_filter(f)
    nhom_may = re.search(r"nhom may (moi|cu)", f)
    pk = re.search(r"nhom phu kien", f)
    rev = re.search(r"doanh thu|dthu|bao nhieu tien|thu ve|tien ban|ban duoc.{0,10}tien|nhieu tien|\btien\b", f) and not re.search(r"tien (hang )?(\d)", f)
    tbinh = re.search(r"trung binh|binh quan", f)
    pc = re.search(r"%|phan tram|chiem|ty le|ti le", f)
    top = re.search(r"\btop\s*(\d+)|(\d+) dong may", f)
    tgt = _match_target(f, D)
    if tgt is False:
        return None
    hs = _hang_trong_cau(f, D)
    if hs and tgt is None and re.search(r"galaxy|redmi|reno|iphone \d|\bpro\b|\bmax\b|ultra|plus|\b[a-z]\d{2}\b|\d+ ?gb", f):
        return None

    def sel(x):
        if flt:
            x = x[x["loai"] == flt]
        return x

    # mặt hàng bán nhiều nhất trong nhóm máy
    if nhom_may and re.search(r"mat hang nao|san pham nao|may nao", f) and re.search(r"nhieu (may )?nhat|chay nhat|ban duoc nhieu nhat", f) and not rev:
        x = Lp[Lp["loai"] == ("may " + nhom_may[1])]
        if not len(x):
            return None
        g = x.groupby("ten")["sl"].sum().sort_values(ascending=False)
        g = g.reset_index().sort_values(["sl", "ten"], ascending=[False, True])
        if len(g) > 1 and g.iloc[0]["sl"] == g.iloc[1]["sl"]:
            return None
        return {"type": "string", "answer": str(g.iloc[0]["ten"])}, "Mặt hàng trong nhóm có tổng số lượng bán lớn nhất"
    # top k dòng máy
    if top and re.search(r"dong may", f) and re.search(r"nhieu|chay", f) and not rev and _co_luat(rulebook, "dong"):
        k = int(top[1] or top[2])
        g = ph.groupby("dong")["sl"].sum().reset_index().sort_values(["sl", "dong"], ascending=[False, True])
        if len(g) < k:
            return None
        if len(g) > k and g.iloc[k - 1]["sl"] == g.iloc[k]["sl"]:
            return None
        return {"type": "list", "answer": [str(x) for x in g.iloc[:k]["dong"]]}, f"Top {k} dòng máy (phần tên trước dung lượng) theo số máy bán"
    if re.search(r"dong may", f) and re.search(r"nao", f) and re.search(r"nhieu (may )?nhat|chay nhat|ban duoc nhieu nhat", f) and not rev and not top and _co_luat(rulebook, "dong"):
        g = ph.groupby("dong")["sl"].sum().reset_index().sort_values(["sl", "dong"], ascending=[False, True])
        if len(g) > 1 and g.iloc[0]["sl"] == g.iloc[1]["sl"]:
            return None
        return {"type": "string", "answer": str(g.iloc[0]["dong"])}, "Dòng máy bán nhiều máy nhất"

    # phần trăm
    if pc and not re.search(r"\bdon\b|\bdh\b|bao hanh|luot", f):
        den = ph
        numr = None
        what = ""
        if tgt is not None:
            numr = Lp[Lp[tgt[0]] == tgt[1]]
            den = Lp
            what = "mặt hàng/dòng"
        elif hs:
            if len(hs) != 1 or not _co_luat(rulebook, "hang"):
                return None
            numr = ph[ph["hangm"] == hs[0]]
            what = "hãng"
        elif flt or nhom_may:
            numr = sel(ph)
            what = "nhóm"
        else:
            return None
        if tgt is not None and not tgt_is_phone(D, tgt):
            den = Lp
        elif tgt is not None:
            den = ph
        key = "tt" if rev else "sl"
        if not len(den) or den[key].sum() == 0:
            return None
        return {"type": "percent", "answer": _pct(numr[key].sum(), den[key].sum())}, f"Tỉ trọng {what}: {key} / tổng {'tiền' if rev else 'số'} máy bán"

    # trung bình mỗi máy
    if tbinh and re.search(r"\bmay\b|dien thoai|\bdt\b", f) and not re.search(r"\bdon\b", f):
        x = sel(ph)
        if not len(x) or x["sl"].sum() == 0:
            return None
        return {"type": "number", "answer": _money(x["tt"].sum() / x["sl"].sum())}, "Giá bán trung bình mỗi máy = doanh thu máy / số máy bán"

    # doanh thu nhóm
    if rev and (nhom_may or pk) and tgt is None and not hs and not tbinh:
        x = Lp[Lp["loai"] == "phu kien"] if pk else Lp[Lp["loai"] == ("may " + nhom_may[1])]
        return {"type": "number", "answer": _money(x["tt"].sum())}, "Doanh thu nhóm hàng (theo cột Loại ở bảng ghi tắt) của đơn không huỷ, không trả"

    # doanh thu / số lượng theo mặt hàng, dòng máy, hãng
    qty_q = re.search(r"bao nhieu (chiec|cai|may|cai|chiec)|\bso may\b|may chiec|may cai|\bmay may\b|mot may|bao nhieu\b.*\b(chiec|cai)\b", f)
    if tgt is not None:
        if tgt[0] == "dong" and tgt_is_phone(D, tgt) and not _co_luat(rulebook, "dong"):
            return None
        x = Lp[Lp[tgt[0]] == tgt[1]]
        if rev and not qty_q:
            return {"type": "number", "answer": _money(x["tt"].sum())}, f"Doanh thu của {tgt[0]} '{tgt[1]}' (đơn không huỷ, không trả)"
        if qty_q and not rev and not tbinh:
            return {"type": "number", "answer": _num_out(x["sl"].sum())}, f"Số lượng bán của {tgt[0]} '{tgt[1]}'"
        return None
    if hs:
        if len(hs) != 1 or not _co_luat(rulebook, "hang"):
            return None
        x = ph[ph["hangm"] == hs[0]]
        if rev and not qty_q:
            return {"type": "number", "answer": _money(x["tt"].sum())}, f"Doanh thu máy hãng {hs[0]}"
        if qty_q and not rev:
            return {"type": "number", "answer": _num_out(x["sl"].sum())}, f"Số máy bán hãng {hs[0]}"
        return None

    # số máy bán (mới / cũ / tất cả)
    if qty_q and not rev and not tbinh and not tgt and not hs and re.search(r"\bmay\b|dien thoai|\bdt\b|chiec", f) \
            and not re.search(r"phu kien|\bdon\b|khach|hang nao|mat hang", re.sub(r"(khong ke|khong tinh|bo) phu kien( ra)?", "", f)):
        x = sel(ph)
        return {"type": "number", "answer": _num_out(x["sl"].sum())}, "Tổng số lượng máy (mới / cũ) bán trong kỳ, đơn không huỷ, không trả"
    return None


def tgt_is_phone(D, tgt):
    x = D.L[D.L[tgt[0]] == tgt[1]]
    return bool(len(x)) and bool(x["phone"].all())


def _num_out(v):
    v = float(v)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


def _bh_may_ban(q, f, periods, D, rulebook):
    """Số máy bán trong tháng/quý đã đem bảo hành ít nhất một lần, tính đến hết ngày X."""
    if not (D.has_bh and re.search(r"bao hanh", f) and re.search(r"may ban|ban ra|da ban", f)
            and re.search(r"bao nhieu may|may nao|da (duoc )?(mang|dem)|mang (toi|lai|den)", f)):
        return None
    if re.search(r"%|phan tram|ty le|ti le|luot", f) or not _co_luat(rulebook, "bh"):
        return None
    snap = [p for p in periods if p.kind in ("snapshot", "day")]
    per = [p for p in periods if p.kind in ("month", "quarter", "range", "year")]
    if len(per) != 1 or len(snap) > 1:
        return None
    X = pd.Timestamp(snap[0].end) if snap else pd.Timestamp(per[0].end)
    p = per[0]
    L, B = D.L, D.B
    sold = {im for ims, ng in zip(L[L["tinh"] & L["phone"]]["imei"], L[L["tinh"] & L["phone"]]["ngay"]) if pd.Timestamp(p.start) <= ng <= pd.Timestamp(p.end) for im in ims}
    bh = {i for i in B[B["ngay"] <= X]["imei"].dropna()}
    return {"type": "number", "answer": len(sold & bh)}, "Số máy (IMEI) bán trong kỳ có ít nhất một lượt bảo hành (ngày nhận <= ngày X)"


def solve(q, f, periods, biz, rulebook):
    D = _get(biz)
    if D is None:
        return None
    try:
        for fn in (_bh_may_ban, _debt, _ban):
            r = fn(q, f, periods, D, rulebook) if fn is not _debt else fn(q, f, periods, D, rulebook)
            if r:
                return r
    except Exception:
        import os
        if os.environ.get("DT_DEBUG"):
            raise
        return None
    return None
