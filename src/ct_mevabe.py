"""Công thức cố định (pandas, không LLM) cho ngành MẸ VÀ BÉ (rulebook §6 "### Mẹ và bé") + doanh thu / đơn của shop kiểu này.
solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.  Không chắc -> None.

Nhận diện shop bằng DẶC ĐIỂM DỮ LIỆU (không dùng business_id / tên shop):
  - sổ bán có dòng "Voucher" / "Phí ship" / "Dùng <n> điểm" (dòng không phải hàng, chỉ ghi HĐ, tên, TT);
  - danh mục hàng có cột Số / Size, Hãng, Nhóm; bảng khách có Ngày đăng ký.
Bảng chuẩn biz.ban của kiểu shop này SAI ở hai chỗ (xem báo cáo): thanh_tien = SL x ĐG (bỏ giảm giá từng dòng, cột "TT"
không nhận ra) và so_luong_co_ban trống (cột Quy đổi dạng chữ "1 thùng = 4 bịch"). Ở đây dựng lại từ bảng gốc biz.tables."""
import re
import copy
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from loader import fold
from sales import status_of

# ------------------------------------------------------------------ điều kiện bật theo rulebook (chữ bỏ dấu, bỏ dấu câu)
_LUAT = {
    "tien": "tra so duong cong theo ngay cua hoa don chi tinh hoa don khong huy khong tra",
    "sua": "hang de trong cot so size thi khong xep vao so size nao",
    "bim": "phan tram cua mot size hay mot hang chia cho tong so bich moi hang bim ta cung ky",
    "khach": "so khach co ngay dang ky trong thang do ke ca khach chua mua lan nao",
    "tt": "so khong co cot trang thai trang thai don ghi o cot ghi chu ghi chu khong nhac trang thai la don hoan thanh",
    "dt": "tong thanh tien cac dong hang",
    "tb": "gia tri trung binh mot don",
}


def _co_luat(rulebook, *keys):
    if not rulebook:
        return False
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    t = re.sub(r"\s+", " ", t)
    return all(_LUAT[k] in t for k in keys)


def _fc(c):
    return fold(str(c)).replace("_", " ").strip()


def _key(s):
    return re.sub(r"[^a-z0-9]+", "", fold(s)) if isinstance(s, str) else ""


# ------------------------------------------------------------------ dựng ngữ cảnh từ bảng gốc
_RE_VOUCHER = re.compile(r"^voucher$")
_RE_SHIP = re.compile(r"^(phi ship|phi giao hang|phi van chuyen|tien ship)$")
_RE_POINT = re.compile(r"^dung (\d+) diem$")


def _find_sales(biz):
    """Bảng sổ bán kiểu mẹ và bé: có cột HĐ, Ngày, Tên hàng, SL, TT, Ghi chú và dòng Voucher / Phí ship / Dùng n điểm."""
    for t, df in biz.tables.items():
        cols = {_fc(c): c for c in df.columns if c != "_nguon"}
        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        cid = pick("hd", "so hd", "ma hd", "so hoa don", "ma don", "so don", "hoa don")
        cdate = pick("ngay", "ngay ban")
        cname = pick("ten hang", "mat hang", "hang")
        cqty = pick("sl", "so luong")
        ctt = pick("tt", "thanh tien", "t tien")
        cnote = pick("ghi chu")
        if None in (cid, cdate, cname, cqty, ctt, cnote):
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[cdate]):
            continue
        if not (pd.api.types.is_numeric_dtype(df[ctt]) and pd.api.types.is_numeric_dtype(df[cqty])):
            continue
        nm = df[cname].map(lambda v: fold(v).strip() if isinstance(v, str) else "")
        if not (nm.map(lambda v: bool(_RE_VOUCHER.match(v) or _RE_SHIP.match(v) or _RE_POINT.match(v)))).any():
            continue
        return t, dict(id=cid, date=cdate, name=cname, qty=cqty, tt=ctt, note=cnote,
                       unit=pick("dvt", "don vi", "don vi tinh"), price=pick("dg", "don gia"),
                       channel=pick("kenh"), cust=pick("khach", "ten khach"))
    return None


def _find_catalog(biz):
    for t, df in biz.tables.items():
        cols = {_fc(c): c for c in df.columns if c != "_nguon"}
        cname = cols.get("ten hang")
        csize = next((c for k, c in cols.items() if re.fullmatch(r"so ?/ ?size|size|so", k)), None)
        cbrand = cols.get("hang")
        cgroup = cols.get("nhom") or cols.get("nhom hang")
        cunit = cols.get("dvt") or cols.get("don vi")
        cconv = cols.get("quy doi")
        if None in (cname, csize, cbrand, cgroup, cunit):
            continue
        return t, dict(name=cname, size=csize, brand=cbrand, group=cgroup, unit=cunit, conv=cconv)
    return None


def _find_members(biz):
    for t, df in biz.tables.items():
        cols = {_fc(c): c for c in df.columns if c != "_nguon"}
        cd = next((c for k, c in cols.items() if "dang ky" in k and pd.api.types.is_datetime64_any_dtype(df[c])), None)
        cn = next((c for k, c in cols.items() if k in ("ten kh", "ten khach", "khach hang", "ten khach hang", "khach")), None)
        if cd is not None and cn is not None:
            return t, cd, cn
    return None


def _factors(txt):
    """'1 thùng = 6 bịch; 1 lốc = 4 gói' -> {'thung': 6.0, 'loc': 4.0}"""
    out = {}
    if isinstance(txt, str):
        for a, x, b in re.findall(r"1\s*([^\s=]+)\s*=\s*(\d+(?:[.,]\d+)?)\s*([^\s;,]+)", txt):
            out[fold(a).strip()] = float(x.replace(",", "."))
    return out


def _ctx(biz):
    c = getattr(biz, "_mevabe_ctx", None)
    if c is not None:
        return c or None
    c = False
    try:
        c = _build(biz) or False
    except Exception:
        c = False
    try:
        setattr(biz, "_mevabe_ctx", c)
    except Exception:
        pass
    return c or None


def _build(biz):
    fs = _find_sales(biz)
    if fs is None:
        return None
    t, r = fs
    df = biz.tables[t]
    df = df[df[r["id"]].notna()].copy()
    df["_hd"] = df[r["id"]].astype(str).str.strip()
    # ngày của đơn: dòng đầu có ngày (ô trống đã điền xuống theo nhóm); trạng thái: dòng nào của đơn cũng áp cho cả đơn
    df["_ngay"] = pd.to_datetime(df.groupby("_hd")[r["date"]].transform("min"), errors="coerce").dt.normalize()
    st = status_of(df[r["note"]])
    df["_huy"] = (st == "huỷ").groupby(df["_hd"]).transform("any")
    df["_tra"] = (st == "trả hàng").groupby(df["_hd"]).transform("any")
    df["_tinh"] = ~df["_huy"] & ~df["_tra"]
    nm = df[r["name"]].map(lambda v: fold(v).strip() if isinstance(v, str) else "")
    df["_kind"] = np.select(
        [nm.map(lambda v: bool(_RE_VOUCHER.match(v))), nm.map(lambda v: bool(_RE_SHIP.match(v))),
         nm.map(lambda v: bool(_RE_POINT.match(v)))], ["voucher", "ship", "point"], "")
    df.loc[(df["_kind"] == "") & df[r["qty"]].isna(), "_kind"] = "khac"      # dòng ghi chú cuối sheet...
    df["_diem"] = nm.map(lambda v: int(_RE_POINT.match(v).group(1)) if _RE_POINT.match(v) else np.nan)
    ctx = dict(df=df, r=r, tbl=t)
    # danh mục
    fc = _find_catalog(biz)
    if fc is not None:
        ct, cr = fc
        cat = biz.tables[ct]
        cat = cat[cat[cr["name"]].notna()].copy()
        ctx["cat"], ctx["cr"] = cat, cr
        byk = {}
        for _, row in cat.iterrows():
            byk[_key(row[cr["name"]])] = row
        ctx["byk"] = byk
    fm = _find_members(biz)
    if fm is not None:
        ctx["mem"] = fm
    return ctx


def _items(ctx):
    """Các dòng hàng (không phải Voucher / Phí ship / Dùng điểm) + thông tin danh mục + số lượng quy về đơn vị cơ bản."""
    if "_items" in ctx:
        return ctx["_items"]
    df, r = ctx["df"], ctx["r"]
    it = df[df["_kind"] == ""].copy()
    it["_tt"] = it[r["tt"]]
    it["_name"] = it[r["name"]]
    if "byk" in ctx:
        cr = ctx["cr"]
        rows = it[r["name"]].map(lambda v: ctx["byk"].get(_key(v)))
        it["_nhom"] = rows.map(lambda x: x[cr["group"]] if x is not None else np.nan)
        it["_hang"] = rows.map(lambda x: x[cr["brand"]] if x is not None else np.nan)
        it["_size"] = rows.map(lambda x: x[cr["size"]] if x is not None else np.nan)
        it["_cat_ok"] = rows.map(lambda x: x is not None)
        base = rows.map(lambda x: fold(x[cr["unit"]]).strip() if x is not None and isinstance(x[cr["unit"]], str) else None)
        conv = rows.map(lambda x: _factors(x[cr["conv"]]) if x is not None and cr["conv"] else {})
        unit = it[r["unit"]].map(lambda v: fold(v).strip() if isinstance(v, str) else None) if r["unit"] else pd.Series(None, index=it.index)
        fac = []
        for u, b, cv in zip(unit, base, conv):
            if b is not None and u == b:
                fac.append(1.0)
            elif u in cv:
                fac.append(cv[u])
            else:
                fac.append(np.nan)
        it["_fac"] = fac
        it["_base"] = base
        it["_qb"] = pd.to_numeric(it[r["qty"]], errors="coerce") * it["_fac"]
    ctx["_items"] = it
    return it


def _bounds(p, ctx):
    st = pd.Timestamp(p.start).normalize() if p.start is not None else ctx["df"]["_ngay"].min()
    return st, pd.Timestamp(p.end).normalize()


def _intp(s, st, en):
    return (s >= st) & (s <= en)


_BAD_COMMON = re.compile(
    r"\bso voi\b|%|phan tram|trung binh|\bnhat\b|\bdon\b|khach nao|\bai\b|ti le|ty le|\bkenh\b|\bzalo\b|\bsan\b|tmdt|"
    r"tai (quay|tiem|cua hang)|\bnhom\b|hang nao|\btung\b|moi thang|lan luot|\bmay lan\b|bao nhieu lan|bao nhieu khach|"
    r"\bmay khach\b|hon hay kem|cao hon|thap hon|\bhon\b|\bkem\b|tang|nhieu hon|it hon|chenh|\btop\b|"
    r"hang\b.*\bnao\b|mat hang|\bmon\b|san pham|\bngay nao\b|thu may|xep hang|liet ke|danh sach|\bgom\b")
_RE_VOU = re.compile(r"\bvoucher\b|\bvou?\s?cher\b|\bv(ou|au|ao)\s?(cho|chu|chor|cha)\b")
_RE_SH = re.compile(r"phi ship|tien ship|phi giao hang|phi van chuyen|\bship\b")
_RE_PT = re.compile(r"\bdiem\b")


def _one_period(periods):
    if len(periods) != 1:
        return None
    p = periods[0]
    if p.kind == "snapshot" or p.start is None:
        return None
    return p


def _num(v):
    v = float(v)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


# ------------------------------------------------------------------ 1) tiền voucher / điểm / phí ship
def _tien(f, p, ctx, rulebook):
    if not _co_luat(rulebook, "tien", "tt") or _BAD_COMMON.search(f):
        return None
    vou, sh, pt = _RE_VOU.search(f), _RE_SH.search(f), _RE_PT.search(f)
    if [bool(vou), bool(sh), bool(pt)].count(True) != 1:
        return None
    if re.search(r"\b(con|tich|hien co|du|tich luy|duoc tich|cong them|tang diem|qua tang|mien phi|free|khong)\b", f):
        return None
    df = ctx["df"]
    st, en = _bounds(p, ctx)
    x = df[df["_tinh"] & _intp(df["_ngay"], st, en)]
    kind = "voucher" if vou else ("ship" if sh else "point")
    x = x[x["_kind"] == kind]
    tt = x[ctx["r"]["tt"]]
    if tt.isna().any():
        return None
    if kind == "point":
        # "Dùng n điểm": mỗi điểm trừ 100đ — kiểm khớp, nếu lệch thì không chắc
        if (x["_diem"] * 100 - tt.abs()).abs().max() > 0.5 if len(x) else False:
            return None
        if re.search(r"bao nhieu diem|may diem|\bso diem\b", f) and not re.search(r"\btien\b|\bdong\b|\bvnd\b|giam|tru", f):
            return {"type": "number", "answer": _num(x["_diem"].sum())}, "Số điểm khách dùng = tổng n của dòng 'Dùng n điểm' (đơn không huỷ, không trả, theo ngày hoá đơn)"
        if not re.search(r"\btien\b|\bdong\b|\bvnd\b|bao nhieu|\bmay\b", f):
            return None
    v = float(tt.abs().sum())
    return {"type": "number", "answer": _num(v)}, \
        f"Tổng |TT| các dòng '{kind}' (số dương) của hoá đơn không huỷ, không trả, ngày hoá đơn {st:%Y-%m-%d}..{en:%Y-%m-%d}"


# ------------------------------------------------------------------ 2) khách thành viên mới trong kỳ
def _khach_moi(f, p, ctx, rulebook):
    if "mem" not in ctx or not _co_luat(rulebook, "khach"):
        return None
    if not re.search(r"\bdang ky\b", f) or not re.search(r"\bkhach\b", f):
        return None
    if re.search(r"\bso voi\b|%|phan tram|trung binh|\bnhat\b|\bai\b|ti le|ty le|tang|hon|kem|\bnao\b|top|liet ke|danh sach|ten\b|\bmua\b|\bdon\b|doanh thu", f):
        return None
    if not re.search(r"bao nhieu|\bmay\b|\bso khach\b|\bdem\b|tong|bnhiu|\bso\b", f):
        return None
    t, cd, cn = ctx["mem"]
    m = ctx_members(ctx)
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    d = pd.to_datetime(m[cd], errors="coerce").dt.normalize()
    return {"type": "number", "answer": int(((d >= st) & (d <= en)).sum())}, \
        "Số khách có Ngày đăng ký trong kỳ (kể cả khách chưa mua lần nào), mỗi dòng bảng khách là một khách"


def ctx_members(ctx):
    return ctx["_biz"].tables[ctx["mem"][0]]


# ------------------------------------------------------------------ 3) số sữa / size bỉm
def _loc_nhom(f, cat, cr):
    toks = set(f.split())
    toks = {re.sub(r"[^a-z0-9]", "", w) for w in toks}
    groups = {}
    for g, sub in cat.groupby(cr["group"]):
        if sub[cr["size"]].notna().any():
            groups[g] = sub
    hit = []
    for g in groups:
        fg = fold(g)
        first = fg.split()[0] if fg.split() else ""
        alias = {"bim": ("ta",)}.get(first, ())
        if re.search(rf"(?<![a-z0-9]){re.escape(fg)}(?![a-z0-9])", f) or first in toks or any(a in toks for a in alias):
            hit.append(g)
    return hit[0] if len(hit) == 1 else None


def _sua_bim(f, p, ctx, rulebook):
    if "cat" not in ctx:
        return None
    cat, cr = ctx["cat"], ctx["cr"]
    if re.search(r"\bso voi\b|tang|giam|hon\b|kem\b|chenh|lan luot|\bkenh\b|\bzalo\b|\bsan\b|tmdt|tai (quay|tiem)|\btien\b|doanh thu|dthu|trung binh|"
                 r"\bdon\b|khach|\bngay\b|\bnhom\b.*\bnhom\b|top|liet ke", f):
        return None
    grp = _loc_nhom(f, cat, cr)
    if grp is None:
        return None
    sub = cat[cat[cr["group"]] == grp]
    # size / số
    sz = None
    msz = re.search(r"\bsize ([a-z]{1,3})\b", f)
    mso = re.search(r"\bso (\d+)\b", f)
    if msz and mso:
        return None
    vals = {fold(v): v for v in sub[cr["size"]].dropna().unique()}
    if msz:
        sz = vals.get("size " + msz.group(1))
        if sz is None:
            return None
    elif mso:
        sz = vals.get("so " + mso.group(1))
        if sz is None:
            return None
    elif re.search(r"\bsize\b", f) and not re.search(r"(moi|tat ca|cac|het|gop|all|nhieu|ca) size|size (nao|bat ky)", f):
        return None
    elif re.search(r"\bso \d", f):
        return None
    # hãng
    brands = {fold(b): b for b in sub[cr["brand"]].dropna().unique()}
    bh = [b for fb, b in brands.items() if re.search(rf"(?<![a-z0-9]){re.escape(fb)}(?![a-z0-9])", f)]
    if len(bh) > 1:
        return None
    brand = bh[0] if bh else None
    # kiểm tra các từ "bỉm Bobby ... mọi size" vv: không có cụm lạ nào về lọc khác
    it = _items(ctx)
    st, en = _bounds(p, ctx)
    x = it[it["_tinh"] & _intp(it["_ngay"], st, en)]
    if len(x) == 0:
        return None
    if (~x["_cat_ok"]).any():
        return None
    x = x[x["_nhom"] == grp]
    if x["_qb"].isna().any():
        return None
    base_units = set(x["_base"].dropna())
    if len(base_units) != 1:
        return None
    unit = next(iter(base_units))
    pct = re.search(r"%|phan tram", f)
    want_brand_top = re.search(r"\bhang nao\b", f)
    sel = x
    if sz is not None:
        sel = sel[sel["_size"] == sz]
    if brand is not None:
        sel = sel[sel["_hang"] == brand]
    if pct:
        if fold(grp) != "bim ta" and not re.search(r"\bbim\b|\bta\b", f):
            return None
        if not _co_luat(rulebook, "bim"):
            return None
        if not re.search(r"chiem|ti le|ty le|chiem bao nhieu|bao nhieu %|may %|bao nhieu phan tram|may phan tram", f):
            return None
        if sz is not None and brand is not None:
            return None
        if mso:
            return None
        if sz is None and brand is None:
            return None
        den = float(x["_qb"].sum())
        if den <= 0:
            return None
        num = float(sel["_qb"].sum())
        return {"type": "percent", "answer": float(Decimal(str(num / den * 100)).quantize(Decimal("0.01"), ROUND_HALF_UP))}, \
            f"% = số {unit} ({'size ' + str(sz) if sz else ''}{' ' + str(brand) if brand else ''}) / tổng số {unit} mọi hàng nhóm '{grp}' cùng kỳ ({num:g}/{den:g})"
    if want_brand_top:
        if sz is None or brand is not None:
            return None
        if not re.search(r"nhieu\b.*\b(nhat|nhut|nhit)\b|ban chay nhat", f):
            return None
        g = sel.groupby("_hang")["_qb"].sum()
        if not len(g):
            return None
        g = g.reset_index().sort_values(["_qb", "_hang"], ascending=[False, True])
        if len(g) > 1 and g.iloc[0]["_qb"] == g.iloc[1]["_qb"]:
            pass   # hoà: theo tên tăng dần (rulebook §9 không có mã)
        return {"type": "string", "answer": str(g.iloc[0]["_hang"])}, \
            f"Hãng có tổng số {unit} lớn nhất trong các hàng '{sz}' của nhóm '{grp}' (hoà: tên tăng dần)"
    if re.search(r"\bnao\b|nhat|\bai\b", f):
        return None
    if sz is None and brand is None:
        return None
    mu = re.search(rf"(?<![a-z0-9]){re.escape(unit)}(?![a-z0-9])", f)
    if not mu or not re.search(r"bao nhieu|\bmay\b|tong|bnhiu|bn\b|nhieu", f):
        return None
    return {"type": "number", "answer": _num(sel["_qb"].sum())}, \
        f"Tổng số {unit} (quy về đơn vị cơ bản) của hàng nhóm '{grp}' có Số/Size = {sz} hãng = {brand}, đơn không huỷ, không trả"


# ------------------------------------------------------------------ 4) doanh thu / đơn: bảng chuẩn dựng lại đúng
def _ban_dung(biz, ctx):
    """biz.ban sửa: thanh_tien lấy từ cột TT (đã trừ giảm giá từng dòng), so_luong_co_ban theo cột Quy đổi."""
    if "_ban2" in ctx:
        return ctx["_ban2"]
    ban = getattr(biz, "ban", None)
    out = None
    if ban is not None and len(ban):
        df, r = ctx["df"], ctx["r"]
        raw = biz.tables[ctx["tbl"]]
        ok = len(raw) == len(ban)
        if ok:
            ids_b = ban["ma_don"].astype(object).where(ban["ma_don"].notna(), None).map(lambda v: None if v is None else str(v).strip()).values
            ids_r = raw[r["id"]].astype(object).where(raw[r["id"]].notna(), None).map(lambda v: None if v is None else str(v).strip()).values
            ok = bool(all(a == b for a, b in zip(ids_b, ids_r)))
        if ok:
            b2 = ban.copy()
            tt = pd.to_numeric(raw[r["tt"]], errors="coerce").values
            qty = pd.to_numeric(raw[r["qty"]], errors="coerce").values
            b2["thanh_tien"] = np.where(~np.isnan(qty), tt, np.nan)
            it = _items(ctx)
            if "_qb" in it:
                qb = pd.Series(np.nan, index=raw.index)
                qb.loc[it.index] = it["_qb"]
                b2["so_luong_co_ban"] = qb.values
                bs = pd.Series(None, index=raw.index, dtype=object)
                bs.loc[it.index] = it["_base"]
                b2["don_vi_co_ban"] = bs.values
            out = b2
    ctx["_ban2"] = out
    return out


def _dt(q, f, periods, biz, ctx, rulebook):
    if not _co_luat(rulebook, "tt", "dt"):
        return None
    b2 = _ban_dung(biz, ctx)
    if b2 is None:
        return None
    biz2 = copy.copy(biz)
    biz2.ban = b2
    import ct_coban, ct_tyle, ct_top
    for mod in (ct_coban, ct_tyle, ct_top):
        try:
            r = mod.solve(q, f, periods, biz2, rulebook)
        except Exception:
            r = None
        if r is not None and r[0].get("type") not in (None, "error"):
            return r[0], f"{mod.__name__} trên bảng chuẩn đã sửa thanh_tien = cột TT: {r[1]}"
    try:
        import cong_thuc
        r = cong_thuc.try_solve(q, periods, biz2, rulebook)
    except Exception:
        r = None
    if r is not None and r[0].get("type") not in (None, "error"):
        return r[0], f"cong_thuc trên bảng chuẩn đã sửa: {r[1]}"
    return None



# ------------------------------------------------------------------ 5) doanh thu nhiều tháng "lần lượt", theo nhóm hàng, theo kênh
_BAD_DT = re.compile(r"\bso voi\b|%|phan tram|trung binh|\bnhat\b|\bdon\b|khach|ti le|ty le|ti trong|chiem|tang|giam|\bhon\b|\bkem\b|"
                     r"chenh|\btop\b|mat hang|\bmon\b|san pham|\bngay\b|thu may|xep hang|liet ke|loi|lai\b|von\b|\bno\b")


def _rev_range(ctx, st, en, mask=None):
    it = _items(ctx)
    x = it[it["_tinh"] & _intp(it["_ngay"], st, en)]
    if mask is not None:
        x = x[mask(x)]
    return x


def _thang_list(f):
    """Danh sách (năm, tháng) theo thứ tự câu hỏi nêu: 'tháng 5, 6, 7/2025' / 'th9, th10, th11/2025' / 'tháng 1/2026, 2/2026, 3/2026'."""
    m = re.search(r"\b(?:thang|th|t)\s*(\d{1,2}(?:\s*,\s*(?:thang|th|t)?\s*\d{1,2})+)\s*/\s*(\d{4})\b", f)
    if m:
        ms = [int(v) for v in re.findall(r"\d{1,2}", m.group(1))]
        return [(int(m.group(2)), v) for v in ms]
    pairs = re.findall(r"(?<![\d/])(\d{1,2})\s*/\s*(\d{4})\b", f)
    if len(pairs) >= 2:
        return [(int(y), int(mo)) for mo, y in pairs]
    return None


def _dt_them(f, periods, ctx, rulebook, as_of_max):
    if not _co_luat(rulebook, "dt", "tt") or _BAD_DT.search(f.replace("tung thang", " ").replace("lan luot", " ")):
        return None
    if not re.search(r"doanh thu|dthu|tien ban|ban (duoc|ra) bao nhieu tien", f):
        return None
    df = ctx["df"]
    lo, hi = df["_ngay"].min(), df["_ngay"].max()
    ml = _thang_list(f)
    if ml and 2 <= len(ml) <= 6 and re.search(r"lan luot|tung thang|moi thang", f) and all(1 <= m <= 12 for _, m in ml) and len(set(ml)) == len(ml):
        out = []
        for y, m in ml:
            st = pd.Timestamp(y, m, 1)
            en = st + pd.offsets.MonthEnd(0)
            if en < lo or st > hi:
                return None
            out.append(int(round(float(_rev_range(ctx, st, en)["_tt"].sum()))))
        return {"type": "list", "answer": out}, "Doanh thu từng tháng (tổng TT các dòng hàng, đơn không huỷ, không trả), theo thứ tự câu hỏi"
    return None


def _nhom_hang(f, p, ctx, rulebook):
    if "cat" not in ctx or not _co_luat(rulebook, "dt", "tt"):
        return None
    if _BAD_DT.search(f) or not re.search(r"\bnhom\b", f):
        return None
    if not re.search(r"bao nhieu|doanh thu|dthu|\bmay\b", f):
        return None
    cat, cr = ctx["cat"], ctx["cr"]
    hit = [g for g in cat[cr["group"]].dropna().unique()
           if re.search(rf"\bnhom (?:hang )?{re.escape(fold(g))}(?![a-z0-9])", f)]
    if len(hit) != 1:
        return None
    it = _items(ctx)
    st, en = _bounds(p, ctx)
    x = it[it["_tinh"] & _intp(it["_ngay"], st, en)]
    if len(x) == 0 or (~x["_cat_ok"]).any():
        return None
    v = float(x.loc[x["_nhom"] == hit[0], "_tt"].sum())
    return {"type": "number", "answer": int(round(v))}, f"Doanh thu nhóm hàng '{hit[0]}' = tổng TT các dòng hàng thuộc nhóm (danh mục), đơn không huỷ, không trả"


def _kenh_values(ctx):
    r = ctx["r"]
    if not r.get("channel"):
        return {}
    return {fold(v).strip(): v for v in ctx["df"][r["channel"]].dropna().unique() if isinstance(v, str)}


def _kenh_in(f, vals):
    """Các kênh được nêu trong câu (theo thứ tự xuất hiện) -> [giá trị kênh trong dữ liệu]; None nếu không rõ."""
    hits = []
    for fv, v in vals.items():
        pos = []
        if re.search(rf"(?<![a-z0-9]){re.escape(fv)}(?![a-z0-9])", f):
            pos.append(f.find(fv))
        first = fv.split()[0]
        if first == "san" and re.search(r"san thuong mai dien tu|\btmdt\b|san tmdt", f):
            m = re.search(r"san thuong mai dien tu|\btmdt\b|san tmdt", f); pos.append(m.start())
        if first == "tai" and re.search(r"tai (quay|tiem|cua hang|shop)", f):
            m = re.search(r"tai (quay|tiem|cua hang|shop)", f); pos.append(m.start())
        if first not in ("san", "tai") and re.search(rf"(?<![a-z0-9]){re.escape(first)}(?![a-z0-9])", f):
            pos.append(re.search(rf"(?<![a-z0-9]){re.escape(first)}(?![a-z0-9])", f).start())
        if pos:
            hits.append((min(pos), v))
    hits.sort()
    return [v for _, v in hits]


def _kenh(f, p, ctx, rulebook):
    vals = _kenh_values(ctx)
    if not vals or not _co_luat(rulebook, "dt", "tt"):
        return None
    if re.search(r"\bso voi\b|trung binh|\bnhat\b|khach|tang|giam|\bhon\b|\bkem\b|chenh|\btop\b|mat hang|\bmon\b|san pham|\bngay\b|"
                 r"thu may|loi|lai\b|von\b|\bno\b|\bnhom\b|\bso don\b|bao nhieu don|\bmay don\b", f):
        return None
    hits = _kenh_in(f, vals)
    if not hits:
        return None
    r = ctx["r"]
    it = _items(ctx)
    st, en = _bounds(p, ctx)
    x = it[it["_tinh"] & _intp(it["_ngay"], st, en)]
    if len(x) == 0:
        return None
    ch = x[r["channel"]]
    if ch.isna().any():
        return None
    pct = re.search(r"%|phan tram", f)
    if pct and len(hits) == 1 and re.search(r"chiem|dong gop|ti trong|ti le|ty le", f) and re.search(r"tong doanh thu|doanh thu", f):
        tot = float(x["_tt"].sum())
        if tot <= 0:
            return None
        v = float(x.loc[ch == hits[0], "_tt"].sum()) / tot * 100
        return {"type": "percent", "answer": float(Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP))}, \
            f"Tỉ trọng kênh '{hits[0]}' = doanh thu kênh / doanh thu cả shop cùng kỳ (đơn không huỷ, không trả)"
    if pct:
        return None
    if re.search(r"doanh thu|dthu|bao nhieu tien|tien ban", f):
        if len(hits) == 1:
            return {"type": "number", "answer": int(round(float(x.loc[ch == hits[0], "_tt"].sum())))}, f"Doanh thu kênh '{hits[0]}'"
        if len(hits) == 2 and re.search(r"tung (ben|kenh|loai)|lan luot|moi (ben|kenh)", f):
            return {"type": "list", "answer": [int(round(float(x.loc[ch == h, "_tt"].sum()))) for h in hits]}, \
                "Doanh thu từng kênh theo thứ tự câu hỏi nêu: " + ", ".join(map(str, hits))
    return None



# ------------------------------------------------------------------ 6) doanh thu / giá trị trung bình một đơn của CẢ shop, câu ngắn có từ thừa (ct_coban từ chối)
_OK_TOK = set("""doanh thu dthu doanhthu tong cong la bao nhieu bn bnhiu nhiu tien cua shop tiem hang quan minh toi tui mik cho hoi xem nho
duoc dc the vay nhi nha a ha ne z zay gia tri trung binh moi mot don thang th t tuan truoc nay roi hom qua tu dau nam den het
tinh ngay gan nhat ban ra ve thu vao co cac tai nam ngoai nhe vay a em e anh chi ơi oi giup""".split())


def _chinh_ky(periods):
    ps = list(periods)
    if len(ps) == 1:
        return ps[0]
    if len(ps) == 2:
        for a, b in ((ps[0], ps[1]), (ps[1], ps[0])):
            if b.kind == "day" and b.end == a.end and a.kind in ("year_to_date", "month_to_date", "week_to_date") and a.start is not None:
                return a
    return None


def _dt_don_gian(f, periods, ctx, rulebook):
    if not _co_luat(rulebook, "dt", "tt", "tb") or "mem" not in ctx and False:
        return None
    p = _chinh_ky(periods)
    if p is None or p.kind == "snapshot" or p.start is None:
        return None
    if re.search(r"\bso voi\b|%|phan tram|\bngay nao\b|bao nhieu ngay|\bmay ngay\b|\bso ngay\b|thu may|\bnhom\b|\bkenh\b|\bzalo\b|\bsan\b|khach|"
                 r"mat hang|\bmon\b|san pham|chenh|tang|giam|\bhon\b|\bkem\b|loi|lai\b|von\b|\bno\b|\btop\b|nhieu nhat|it nhat", f):
        return None
    toks = re.findall(r"[a-z]+", re.sub(r"\d+", " ", f))
    if any(t not in _OK_TOK for t in toks):
        return None
    avg = re.search(r"trung binh|binh quan", f)
    if re.search(r"\bdon\b", f) and not avg:
        return None
    if not re.search(r"doanh thu|dthu|doanhthu|gia tri|tien", f):
        return None
    df = ctx["df"]
    lo, hi = df["_ngay"].min(), df["_ngay"].max()
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    if en < lo or st > hi:
        return None
    x = _rev_range(ctx, st, en)
    if not len(x):
        return None
    tot = float(x["_tt"].sum())
    if avg:
        if not re.search(r"\bdon\b", f):
            return None
        n = x["_hd"].nunique()
        v = int(Decimal(str(tot / n)).quantize(Decimal("1"), ROUND_HALF_UP))
        return {"type": "number", "answer": v}, f"Doanh thu / số đơn ({tot:,.0f} / {n}), làm tròn nửa lên"
    return {"type": "number", "answer": int(round(tot))}, f"Doanh thu = tổng TT các dòng hàng, đơn không huỷ, không trả, {st:%Y-%m-%d}..{en:%Y-%m-%d}"


# ------------------------------------------------------------------ vào chính
def solve(q, f, periods, biz, rulebook):
    if not _co_luat(rulebook, "tt"):
        return None
    ctx = _ctx(biz)
    if ctx is None:
        return None
    ctx["_biz"] = biz
    r = _dt_them(f, periods, ctx, rulebook, None)
    if r is not None:
        return r
    p = _one_period(periods)
    if p is not None:
        for fn in (_tien, _khach_moi, _sua_bim, _nhom_hang, _kenh):
            r = fn(f, p, ctx, rulebook)
            if r is not None:
                return r
    r = _dt(q, f, periods, biz, ctx, rulebook)
    if r is not None:
        return r
    return _dt_don_gian(f, periods, ctx, rulebook)
