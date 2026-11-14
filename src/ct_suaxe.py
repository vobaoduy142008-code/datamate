"""Công thức cố định (pandas, không LLM) cho ngành SỬA XE — bám mục "### Sửa xe" của rulebook.
Nhận diện shop bằng đặc điểm dữ liệu (cột biển số / thợ / làm gì / tiền / ghi chú bảo hành), không dùng business_id.
Không chắc -> None để LLM làm."""
import re
import math
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from loader import fold

# ---------------------------------------------------------------- gate theo chữ rulebook
_LUAT = {
    "luot": "moi dong so sua xe la mot luot xe",
    "tb": "chia cho so phieu co thu tien",
    "bh": "so phieu bao hanh co ngay trong ky chia cho so phieu co thu tien",
    "quen": "xe co tu 2 phieu co thu tien trong ky",
    "viec": "co lam mot viec phieu co viec do",
    "nhot": "nhot ghi tron la nhot may khong phai nhot lap",
    "ruot": "ruot ghi tron la thay ruot moi",
    "dien": "sua dien khong gom thay bong den thay ac quy",
    "xe": "so chiec xe mang toi so bien so khac nhau",
    "tho": "tho lam nhieu nhat theo so luot xe",
    "chamchi": "cham chi nhat nguoi co nhieu ngay lam viec nhat",
    "thu": "chia cho so ngay thu x trong ky",
}


def _co_luat(rulebook, key):
    if rulebook is None:
        return True
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    return _LUAT[key] in re.sub(r"\s+", " ", t)


def _gate(rulebook):
    if rulebook is None:
        return True
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    t = re.sub(r"\s+", " ", t)
    return "sua xe" in t and _LUAT["luot"] in t


# ---------------------------------------------------------------- làm tròn
def _rhu(x):
    """làm tròn về đồng, nửa đồng làm tròn lên"""
    return int(Decimal(repr(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _r2(x):
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- nhận diện & chuẩn bị bảng
_PLATE = re.compile(r"(?<![0-9A-Za-z])(\d{2}[- ]?[A-Za-z][A-Za-z0-9]?[- ]?\d{3}[. ]?\d{2})(?![0-9A-Za-z])")


class _Shop:
    pass


def _prep(biz):
    c = getattr(biz, "_suaxe_cache", "none")
    if c != "none":
        return c
    try:
        res = _prep2(biz)
    except Exception:
        res = None
    try:
        object.__setattr__(biz, "_suaxe_cache", res)
    except Exception:
        pass
    return res


def _prep2(biz):
    tabs = getattr(biz, "tables", None) or {}
    for name, df in tabs.items():
        cols = {c: fold(c) for c in df.columns if not str(c).startswith("_")}
        dcol = [c for c in cols if str(df[c].dtype).startswith("datetime")]
        if not dcol:
            continue
        dcol = sorted(dcol, key=lambda c: 0 if "ngay" in cols[c] else 1)[0]
        pcol = None
        for c in cols:
            if df[c].dtype != object:
                continue
            s = df[c].dropna().astype(str)
            if len(s) >= 20 and s.map(lambda v: bool(_PLATE.search(v))).mean() >= 0.7:
                pcol = c
                break
        tcol = [c for c in cols if re.search(r"\btho\b", cols[c])]
        wcol = [c for c in cols if re.search(r"lam gi|cong viec|\bviec\b", cols[c])]
        mcol = [c for c in cols if re.search(r"\btien\b", cols[c]) and pd.api.types.is_numeric_dtype(df[c])]
        ncol = [c for c in cols if "ghi chu" in cols[c]]
        if not (pcol and tcol and wcol and mcol):
            continue
        tcol, wcol, mcol = tcol[0], wcol[0], mcol[0]
        ncol = ncol[0] if ncol else None
        d = pd.DataFrame({"ngay": pd.to_datetime(df[dcol], errors="coerce").dt.normalize(),
                          "xe": df[pcol], "lg": df[wcol], "tho_raw": df[tcol],
                          "tien": pd.to_numeric(df[mcol], errors="coerce"),
                          "note": df[ncol] if ncol else None})
        d = d[d["ngay"].notna() & d["xe"].notna()].copy()
        if len(d) < 50:
            continue
        sh = _Shop()
        sh.name = name
        fnote = d["note"].fillna("").astype(str).map(fold)
        flg = d["lg"].fillna("").astype(str).map(fold)
        d["bh"] = fnote.str.contains("bao hanh") | flg.str.match(r"^(bh|bao hanh)\b")
        if d["bh"].sum() == 0:
            continue
        pl = d["xe"].astype(str).map(lambda v: (_PLATE.search(v).group(1) if _PLATE.search(v) else None))
        d["plate_txt"] = pl
        d["plate"] = pl.map(lambda v: re.sub(r"[^A-Za-z0-9]", "", v).upper() if v else None)
        sh.plate_ok = bool(d["plate"].notna().all())
        d["tien"] = d["tien"].fillna(0.0)
        d["tc"] = (~d["bh"]) & (d["tien"] > 0)
        # dòng xe: phần ô Xe ngoài biển số; gộp các cách ghi bằng cách cùng biển số
        orig = [(re.sub(r"\s+", " ", _PLATE.sub(" ", str(x)).strip(" -,.")) if p else "") for x, p in zip(d["xe"], pl)]
        keys = [fold(o) for o in orig]
        d["dk"] = keys
        sh.dong_ok = sh.plate_ok and (pd.Series(keys) != "").mean() > 0.95
        if sh.dong_ok:
            par = {}

            def find(a):
                while par.setdefault(a, a) != a:
                    par[a] = par[par[a]]
                    a = par[a]
                return a
            for p, grp in pd.DataFrame({"p": d["plate"].values, "k": keys}).groupby("p")["k"]:
                ks = sorted(set(grp))
                for k in ks[1:]:
                    ra, rb = find(ks[0]), find(k)
                    if ra != rb:
                        par[rb] = ra
            root = [find(k) for k in keys]
            d["dong"] = root
            cnt = pd.DataFrame({"r": root, "o": orig}).groupby(["r", "o"]).size().reset_index(name="n")
            cnt = cnt.sort_values(["r", "n", "o"], ascending=[True, False, True])
            sh.dong_disp = cnt.drop_duplicates("r").set_index("r")["o"].to_dict()
            sh.dong_keys = {}
            for k in set(keys):
                sh.dong_keys.setdefault(find(k), set()).add(k)
        # thợ: khớp tên gọi (từ cuối) với họ tên trong danh sách thợ
        roster = {}
        for n2, d2 in tabs.items():
            c2 = {c: fold(c) for c in d2.columns}
            nm = [c for c in c2 if c2[c] in ("ho ten", "ten", "ten tho", "ho va ten")]
            hh = [c for c in c2 if "hoa hong" in c2[c]]
            if nm and hh and n2 != name:
                for _, r in d2.iterrows():
                    if pd.notna(r[nm[0]]):
                        roster[str(r[nm[0]]).strip()] = float(r[hh[0]]) if pd.notna(r[hh[0]]) else None
        sh.roster = roster
        last = {}
        for full in roster:
            toks = [t for t in re.split(r"[\s.]+", fold(full)) if t]
            if toks:
                last.setdefault(toks[-1], []).append(full)
        mp = {}
        ok = bool(roster)
        for v in d["tho_raw"].dropna().unique():
            toks = [t for t in re.split(r"[\s.]+", fold(v)) if t]
            cand = last.get(toks[-1], []) if toks else []
            if len(cand) == 1:
                mp[v] = cand[0]
            else:
                ok = False
        d["tho"] = d["tho_raw"].map(mp)
        if d["tho"].isna().any():
            ok = False
        sh.tho_ok = ok
        sh.last = last
        # việc ghi trong ô Làm gì
        d["jobs"] = [_jobs(v) for v in d["lg"].fillna("")]
        sh.df = d.reset_index(drop=True)
        sh.lo, sh.hi = d["ngay"].min(), d["ngay"].max()
        return sh
    return None


def _jobs(text):
    js = set()
    for it in re.split(r"[,;+/&]", fold(text)):
        it = it.strip()
        if not it or re.match(r"^(bh|bao hanh)\b", it):
            continue
        if re.search(r"\blap\b|hop so", it):
            js.add("lap")
        elif re.search(r"nhot|castrol|motul|shell", it):
            js.add("may")
        if re.search(r"\bva\b", it):
            js.add("va")
        elif re.search(r"\bruot\b", it):
            js.add("ruot")
        if re.search(r"\bdien\b", it) and not re.search(r"bong|binh|ac quy|\bden\b", it):
            js.add("dien")
    return js


# ---------------------------------------------------------------- kỳ
def _span(periods, f, sh):
    if not periods or any(p.kind == "snapshot" for p in periods):
        return None
    if len(periods) == 1:
        p = periods[0]
        st = pd.Timestamp(p.start) if p.start is not None else sh.lo
        en = pd.Timestamp(p.end)
    elif len(periods) == 2 and re.search(r"\bden\b", f) and all(q.start is not None for q in periods):
        st = min(pd.Timestamp(q.start) for q in periods)
        en = max(pd.Timestamp(q.end) for q in periods)
        p = None
    else:
        return None
    st, en = st.normalize(), en.normalize()
    if en < sh.lo or st > sh.hi:
        return None
    return st, en, p


_WD = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tu": 2, "4": 2, "nam": 3, "5": 3, "sau": 4, "6": 4, "bay": 5, "7": 5}


def _tho_named(f, sh):
    """-> (tên đầy đủ | None, có_nhắc_thợ_cụ_thể). Nhập nhằng -> ('?', True)."""
    hit = set()
    for full in sh.roster:
        ff = fold(full)
        toks = ff.split()
        if re.search(rf"(?<![a-z0-9]){re.escape(ff)}(?![a-z0-9])", f):
            hit.add(full)
            continue
        l = toks[-1]
        if re.search(rf"\btho (?:a |anh )?{re.escape(l)}\b", f):
            hit.add(full)
        elif len(l) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(l)}(?![a-z0-9])", f):
            hit.add(full)
    if len(hit) == 1:
        return next(iter(hit)), True
    if len(hit) > 1:
        return "?", True
    return None, False


def _dong_named(f, sh):
    """-> (root | None, có_nhắc). Nhắc nhập nhằng -> ('?', True)"""
    if not sh.dong_ok:
        return None, False
    ms = []
    for r, ks in sh.dong_keys.items():
        for k in ks:
            if len(k) < 2:
                continue
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", f):
                ms.append((r, m.start(), m.end()))
    ms = [m for m in ms if not any(o != m and o[1] <= m[1] and o[2] >= m[2] and (o[2] - o[1]) > (m[2] - m[1]) for o in ms)]
    roots = {m[0] for m in ms}
    if not roots:
        return None, False
    if len(roots) > 1:
        return "?", True
    return next(iter(roots)), True


def _job_named(f):
    js = set()
    if re.search(r"\blap\b|hop so", f):
        js.add("lap")
    elif re.search(r"nhot", f):
        js.add("may")
    elif re.search(r"\bdau\b", f):
        js.add("?")
    if re.search(r"\bva\b", f):
        js.add("va")
    elif re.search(r"\bruot\b", f):
        js.add("ruot")
    if re.search(r"\bdien\b", f) and not re.search(r"bong|binh|ac quy|\bden\b", f):
        js.add("dien")
    if re.search(r"bong den|ac quy|thay binh|binh gs|bugi|bu gi|bo thang|phanh|curoa|cu roa|kim phun|loc gio|bao duong|rua xe|chinh sen|can sen|tang sen|lam noi|ve sinh noi|vs noi|\bbdtq\b", f):
        js.add("?")
    return js


# ---------------------------------------------------------------- giải
def solve(q, f, periods, biz, rulebook):
    if not _gate(rulebook):
        return None
    sh = _prep(biz)
    if sh is None:
        return None
    try:
        return _solve(q, f, periods, sh, rulebook)
    except Exception:
        return None


def _solve(q, f, periods, sh, rb):
    sp = _span(periods, f, sh)
    if sp is None:
        return None
    st, en, p = sp
    d = sh.df
    sel = d[(d["ngay"] >= st) & (d["ngay"] <= en)]
    tho_full, tho_hit = _tho_named(f, sh) if sh.roster else (None, False)
    if tho_full == "?":
        return None
    dong_root, dong_hit = _dong_named(f, sh)
    if dong_root == "?":
        return None
    jobs = _job_named(f)
    if "?" in jobs or len(jobs) > 1:
        return None
    job = next(iter(jobs)) if jobs else None
    has_tho_word = bool(re.search(r"\btho\b", f))
    if tho_hit and not sh.tho_ok:
        return None
    if tho_hit:
        sel_t = sel[sel["tho"] == tho_full]
    else:
        sel_t = sel
    sel_f = sel_t[sel_t["dong"] == dong_root] if dong_hit else sel_t

    ratio = re.search(r"ti le|ty le|phan tram|%|chiem", f)
    bhw = "bao hanh" in f
    cnt_w = re.search(r"bao nhieu|\bmay\b|\bmi\b|\bdem\b|tong so|\bso\b|nhieu", f)
    ask_unit = re.search(r"\bphieu\b|\bluot\b|\bnuot\b", f)
    avg = re.search(r"trung binh|binh quan|\btb\b", f)
    money = re.search(r"\btien\b|doanh thu|dthu|thu ve|tien hang", f)
    cnt_unit = re.search(r"(bao nhieu|\bmay\b|\bmi\b|\bdem\b|tong so|\bso)\s+(\w+\s+)?(phieu|luot|nuot)|(phieu|luot|nuot)( xe)? (bao nhieu|\bmay\b)", f)
    quen = re.search(r"khach quen|(tu|tren|hon|qua)\s+(\d+)\s+lan|\bxe\b.*nhieu lan", f)

    # ---- % thay đổi doanh thu so với kỳ trước
    if re.search(r"phan tram|%", f) and re.search(r"doanh thu|dthu|tien ban|ban ra", f) and re.search(r"so voi|\bvoi\b|\bso (thang|quy|ky)", f) \
            and not (has_tho_word or tho_hit or dong_hit or job or bhw):
        if re.search(r"nam ngoai|cung ky|nam truoc", f) or p is None:
            return None
        if p.kind == "month" and re.search(r"thang (lien )?truoc|truoc do|lien truoc", f):
            pst = (st - pd.Timedelta(days=1)).replace(day=1)
            pen = st - pd.Timedelta(days=1)
        elif p.kind == "quarter" and re.search(r"quy (lien )?truoc|truoc do|lien truoc", f):
            pen = st - pd.Timedelta(days=1)
            pst = (st - pd.DateOffset(months=3))
        else:
            return None
        if pst < sh.lo:
            return None
        cur = float(sel["tien"].sum())
        prev = float(d[(d["ngay"] >= pst) & (d["ngay"] <= pen)]["tien"].sum())
        if prev <= 0:
            return None
        return {"type": "percent", "answer": _r2((cur - prev) / prev * 100)}, \
            f"Doanh thu kỳ {cur:,.0f} so kỳ trước {prev:,.0f} (tổng Tiền các phiếu)"

    # ---- tỉ lệ bảo hành
    if bhw and ratio and _co_luat(rb, "bh") and not (has_tho_word or tho_hit or dong_hit or job):
        if re.search(r"\bti le\b|\bty le\b", f) or re.search(r"thu tien", f):
            den = int(sel["tc"].sum())
            if den == 0:
                return None
            num = int(sel["bh"].sum())
            return {"type": "percent", "answer": _r2(num / den * 100)}, \
                f"Tỉ lệ bảo hành = {num} phiếu bảo hành / {den} phiếu có thu tiền"
        return None

    # ---- trung bình mỗi ngày thứ X
    mwd = re.search(r"(?:moi|cac|vao|ngay|tung)\s+(?:ngay\s+)?thu\s+(hai|ba|tu|nam|sau|bay|[2-7])\b", f)
    if mwd and avg and _co_luat(rb, "thu"):
        if has_tho_word or tho_hit or dong_hit or job or bhw or p is None:
            return None
        wd = _WD[mwd.group(1)]
        days = pd.date_range(max(st, sh.lo), en)
        n = int((days.weekday == wd).sum())
        if n == 0:
            return None
        sub = sel[sel["ngay"].dt.weekday == wd]
        if ask_unit and not re.search(r"doanh thu|dthu|\btien\b", f):
            return {"type": "number", "answer": _r2(len(sub) / n)}, f"{len(sub)} phiếu / {n} ngày thứ đó trong kỳ"
        if re.search(r"doanh thu|dthu", f) and not ask_unit:
            return {"type": "number", "answer": _rhu(sub["tien"].sum() / n)}, f"Tiền {sub['tien'].sum():,.0f} / {n} ngày"
        return None

    # ---- trung bình một phiếu (chia số phiếu có thu tiền)
    if avg and ask_unit and money and _co_luat(rb, "tb") and not re.search(r"\bnao\b|\bai\b|nhat", f):
        if has_tho_word and not tho_hit:
            return None
        den = int(sel_f["tc"].sum())
        if den == 0:
            return None
        return {"type": "number", "answer": _rhu(sel_f["tien"].sum() / den)}, \
            f"Tiền {sel_f['tien'].sum():,.0f} / {den} phiếu có thu tiền"

    # ---- khách quen / xe tới nhiều lần (đếm phiếu có thu tiền theo biển số)
    if quen and re.search(r"\bxe\b", f) and _co_luat(rb, "quen") and sh.plate_ok and not (has_tho_word or tho_hit or job):
        m = re.search(r"(tu|tren|hon|qua)\s+(\d+)\s+lan(\s+tro\s+len)?", f)
        if m:
            n = int(m.group(2))
            thr = n if (m.group(1) == "tu") else n + 1
        elif "khach quen" in f:
            thr = 2
        else:
            return None
        if re.search(r"\bnao\b|liet ke|danh sach|nhung xe", f):
            return None
        if not re.search(r"bao nhieu|\bmay\b|\bdem\b|\bso\b", f):
            return None
        s2 = sel_f[sel_f["tc"]]
        vc = s2.groupby("plate").size()
        return {"type": "number", "answer": int((vc >= thr).sum())}, \
            f"Số biển số có >= {thr} phiếu có thu tiền (không tính bảo hành) trong kỳ"

    # ---- xe nào tới nhiều nhất (biển số)
    if re.search(r"\bxe nao\b", f) and re.search(r"nhieu (lan|phieu|luot)? ?nhat|nhieu nhat", f) and sh.plate_ok \
            and not (money or has_tho_word or tho_hit or job or dong_hit or bhw) and _co_luat(rb, "quen"):
        s2 = sel[sel["tc"]]
        vc = s2.groupby("plate").size()
        if vc.empty:
            return None
        top = vc.reset_index(name="n").sort_values(["n", "plate"], ascending=[False, True]).iloc[0]["plate"]
        txt = s2[s2["plate"] == top]["plate_txt"].iloc[0]
        return {"type": "string", "answer": txt}, "Biển số có nhiều phiếu có thu tiền nhất"

    # ---- thợ xếp hạng (hoa hồng / lượt / tiền / ngày làm)
    if has_tho_word and not tho_hit and re.search(r"\bnao\b|\bai\b|\bmo\b", f) and re.search(r"nhat", f) \
            and not (dong_hit or job) and sh.tho_ok:
        if re.search(r"hoa hong|hue hong", f):
            if p is None or p.kind not in ("month", "month_to_date"):
                return None
            hh = _hoa_hong(sh, sel)
            if not hh or not re.search(r"cao nhat|nhieu nhat", f):
                return None
            top = sorted(hh.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            return {"type": "string", "answer": top}, "Thợ có hoa hồng cao nhất (tỉ lệ x tổng Tiền các phiếu)"
        if re.search(r"cham chi|nhiet tinh|sieng nang", f):
            if not _co_luat(rb, "chamchi"):
                return None
            g = sel.groupby("tho")["ngay"].nunique()
            tag = "số ngày có phiếu"
        elif re.search(r"nhieu tien|doanh thu|dthu|thu ve|ban (duoc|ra)|lam ra|tien (cao|nhieu)|cao nhat", f) and money:
            g = sel.groupby("tho")["tien"].sum()
            tag = "tổng Tiền các phiếu"
        elif re.search(r"nhieu (luot|phieu)|luot xe|lam nhieu nhat|nhan nhieu|sua nhieu", f) and not money:
            if not _co_luat(rb, "tho"):
                return None
            g = sel.groupby("tho").size()
            tag = "số lượt xe (kể cả bảo hành)"
        else:
            return None
        if g.empty:
            return None
        top = g.reset_index(name="v").sort_values(["v", "tho"], ascending=[False, True]).iloc[0]["tho"]
        return {"type": "string", "answer": str(top)}, f"Thợ đứng đầu theo {tag}"

    # ---- hoa hồng
    if re.search(r"hoa hong|hue hong", f):
        if p is None or p.kind not in ("month", "month_to_date") or not sh.tho_ok or dong_hit or job:
            return None
        hh = _hoa_hong(sh, sel)
        if not hh:
            return None
        if tho_hit:
            if tho_full not in hh or not re.search(r"bao nhieu|\bmay\b|tong|lanh|\bnhan\b|duoc", f):
                return None
            return {"type": "number", "answer": hh[tho_full]}, "Hoa hồng = % x tổng Tiền các phiếu thợ làm trong tháng"
        if re.search(r"bang|danh sach|tung tho|moi tho|cac tho|tat ca", f) and re.search(r"giam dan|cao xuong|nhieu .*len tren|xep|nhieu tien", f):
            rows = [[k, v] for k, v in sorted(hh.items(), key=lambda kv: (-kv[1], kv[0]))]
            return {"type": "table", "answer": {"columns": ["Thợ", "Hoa hồng"], "rows": rows}}, \
                "Bảng hoa hồng từng thợ: % x tổng Tiền, giảm dần"
        return None

    # ---- dòng xe nhiều tiền nhất / nhiều lượt nhất
    if re.search(r"dong xe", f) and re.search(r"\bnao\b|\bmo\b|\bgi\b", f) and re.search(r"nhat", f) and sh.dong_ok \
            and not (has_tho_word or tho_hit or job or dong_hit):
        if re.search(r"nhieu tien|doanh thu|dthu|tien sua", f):
            g = sel.groupby("dong")["tien"].sum()
        elif re.search(r"nhieu (luot|phieu)", f):
            g = sel.groupby("dong").size()
        else:
            return None
        if g.empty:
            return None
        top = g.reset_index(name="v").sort_values(["v", "dong"], ascending=[False, True]).iloc[0]["dong"]
        return {"type": "string", "answer": sh.dong_disp[top]}, "Dòng xe đứng đầu (gộp các cách ghi cùng biển số)"

    # ---- ngày nhiều tiền / đắt khách nhất
    if re.search(r"\bngay (nao|ma)\b", f) and re.search(r"nhat", f) and not (has_tho_word or tho_hit or job or dong_hit or bhw):
        if re.search(r"dat khach|nhieu (phieu|luot)", f):
            g = sel.groupby("ngay").size()
        elif re.search(r"nhieu tien|doanh thu|dthu|thu ve|ban (duoc|ra)", f):
            g = sel.groupby("ngay")["tien"].sum()
        else:
            return None
        if g.empty:
            return None
        top = g.reset_index(name="v").sort_values(["v", "ngay"], ascending=[False, True]).iloc[0]["ngay"]
        return {"type": "date", "answer": top.strftime("%Y-%m-%d")}, "Ngày có nhiều tiền/phiếu nhất trong kỳ"

    # ---- số chiếc xe khác nhau (biển số)
    if (re.search(r"\bchiec\b|\bxe\b.*khac nhau|(bao nhieu|\bmay\b|\bdem\b|\bso) (chiec )?xe\b", f)
            and not quen and not ask_unit and not money and not (has_tho_word or tho_hit or job)
            and sh.plate_ok and _co_luat(rb, "xe") and not re.search(r"\bnao\b|nhat|trung binh", f)):
        n = int(sel_f["plate"].nunique())
        return {"type": "number", "answer": n}, "Số biển số khác nhau trong kỳ (tính cả xe chỉ tới bảo hành)"

    # ---- đếm phiếu / lượt (tiệm, thợ, bảo hành, việc)
    if ask_unit and (cnt_unit or (cnt_w and not money)) and not avg and not ratio and not re.search(r"\bnao\b|\bai\b|nhat", f) \
            and not (money and not re.search(r"thu tien", f)):
        if has_tho_word and not tho_hit:
            return None
        if job:
            key = {"may": "nhot", "lap": "nhot", "va": "ruot", "ruot": "ruot", "dien": "dien"}[job]
            if not (_co_luat(rb, "viec") and _co_luat(rb, key)):
                return None
            if tho_hit or dong_hit:
                pass
            n = int(sel_f[(~sel_f["bh"]) & sel_f["jobs"].map(lambda s: job in s)].shape[0])
            return {"type": "number", "answer": n}, f"Số phiếu (không bảo hành) có việc '{job}' trong ô Làm gì"
        if bhw and not re.search(r"khong (tinh|ke|gom|phai)|tru |ngoai ", f):
            n = int(sel_f["bh"].sum())
            return {"type": "number", "answer": n}, "Số phiếu bảo hành (có ngày trong kỳ)"
        if re.search(r"thu tien|khong (tinh|ke|gom|phai).*bao hanh|tru .*bao hanh|ngoai .*bao hanh", f):
            return {"type": "number", "answer": int(sel_f["tc"].sum())}, "Số phiếu có thu tiền (không tính bảo hành)"
        if bhw:
            return None
        if not _co_luat(rb, "luot"):
            return None
        return {"type": "number", "answer": int(len(sel_f))}, "Số phiếu/lượt xe: mỗi dòng sổ là một phiếu, kể cả bảo hành"

    # ---- doanh thu / tiền
    if money and not re.search(r"\bnao\b|\bai\b|nhat|trung binh|%|phan tram|so voi|\bso (thang|quy|ky)|moi thu|bao hanh|phieu thu tien", f) \
            and re.search(r"bao nhieu|\bmay\b|tong|duoc|\bla\b", f) and not job and not dong_hit and not cnt_unit:
        if has_tho_word and not tho_hit:
            return None
        return {"type": "number", "answer": _rhu(sel_f["tien"].sum())}, "Doanh thu = tổng Tiền các phiếu trong kỳ"
    return None


def _hoa_hong(sh, sel):
    out = {}
    for full, pct in sh.roster.items():
        if pct is None:
            return None
        s = sel[sel["tho"] == full]
        if len(s) == 0:
            continue
        out[full] = _rhu(float(s["tien"].sum()) * pct / 100.0)
    return out
