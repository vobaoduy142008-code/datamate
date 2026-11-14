"""Phụ thêm: công thức cố định cho các câu LOẠI THƯỜNG mà ct_coban / ct_tyle / ct_top từ chối vì cách hỏi lạ
(doanh thu / số đơn / TB mỗi đơn / % tăng giảm / tỉ trọng / top / nhóm hàng / kênh / thanh toán / nhiều tháng / đơn vị bán).

solve(q, f, periods, biz, rulebook) -> (answer_dict, mo_ta) | None.
Nguyên tắc: chỉ nhận câu khi MỌI từ còn lại sau khi bỏ cụm kỳ, tên mặt hàng / nhóm / kênh, cụm ý định đều vô hại.
Không chắc -> None. Mọi định nghĩa lấy theo rulebook (chỉ bật khi rulebook đang chạy còn ghi đúng định nghĩa)."""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from loader import fold
from periods import find_periods, Calendar, Period, previous

# ------------------------------------------------------------------ luật rulebook
_LUAT = {
    "dt": "tong thanh tien cac dong hang",
    "dt2": "don khong huy va khong tra hang",
    "don": "chi tinh don khong huy khong tra",
    "tb": "gia tri trung binh mot don",
    "pct": "phan tram thay doi tang truong so voi ky truoc",
    "huy": "phan tram don huy trong moi don dat trong ky",
    "chay": "theo so luong quy ve don vi co ban",
    "phi": "phi phuc vu khong phai mat hang",
    "hoa": "bang nhau o tieu chi cau hoi thi xep theo ma tang dan",
    "don_vi": "co don vi co ban la",
    "tt_nhom": "ti trong cua nhom chia cho doanh thu ca shop cung ky",
    "loi": "theo tong lai gop cua mat hang",
    "gv": "gia von cua moi dong ban bang so luong nhan voi gia von ghi tren dong ban",
    "nv": "nhan vien ban gioi nhat theo doanh thu",
    "khach": "moi ma khach la mot khach",
    "khach_tien": "khach mua nhieu tien nhat xep theo doanh thu",
    "khach_mua": "khach da mua trong ky la khach co it nhat mot don",
    "ca": "ca sang 06 00 13 59 ca chieu 14 00 21 59",
    "ke_don": "ti le ke don tinh theo doanh thu",
    "unans": "hoan toan ngoai khoang ngay co du lieu",
}


def _rbn(rulebook):
    if rulebook is None:
        return None
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook)))


def _co(rb, *keys):
    return rb is not None and all(_LUAT[k] in rb for k in keys)


# ------------------------------------------------------------------ ngữ cảnh (hôm nay, lịch âm)
CTX = None


def _ctx():
    """as_of_date của câu đang chạy + Calendar, lấy từ khung gọi (harness / agent)."""
    if CTX is not None:                      # agent đặt sẵn (as_of_date, Calendar) trước khi gọi solve
        return CTX
    fr = sys._getframe(2)
    as_of = cal = None
    for _ in range(14):
        if fr is None:
            break
        L = fr.f_locals
        if as_of is None:
            for v in list(L.values()):
                if isinstance(v, dict) and "as_of_date" in v:
                    as_of = v["as_of_date"]
                    break
        if cal is None and isinstance(L.get("cal"), Calendar):
            cal = L["cal"]
        fr = fr.f_back
    return as_of, cal


# ------------------------------------------------------------------ chuẩn hoá thêm cho câu (đã bỏ dấu)
_MON = r"(?:thang|th|t)"


def _prep(f):
    g = " " + re.sub(r"\s+", " ", f.lower()) + " "
    g = g.replace("≥", " >= ")
    g = re.sub(r"[^\w/%>=.,\-+ ]+", " ", g)
    g = re.sub(r"\b(chong|trog|trg|trong)\b", "trong", g)
    g = re.sub(r"\bo ?(?:do|der|dow|de)\b|\borders?\b", "don", g)
    g = re.sub(r"\b(?:dh|dhang)\b", "don hang", g)
    g = re.sub(r"\btog cog\b|\btong cog\b", "tong cong", g)
    g = re.sub(r"\bnhan (?:dien|zien|vien)\b", "nhan vien", g)
    g = re.sub(r"\b(may|bao nhieu|bn|bnhiu) bua\b", r"\1 ngay", g)
    g = re.sub(r"\bmi (don|ngay|mat|khach)\b", r"may \1", g)
    g = re.sub(r"\bbi[nh]{1,2} quan\b|\bbq\b|\btrung bi[nh]{1,2}\b|\btb\b", "binh quan", g)
    g = re.sub(r"\bchen\b", "tren", g)
    g = re.sub(r"\b(?:bnhui|bnhieu|bnhiu|bnhieuu|bao nhieuu)\b", "bao nhieu", g)
    g = re.sub(r"\bbao nhieu ten\b", "bao nhieu tien", g)
    g = re.sub(r"\bbna\b", "ban", g)
    g = re.sub(r"\bwan\b", "quan", g)
    g = re.sub(r"\bnhieu[fsrxj]\b", "nhieu", g)
    g = re.sub(r"\bkua\b", "cua", g)
    g = re.sub(r"\bnhut\b", "nhat", g)
    g = re.sub(r"\b(?:muon |dinh )?tang qu?a\b(?: khach quen)?", " ", g)
    g = re.sub(r"\broo?if\b", "roi", g)
    g = re.sub(r"\bmi (?=%|phan tram)", "may ", g)
    g = re.sub(r"\bthang (mot|hai|ba|bon|bay|tam|chin|muoi)\b(?! (?:nam|nay|ngoai|truoc|roi))", lambda m: "thang " + str({"mot":1,"hai":2,"ba":3,"bon":4,"bay":7,"tam":8,"chin":9,"muoi":10}[m.group(1)]), g)
    g = re.sub(r"\bduco\b", "duoc", g)
    g = re.sub(r"\bcko\b", "cho", g)
    g = re.sub(r"\bngay ?nao\b", "ngay nao", g)
    g = re.sub(r"\bbao nhe?iu\b|\bbao nheiu\b", "bao nhieu", g)
    g = re.sub(r"\bbao nhieu tieng\b", "bao nhieu tien", g)
    g = re.sub(r"\blan nuot\b", "lan luot", g)
    g = re.sub(r"\bso doi\b", "so voi", g)
    g = re.sub(r"\blne\b", "len", g)
    g = re.sub(r"\bmang ve (?=bao nhieu|doanh thu|dthu|may %)", " ", g)
    g = _expand_months(g)
    g = re.sub(r"\b(it nhat|toi thieu|tu) (\d{1,3}) dong\b(?! ?(?:tro len))", r"\1 \2 don", g)
    g = re.sub(r"\b(it nhat|toi thieu|tu) (\d{1,3}) dong(?= tro len)", r"\1 \2 don", g)
    g = re.sub(r"\b(may|bao nhieu) dong (?=tien hang|co |dat )", r"\1 don ", g)
    g = re.sub(r"\b(nhieu|it)(don|tien|lan)\b", r"\1 \2", g)
    g = re.sub(r"(?<![a-z])(cao|nhieu|nhiu|lon|thap|it) ?(?:nhat|nhut)\b", lambda m: ("nhieu" if m.group(1) in ("nhiu",) else m.group(1)) + " nhat", g)
    g = re.sub(r"\bban chay nhut\b", "ban chay nhat", g)
    g = re.sub(r"\bxi o di\b", "cod", g)
    g = re.sub(r"\b(?:tinh |ke |gom )?(?:luon|ca) (?:hom nay|hnay)(?: thi)?\b", " ", g)
    g = re.sub(r"\bcung thang nam (truoc|ngoai)\b", "cung ky nam truoc", g)
    g = re.sub(r"\bcung so ngay (?:cua )?(?:dau )?thang (?:truoc|roi)\b", "cung ky thang truoc", g)
    g = re.sub(r"\bmung 1\b(?: thang nay)?", "dau thang nay", g)
    g = re.sub(r"\b(?:tu )?dau thang(?: nay)?,? (?:den|toi|cho den|tinh den|tinh toi)(?: het)? (?:hom nay|nay|bay gio|gio|hien tai)\b", " tu dau thang nay den het hom nay ", g)
    g = re.sub(r"(?<!\d )\b(thang nay|tuan nay|nam nay|quy nay),? (?:tinh )?(?:den|toi)(?: het)? (?:hom nay|nay|bay gio|gio|hien tai)\b", r"\1", g)
    g = re.sub(r"\bq ?([1-4]) ?[/-] ?(\d{4})\b", r"quy \1/\2", g)
    g = re.sub(r"\bquy ([1-4]) (20\d\d)\b", r"quy \1/\2", g)
    g = re.sub(r"\b" + _MON + r" ?(\d{1,2})(?=\s*[/,\-]|\s|$)", r"thang \1", g)
    g = re.sub(r"\bthang (\d{1,2}) ?([/-]) ?(\d{2})(?!\d)", r"thang \1/20\3", g)
    g = re.sub(r"\bthang (\d{1,2}) ?- ?(\d{4})\b", r"thang \1/\2", g)
    g = re.sub(r"(?<![\w/.\-])(?<!quy )(\d{1,2}) ?- ?(\d{4})\b", r"thang \1/\2", g)
    g = re.sub(r"(?<![\w/\-])(?<!quy )(\d{1,2})/(\d{4})\b", r"thang \1/\2", g)
    g = re.sub(r"(?<![\w/\-.])(0?[1-9]|1[0-2])/(2\d)(?![\d/])", r"thang \1/20\2", g)
    g = re.sub(r"\bthang thang\b", "thang", g)
    g = _fill_years(g)
    g = _nam_ngoai(g)
    g = _n_thang_lien(g)
    g = re.sub(r"\s+", " ", g)
    return " " + g.strip() + " "


def _fill_years(g):
    """'thang 7, thang 8, thang 9/2025' -> các tháng trước lấy năm của tháng cuối dãy."""
    ms = list(re.finditer(r"thang (\d{1,2})(?:/(\d{4}))?(?![\d/])", g))
    ys = [m.group(2) for m in ms]
    for i in range(len(ms) - 2, -1, -1):
        if ys[i] is None and ys[i + 1] is not None:
            gap = g[ms[i].end():ms[i + 1].start()]
            if re.fullmatch(r"\s*(?:,|va|roi|ca|den|toi|-)?\s*", gap):
                ys[i] = ys[i + 1]
    out, last = [], 0
    for m, y in zip(ms, ys):
        out.append(g[last:m.start()])
        out.append(f"thang {m.group(1)}/{y}" if y else m.group(0))
        last = m.end()
    out.append(g[last:])
    return "".join(out)


def _expand_months(g):
    """'thang 10, 11, 12 nam 2025' -> 'thang 10/2025, thang 11/2025, thang 12/2025'."""
    def f(m):
        nums = re.findall(r"\d{1,2}", m.group(1))
        return ", ".join(f"thang {n}/{m.group(2)}" for n in nums)
    return re.sub(r"\bthang ((?:\d{1,2}\s*(?:,|va)\s*)+\d{1,2}) (?:nam |/)(\d{4})\b", f, g)


def _nam_ngoai(g):
    """'thang 5/2026 so voi thang 5 nam ngoai' -> tháng 5 của năm liền trước tháng đã nêu (cùng tháng)."""
    m0 = re.search(r"thang (\d{1,2})/(\d{4})", g)
    if not m0:
        return g
    return re.sub(r"\bthang (\d{1,2}) nam (?:ngoai|truoc)\b",
                  lambda m: f"thang {m.group(1)}/{int(m0.group(2)) - 1}" if m.group(1) == m0.group(1) else m.group(0), g)


def _n_thang_lien(g):
    m = re.search(r"\b(\d) thang (?:lien|lien tiep|lien nhau|gan nhat|vua qua)\b.*?(?:toi|den|tinh toi|tinh den) thang (\d{1,2})/(\d{4})", g)
    if not m or re.search(r"\btu thang \d", g):
        return g
    n, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    t = y * 12 + mo - 1 - (n - 1)
    sy, sm = t // 12, t % 12 + 1
    return g[:m.start()] + f" tu thang {sm}/{sy} den thang {mo}/{y} " + g[m.end():]


# ------------------------------------------------------------------ tiện ích
def _tien(v):
    return int(Decimal(repr(round(float(v), 6))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _sl(v):
    v = round(float(v), 6)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


def _pct(num, den):
    if not den:
        return None
    return float(Decimal(repr(float(num) / float(den) * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _words(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


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


def _unans(why):
    return {"type": "unanswerable", "reason": why}


def _rev(x):
    return float(x.loc[x["tinh_dt"], "thanh_tien"].sum())


def _nord(x):
    return int(x.loc[x["tinh_dt"], "ma_don"].nunique())


class T:
    """Văn bản đang bóc dần; mỗi lần nhận một cụm thì thay bằng nhãn VIẾT HOA."""

    def __init__(s, g):
        s.g = " " + g.strip() + " "

    def take(s, pat, tag=" X "):
        m = re.search(pat, s.g)
        if m:
            s.g = s.g[:m.start()] + tag + s.g[m.end():]
        return m

    def has(s, pat):
        return re.search(pat, s.g)

    def toks(s):
        return re.findall(r"[a-z0-9%]+", re.sub(r"\b[A-Z]+\b", " ", s.g))


_TIME_PATS = [
    r"(?:\b(?:tu|den|toi|het|tinh|ke tu) )*(?:ngay )?\b\d{1,2} ?/ ?\d{1,2}(?: ?/ ?\d{2,4})?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:tu )?(?:thang|quy) \d{1,2}(?: ?/ ?\d{4}| nam \d{4}| nam (?:nay|ngoai|truoc))?\b",
    r"\bnam \d{4}\b",
    r"\b\d+ ngay (?:gan nhat|qua|vua qua|gan day|truoc|cuoi)\b",
    r"\b(?:hom|bua) (?:nay|qua|kia)\b",
    r"\b(?:tuan|thang|nam|quy) (?:nay|truoc|roi|ngoai|vua roi|vua qua|lien truoc|truoc do)\b",
    r"\b(?:tu |ke tu )?dau (?:nam|thang|tuan|quy)(?: nay)?\b",
    r"\b(?:den|toi|tinh den|tinh toi|cho den) (?:het )?(?:hom nay|nay|bay gio|gio|hien tai|bay chu)\b",
    r"\bcung ky(?: thang| nam| tuan| quy)?(?: truoc| ngoai)?\b",
    r"\b(?:dip |mua )?tet(?: nguyen dan| am lich)?(?: nam)?(?: \d{4}| nay| ngoai)?\b",
]


def _strip_time(t):
    n = 0
    for pat in _TIME_PATS:
        while t.take(pat, " TIME "):
            n += 1
            if n > 40:
                break
    return n


_SAFE = set("""cho minh mik toi tui em anh chi oi ad be e a ha ah nha nhe nhoa nghen hen hoi xiu nua di vay nhi z ne ak ay
la bao nhieu bn may so tong cong co duoc dc ra cua hang shop quan tiem sop tat ca da mot va ban mua thi ma thoi do kia
tien dong vnd d the nao hien tai nay giup gium jum jup dum dup xem coi check kiem tra biet cam on thanks
tu den toi het dau tinh trong vao luc o roi luon dem thang nam tuan quy ngay hom
chot len nhan dat lap rieng hi hy alo za zi nho mk day dza ua
""".split())


def _clean(t, extra=()):
    ok = _SAFE | set(extra)
    bad = [x for x in t.toks() if x not in ok]
    if bad:
        WHY.append("TOK:" + ",".join(bad[:4]))
        return False
    return True


WHY = []


# ------------------------------------------------------------------ meta của shop (tên mặt hàng, nhóm, kênh...)
_META = {}


def _meta(ban):
    m = _META.get("m")
    if m is not None and m[0] is ban:
        return m[1]
    d = {}
    nonfee = ban.loc[~ban["la_phi"]] if "la_phi" in ban else ban
    names = nonfee["ten_hang"].dropna().unique() if "ten_hang" in ban else []
    d["items"] = [(_words(fold(n)), n) for n in names if len(_words(fold(n))) >= 3]
    for c in ("nhom_hang", "kenh", "nhom_khach"):
        vals = ban[c].dropna().unique() if c in ban else []
        d[c] = [(_words(fold(str(v))), v) for v in vals if not str(v).startswith("(") and len(_words(fold(str(v)))) >= 2]
    _META["m"] = (ban, d)
    return d


def _find_vals(g, vals):
    """Các giá trị (key đã bỏ dấu) xuất hiện nguyên từ trong g -> [(đầu, cuối, key, giá trị gốc)] (None nếu chồng lấn nhập nhằng)."""
    gw = " " + _words(g) + " "
    hits = []
    for k, v in vals:
        for m in re.finditer(r"(?<=\s)" + re.escape(k) + r"(?=\s)", gw):
            hits.append((m.start(), m.end(), k, v))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, last = [], -1
    for s, e, k, v in hits:
        if s >= last:
            out.append((s, e, k, v))
            last = e
        elif e > last:
            return None
    return out


# ------------------------------------------------------------------ ngưỡng tiền / số đơn
def _num(s):
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        return float(s.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
        return float(s.replace(",", ""))
    if re.fullmatch(r"\d+[.,]\d+", s):
        return float(s.replace(",", "."))
    if s.isdigit():
        return float(s)
    return None


_MULT = {"k": 1e3, "nghin": 1e3, "ngan": 1e3, "tr": 1e6, "trieu": 1e6, "cu": 1e6, "d": 1.0, "dong": 1.0, "vnd": 1.0}
_NUMRE = r"\d{1,3}(?:\.\d{3})+|\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?"


def _thr(t):
    """Ngưỡng tiền 'từ X trở lên' (>=) hoặc 'trên X' (>). -> (op, giá trị) | None | 'BAD'."""
    pat = (r"(?:\b(tu|toi thieu|it nhat|>=|tren|hon|qua|vuot)\s*)?(" + _NUMRE + r")\s*(tr|trieu|cu|k|nghin|ngan|dong|vnd|d)?(\d+)?\b"
           r"(?:\s*(tro len|tro nen|tro di))?")
    for m in re.finditer(pat, t.g):
        pre, n, un, frac, tl = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if not (tl or pre in (">=", "toi thieu", "tren", "hon", "qua", "vuot")):
            continue
        if un is None and not tl:
            continue
        after = t.g[m.end():m.end() + 12]
        if un is None and re.match(r"\s*(don|ngay|lan|hoa don|bill|khach|nguoi|mat hang|san pham|mon)\b", after):
            continue
        v = _num(n)
        if v is None:
            return "BAD"
        if un in ("d", "dong", "vnd") and v < 1000:
            return "BAD"
        if un:
            if frac and un in ("tr", "trieu", "cu"):
                v = float(f"{int(v)}.{frac}") if float(v) == int(v) else v
            v *= _MULT[un]
        elif v < 1000:
            return "BAD"
        op = ">" if pre in ("tren", "hon", "qua", "vuot") else ">="
        if op == ">" and tl:
            op = ">="
        t.g = t.g[:m.start()] + " THR " + t.g[m.end():]
        return op, v
    return None


def _cnt_thr(t):
    """'ít nhất / từ N đơn(lần) trở lên'."""
    m = re.search(r"\b(?:it nhat|tu|>=|toi thieu)\s*(\d+)\s*(?:don|lan|hoa don)(?: hang)?(?: khac nhau)?(?:\s*tro len)?\b", t.g) \
        or re.search(r"\b(\d+)\s*(?:don|lan|hoa don)\s*tro len\b", t.g)
    if m:
        t.g = t.g[:m.start()] + " CTHR " + t.g[m.end():]
        return int(m.group(1))
    m = re.search(r"\b(?:hon|tren|qua)\s*(\d+)\s*(?:don|lan)\b", t.g)
    if m:
        t.g = t.g[:m.start()] + " CTHRG " + t.g[m.end():]
        return int(m.group(1)) + 1
    return None


# ------------------------------------------------------------------ bộ lọc: nhóm / kênh / nhóm khách / mặt hàng / thanh toán
_PAYC = [("tm", r"tien mat|\btm\b|cash"), ("ck", r"chuyen khoan|\bck\b|transfer|bank"),
         ("vi", r"vi dien tu|e ?wallet|momo"), ("cod", r"\bcod\b|thu ho"), ("no", r"ghi no|debt|cong no")]
_PAYW = {"tm": r"tien mat", "ck": r"chuyen khoan", "vi": r"vi dien tu|momo", "cod": r"cod|thu ho", "no": r"ghi no|mua chiu"}


def _payc(v):
    t = fold(str(v))
    for k, pat in _PAYC:
        if re.search(pat, t):
            return k
    return None


_PAYCACHE = {}


def _pay_ids(biz, ban):
    """{lớp thanh toán: tập mã đơn} theo hình thức GHI TRÊN ĐƠN; None nếu shop không có / không rõ."""
    c = _PAYCACHE.get("p")
    if c is not None and c[0] is biz:
        return c[1]
    res = None
    try:
        res = _pay_ids0(biz, ban)
    except Exception:  # noqa: BLE001
        res = None
    _PAYCACHE["p"] = (biz, res)
    return res


def _classes(values):
    cls = {v: _payc(v) for v in values}
    if not cls or any(c is None for c in cls.values()) or len(set(cls.values())) != len(cls):
        return None
    return cls


def _pay_ids0(biz, ban):
    nd = getattr(biz, "no_don", None)
    if nd is not None and "pttt" in nd and nd["pttt"].notna().sum() > 0:
        cls = _classes(nd["pttt"].dropna().unique())
        if cls is None:
            return None
        return {k: set(nd.loc[nd["pttt"].map(lambda v: cls.get(v) == k), "ma_don"].astype(str)) for k in set(cls.values())}
    ids = set(ban["ma_don"].astype(str))
    best = None
    for tn, df in getattr(biz, "tables", {}).items():
        if re.search(r"phieu thu|so quy|thu tien|nhap", fold(str(tn))):
            continue
        for c in df.columns:
            if not re.fullmatch(r"(hinh thuc thanh toan|phuong thuc thanh toan|thanh toan bang|pttt|hinh thuc tt)", fold(str(c)).replace("_", " ").strip()):
                continue
            if df[c].notna().sum() == 0 or df[c].dropna().nunique() > 12:
                continue
            for ic in df.columns:
                if ic == c:
                    continue
                col = df[ic]
                if col.dtype.kind in "fiu" or col.nunique() < 0.5 * len(ids):
                    continue
                cov = col.astype(str).isin(ids).mean()
                if cov < 0.95:
                    continue
                d = df[[ic, c]].dropna(subset=[ic]).copy()
                d[ic] = d[ic].astype(str)
                d = d[d[ic].isin(ids)]
                g = d.groupby(ic)[c].agg(lambda x: x.dropna().unique().tolist())
                if (g.map(len) > 1).any():
                    continue
                cover = (g.map(len) == 1).sum() / max(1, len(ids))
                if cover < 0.98:
                    continue
                if best is None or cover > best[0]:
                    best = (cover, g.map(lambda x: x[0] if x else None))
    if best is None:
        return None
    mp = best[1].dropna()
    cls = _classes(mp.unique())
    if cls is None:
        return None
    return {k: set(mp.index[mp.map(lambda v: cls.get(v) == k)]) for k in set(cls.values())}


def _repl_words(t, key, tag, prefix=""):
    pat = prefix + r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(w) for w in key.split()) + r"(?![a-z0-9])"
    t.g, n = re.subn(pat, f" {tag} ", t.g)
    return n


def _filters(t, ban, biz, F):
    """Bóc bộ lọc khỏi t. Trả False nếu nhập nhằng / không dùng được."""
    M = _meta(ban)
    F.update(group=None, kenh=[], nkh=None, item=None, kedon=False, pay=[], ca=None)
    if M["nhom_hang"] and re.search(r"\bnhom\b", t.g):
        gw = " " + _words(t.g) + " "
        hits = [(k, v) for k, v in M["nhom_hang"] if re.search(r"(?<=\snhom )(?:hang )?" + re.escape(k) + r"(?=\s)", gw)
                or re.search(r"(?<=\snhom hang )" + re.escape(k) + r"(?=\s)", gw)]
        if hits:
            hits.sort(key=lambda h: -len(h[0]))
            k, v = hits[0]
            if len(hits) > 1 and not all(k2 in k for k2, _ in hits[1:]):
                return False
            if any(k == k2 for k2, _ in M["nhom_khach"]):
                return False
            if _repl_words(t, k, "GROUP", r"\bnhom (?:hang )?") != 1:
                return False
            F["group"] = v
    if M["nhom_khach"] and re.search(r"\bnhom\b", t.g):
        gw = " " + _words(t.g) + " "
        hits = [(k, v) for k, v in M["nhom_khach"]
                if re.search(r"(?<=\snhom )(?:khach )?(?:hang )?" + re.escape(k) + r"(?=\s)", gw)
                or re.search(r"(?<=\skhach nhom )" + re.escape(k) + r"(?=\s)", gw)]
        if hits:
            hits.sort(key=lambda h: -len(h[0]))
            k, v = hits[0]
            if F["group"] is not None:
                return False
            pre = r"(?:\bkhach(?: hang)? )?\bnhom (?:khach(?: hang)? )?|\bkhach(?: hang)? nhom "
            n = _repl_words(t, k, "NKH", r"(?:" + pre + r")")
            if n != 1:
                return False
            F["nkh"] = v
    if M["kenh"]:
        h = _find_vals(t.g, M["kenh"])
        if h is None:
            return False
        for s, e, k, v in h:
            if _repl_words(t, k, "KENH", r"(?:\bdon(?: hang)? )?(?:\bban )?(?:\b(?:qua|tai|kenh) )?") < 1:
                return False
        F["kenh"] = [v for s, e, k, v in h]
    gw = " " + _words(t.g) + " "
    hits = {}
    for k, n in M["items"]:
        if f" {k} " in gw:
            hits.setdefault(k, []).append(n)
    if hits:
        keys = sorted(hits, key=len, reverse=True)
        top = keys[0]
        for k in keys[1:]:
            if f" {k} " not in f" {top} ":
                return False
        if len(hits[top]) > 1:
            return False
        if _repl_words(t, top, "ITEM") != 1:
            return False
        F["item"] = hits[top][0]
    if re.search(r"\bthuoc ke don\b", t.g) and "ke_don" in ban and ban["ke_don"].notna().any():
        t.take(r"\bthuoc ke don\b", " KEDON ")
        F["kedon"] = True
    pays = []
    for k, pat in _PAYW.items():
        for m in re.finditer(r"(?:\bdon(?: hang)? )?(?:\b(?:duoc |da )?(?:tra|thanh toan)(?: bang| qua)? )?(?:\bbang )?\b(?:" + pat + r")\b", t.g):
            pays.append((m.start(), m.end(), k))
    if pays:
        pm = _pay_ids(biz, ban)
        if pm is None:
            return False
        pays.sort()
        for a, b2 in zip(pays, pays[1:]):
            if b2[0] < a[1]:
                return False
        out = []
        for s_, e_, k in pays:
            if k not in pm:
                return False
            out.append(k)
        for s_, e_, k in reversed(pays):
            t.g = t.g[:s_] + " PAY " + t.g[e_:]
        F["pay"] = out
        F["pay_ids"] = {k: pm[k] for k in set(out)}
    m = re.search(r"\bca (sang|chieu)\b", t.g)
    if m:
        t.take(r"\bca (sang|chieu)\b", " CA ")
        F["ca"] = m.group(1)
    return True


def _apply(x, F, which=None):
    """Lọc dòng theo bộ lọc; which = ('kenh', v) / ('pay', k) khi xét từng giá trị. None nếu không áp được."""
    if F["group"] is not None:
        x = x[x["nhom_hang"] == F["group"]]
    if F["nkh"] is not None:
        x = x[x["nhom_khach"] == F["nkh"]]
    if F["item"] is not None:
        x = x[x["ten_hang"] == F["item"]]
    if F["kedon"]:
        x = x[x["ke_don"] == True]  # noqa: E712
    if F["ca"]:
        if "thoi_diem" not in x or x["thoi_diem"].isna().any():
            return None
        h = x["thoi_diem"].dt.hour
        x = x[(h >= 6) & (h < 14)] if F["ca"] == "sang" else x[(h >= 14) & (h < 22)]
    if which:
        kind, v = which
        if kind == "kenh":
            x = x[x["kenh"] == v]
        elif kind == "pay":
            x = x[x["ma_don"].isin(F["pay_ids"][v])]
    elif len(F["kenh"]) == 1:
        x = x[x["kenh"] == F["kenh"][0]]
    elif len(F["pay"]) == 1:
        x = x[x["ma_don"].isin(F["pay_ids"][F["pay"][0]])]
    return x


# ------------------------------------------------------------------ các cụm ý định
_REV_PATS = [
    r"\b(?:tong )?(?:doanh thu|dthu|doanh so)(?: ban hang| ban ra| thu ve| cua hang| tong cong)?\b",
    r"\b(?:tong )?tien (?:ban|hang)(?: ra| duoc| hang)?\b",
    r"\b(?:ban|thu|chot|kiem|dat|mang|lay)(?: (?:ra|duoc|dc|ve|tong|cong|tat|ca|het|luon|total|vao|roi|ITEM|GROUP|KENH|NKH|PAY|TIME|CA))*? (?:bao nhieu|nhieu|may|bn) (?:tien|dong)\b",
    r"\bbao nhieu (?:tien|dong)\b",
]
_CNT_PAT = r"\b(?:bao nhieu|may|bn|so luong|tong so|so) (?:luong )?don(?: hang)?\b"
_HUY_PATS = [
    r"\b(?:dem )?(?:bao nhieu|may|so luong|so)(?: tat ca| luong)? don(?: hang)?(?: bi| da)? huy\b",
    r"\bdon(?: hang)?(?: bi| da)? huy(?: duoc| la| co)? (?:bao nhieu|may)\b",
    r"\b(?:bi |da )?huy (?:mat |het |duoc )?(?:bao nhieu|may) don(?: hang)?\b",
    r"\bdem don(?: hang)? huy(?: duoc)? (?:bao nhieu|may)\b",
]
_RATE_PATS = [
    r"\b(?:ti le|ty le|ti so)(?: cac)?(?: don(?: hang)?)?(?: bi| da)? huy(?: don(?: hang)?)?(?: la)?\b",
    r"\b(?:bi |da )?huy (?:mat |het )?(?:bao nhieu|may) ?(?:%|phan tram)(?: don(?: hang)?)?",
    r"\bdon(?: hang)?(?: bi| da)? huy(?: chiem)?(?: la)? (?:bao nhieu|may) ?(?:%|phan tram)",
]
_XTRA = {"bao", "nhieu", "may", "bn", "%", "phan", "tram"}


def _take_any(t, pats):
    for p in pats:
        m = t.take(p, " INTENT ")
        if m:
            return m
    return None


class C:
    pass


def _months_of(P):
    out = []
    for p in P:
        if p.kind == "month":
            out.append(p)
        elif p.kind == "range" and p.meta.get("months"):
            for m in pd.period_range(pd.Timestamp(p.start), pd.Timestamp(p.end), freq="M"):
                out.append(Period(m.start_time.date(), m.end_time.date(), "month", str(m)))
        else:
            return None
    return out


def _h_bool(c):
    ms = _months_of(c.P)
    if not ms or len(ms) < 2:
        return None
    g = c.g
    yn = re.search(r"(khong|hong|ko|ha|hok|nhi|a)\s*$", g.strip()) or re.search(r"phai khong|dung (?:vay|khong|ko)|co phai", g)
    if not yn:
        return None
    if not re.search(r"doanh thu|dthu|tien ban|tien hang", g):
        return None
    t = T(g)
    _strip_time(t)
    up = r"\b(?:cao hon|lon hon|nhieu hon|hon|tang)\b"
    dn = r"\b(?:thap hon|it hon|kem|giam)\b"
    clauses = re.findall(r"(thang \d{1,2}/\d{4}) (?:lai |cung |cung lai |con )?(cao hon|thap hon|lon hon|nhieu hon|it hon) (thang \d{1,2}/\d{4})", c.g)
    seq_cue = re.search(r"\b(?:luon|deu|lien tuc)\b", g)
    mon = {}
    for p in ms:
        mon[f"thang {p.start.month}/{p.start.year}"] = p
    if clauses and not seq_cue:
        if len(clauses) != len(ms) - 1 and len(clauses) != 1:
            return None
        if any(a not in mon or b not in mon for a, _, b in clauses):
            return None
        chk = re.sub(r"thang \d{1,2}/\d{4}|\b(?:doanh thu|dthu|lai|cung|con|cao hon|thap hon|lon hon|nhieu hon|it hon|va|roi|phai|khong|hong|ko|ha|hok|nhi|a|dung|vay|co)\b", " ", g)
        chk = " ".join(chk.replace(",", " ").split())
        if chk:
            WHY.append("BOOL:" + chk)
            return None
        rs = {}
        for k, p in mon.items():
            if _outside(p, c.ban):
                return None
            rs[k] = _rev(_sel(c.ban, p))
        ok = all((rs[a] > rs[b]) if w in ("cao hon", "lon hon", "nhieu hon") else (rs[a] < rs[b]) for a, w, b in clauses)
        return {"type": "boolean", "answer": bool(ok)}, "So doanh thu từng cặp tháng: " + ", ".join(f"{k}: {v:,.0f}" for k, v in rs.items())
    u, d = re.search(up, g), re.search(dn, g)
    if bool(u) == bool(d):
        return None
    if not (seq_cue or re.search(r"thang sau .*thang truoc", g)):
        if not (re.search(r"\btang\b|\bgiam\b", g) and len(ms) >= 3):
            return None
    chk = " ".join(re.sub(r"\b(?:doanh thu|dthu|thang|sau|truoc|luon|deu|lien tuc|cao hon|lon hon|nhieu hon|hon|tang|thap hon|it hon|kem|giam|"
                          r"co|khong|hong|ko|ha|hok|nhi|a|phai|dung|vay|wa|qua|rui|roi|lai|cung|va|den|toi|tu|so|sanh)\b", " ", t.g).split())
    if re.sub(r"[^a-z0-9]+", " ", re.sub(r"\b(?:TIME|X)\b", " ", chk)).strip():
        WHY.append("BOOL:" + chk)
        return None
    ms2 = sorted(ms, key=lambda p: p.start)
    if ms2 != ms and not seq_cue:
        return None
    rv = []
    for p in ms2:
        if _outside(p, c.ban):
            return None
        rv.append(_rev(_sel(c.ban, p)))
    ok = all(rv[i + 1] > rv[i] for i in range(len(rv) - 1)) if u else all(rv[i + 1] < rv[i] for i in range(len(rv) - 1))
    return {"type": "boolean", "answer": bool(ok)}, "So doanh thu các tháng liên tiếp: " + ", ".join(f"{v:,.0f}" for v in rv)


def _h_months(c):
    ms = _months_of(c.P)
    if not ms or len(ms) < 2:
        return None
    t = T(c.g)
    if re.search(r"%|phan tram|\bso voi\b|\bso vs\b|\b(?:tang|giam|hon|kem)\b", t.g):
        return None
    _strip_time(t)
    arg = None
    m = t.take(r"\bthang nao\b", " Q ")
    want_arg = bool(m)
    t.take(r"\bso sanh\b|\bxem\b", " Q ")
    isrev = bool(_take_any(t, _REV_PATS))
    iscnt = False
    if not isrev:
        iscnt = bool(t.take(_CNT_PAT, " INTENT "))
    if isrev == iscnt:
        return None
    if want_arg:
        mm = t.take(r"\b(cao|lon|nhieu|thap|it)? ?nhat\b", " Q ")
        if not mm:
            return None
        arg = "min" if mm.group(1) in ("thap", "it") else "max"
    n = len(ms)
    t.take(r"\b" + str(n) + r" thang(?: lien(?: nhau| tiep)?| gan nhat)?\b", " Q ")
    for pat in (r"\blan luot\b", r"\b(?:moi|tung) thang\b", r"\btheo (?:dung )?(?:thu tu|trinh tu)(?: thoi gian)?(?: do)?\b", r"\bdung thu tu do\b",
                r"\bxep (?:tu )?thang cu (?:toi|den) (?:thang )?(?:moi|gan nhat)\b", r"\bliet ke\b", r"\bke\b", r"\bghi\b", r"\bcho xin\b", r"\btung thang\b",
                r"\bcua\b", r"\bnhau\b"):
        t.take(pat, " Q ")
    if not _clean(t, extra={"xin", "thu", "tu", "cu", "gan", "nhat", "gi", "q", "tung", "moi", "cac", str(n)}):
        return None
    sort_cue = re.search(r"theo (?:dung )?(?:thu tu|trinh tu) thoi gian|tu thang cu|thu tu thoi gian", c.g)
    ms2 = sorted(ms, key=lambda p: p.start) if (sort_cue or arg) else ms
    vals = []
    for p in ms2:
        if _outside(p, c.ban):
            return None
        x = _sel(c.ban, p)
        vals.append(_tien(_rev(x)) if isrev else _nord(x))
    if arg:
        best = max(range(len(ms2)), key=lambda i: (vals[i] if arg == "max" else -vals[i], -ms2[i].start.toordinal()))
        p = ms2[best]
        return {"type": "date", "answer": f"{p.start.year:04d}-{p.start.month:02d}"}, "Tháng " + arg + f" trong các tháng nêu: {vals}"
    return {"type": "list", "answer": vals}, "Giá trị từng tháng theo thứ tự nêu"


def _cmp_words(t):
    for pat in (r"\bso voi\b", r"\bso vs\b", r"\bso\b", r"\bvoi\b", r"\bcung ky\b", r"\btang truong\b", r"\btang hay giam\b", r"\btang giam\b",
                r"\bthay doi\b", r"\bhon hay kem\b", r"\bcao hay thap(?: hon)?\b", r"\blen hay xuong\b", r"\bnhieu hon hay it hon\b", r"\bhon hay it\b"):
        while t.take(pat, " CMP "):
            pass
    for pat in (r"\btang\b", r"\bgiam\b", r"\bhon\b", r"\bkem\b", r"\blen\b", r"\bxuong\b", r"\bcao\b", r"\bthap\b", r"\bhay\b"):
        while t.take(pat, " CMP "):
            pass


def _h_cmp(c):
    if not re.search(r"%|phan tram", c.g) or not _co(c.rb, "pct"):
        return None
    P = c.P
    cur = ref = None
    if len(P) == 1:
        cur, ref = P[0], P[0].meta.get("compare")
        if ref is None:
            if cur.kind in ("month", "week", "quarter") and re.search(r"\b(?:thang|tuan|quy) (?:lien )?truoc(?: do)?\b|\bky truoc\b", c.g):
                ref = previous(cur, c.cal)
            else:
                return None
    elif len(P) == 2:
        kinds = {P[0].kind, P[1].kind}
        if kinds == {"tet", "before_tet"}:
            cur = P[0] if P[0].kind == "tet" else P[1]
            ref = P[1] if cur is P[0] else P[0]
        elif P[0].kind == P[1].kind and P[0].kind in ("month", "quarter"):
            toks = list(re.finditer(r"(?:thang|quy) \d{1,2}/\d{4}", c.g))
            ci = [m.start() for m in re.finditer(r"\bso voi\b|\bso vs\b|\bso\b", c.g)]
            if len(toks) != 2 or len(ci) != 1:
                return None
            before = [m for m in toks if m.start() < ci[0]]
            if len(before) == 1:
                cur, ref = P[0], P[1]
            elif len(before) == 0:
                cur, ref = P[1], P[0]
            else:
                return None
        else:
            return None
    else:
        return None
    if ref.start is None:
        return None
    t = T(c.g)
    _strip_time(t)
    _cmp_words(t)
    t.take(r"%|\bphan tram\b", " PCT ")
    t.take(r"\b(?:bao nhieu|may|bn)\b", " Q ")
    isrev = bool(_take_any(t, _REV_PATS))
    iscnt = False
    if not isrev:
        iscnt = bool(t.take(_CNT_PAT, " INTENT "))
    if isrev == iscnt:
        return None
    if not _clean(t, extra={"gi", "q"}):
        return None
    if _outside(cur, c.ban) or _outside(ref, c.ban):
        if _co(c.rb, "unans"):
            return _unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu"
        return None
    xa, xb = _sel(c.ban, cur), _sel(c.ban, ref)
    a, b = (_rev(xa), _rev(xb)) if isrev else (_nord(xa), _nord(xb))
    v = _pct(a - b, b)
    if v is None:
        return None
    return {"type": "percent", "answer": v}, f"({cur.label} {a:,.0f} - {ref.label} {b:,.0f}) / {ref.label}"


_UNITS_CACHE = {}


def _units(ban):
    k = _UNITS_CACHE.get("u")
    if k is not None and k[0] is ban:
        return k[1]
    u = {}
    if "don_vi_co_ban" in ban:
        for v in ban["don_vi_co_ban"].dropna().unique():
            u[fold(str(v)).strip()] = v
    _UNITS_CACHE["u"] = (ban, u)
    return u


def _unit_in(t, ban):
    """Đơn vị cơ bản được hỏi: 'bao nhieu <dv>' / 'may <dv>' / 'tong so <dv>' / 'theo <dv>' / 'nhieu <dv> nhat'."""
    us = _units(ban)
    if not us:
        return None, None
    alt = "|".join(sorted((re.escape(k) for k in us), key=len, reverse=True))
    m = re.search(r"\b(?:bao nhieu|may|bn|tong so|tong cong|so luong|so|theo|nhieu)\s+(" + alt + r")\b(?! nhieu)", t.g)
    return (m, fold(m.group(1))) if m else (None, None)


def _h_top(c):
    if len(c.P) != 1:
        return None
    p = c.P[0]
    if not _co(c.rb, "hoa"):
        return None
    t = T(c.g)
    F = {}
    if not _filters(t, c.ban, c.biz, F):
        return None
    if F["kenh"] or F["pay"] or F["nkh"] or F["kedon"] or F["ca"] or F["item"]:
        return None
    _strip_time(t)
    ban = c.ban
    b = _sel(ban, p)
    b = b[b["tinh_dt"]]
    if not len(b) or _outside(p, ban):
        return None
    n = None
    m = re.search(r"\btop (\d{1,2})\b", t.g) or re.search(r"(?<![/\d])\b(\d{1,2}) (?:mat hang|san pham|sp|mon)\b", t.g)
    if m:
        n = int(m.group(1))
        t.g = t.g[:m.start()] + " N " + t.g[m.end():]
    t.take(r"\btop\b", " N ")
    ms = []
    pats = {
        "rev": r"\b(?:doanh thu|dthu|tien) (?:cao|lon|nhieu) nhat\b|\bnhieu tien nhat\b|\bthu ve nhieu tien nhat\b|\b(?:cao|lon|nhieu) nhat (?:ve )?(?:doanh thu|tien)\b|\bco doanh thu (?:cao|lon) nhat\b",
        "lai": r"\blai(?: gop)? (?:cao|nhieu|lon) nhat\b|\blai nhat\b|\bnhieu lai nhat\b",
        "don": r"\b(?:nhieu|cao) (?:don|lan)(?: hang)? nhat\b|\bso don (?:mua )?nhieu nhat\b",
        "qty": r"\bban chay(?: nhat)?\b|\bhot nhat\b|\bchay nhat\b|\bban (?:duoc )?nhieu nhat\b",
    }
    for k, pat in pats.items():
        if re.search(pat, t.g):
            ms.append(k)
    mu, unit = _unit_in(t, ban)
    if mu is not None and re.search(r"nhieu\s+" + re.escape(unit) + r"\s+nhat", t.g):
        ms.append("qty_u")
    if len(ms) != 1:
        return None
    metric = ms[0]
    if metric == "qty_u":
        t.take(r"\bnhieu\s+" + re.escape(unit) + r"\s+nhat\b", " INTENT ")
    else:
        t.take(pats[metric], " INTENT ")
    t.take(r"\bban (?:duoc )?", " Q ")
    t.take(r"\bmua hang\b", " Q ")
    for pat in (r"\b(?:cua hang|quan|tiem|shop|sop)\b",):
        while t.take(pat, " Q "):
            pass
    cands = []
    if re.search(r"\b(?:mat hang|san pham|sp|mon)\b|\bhang nao\b", t.g) and metric in ("rev", "lai", "qty", "qty_u"):
        cands.append(("item", r"\b(?:mat hang|san pham|sp|mon|hang)\b"))
    if re.search(r"\b(?:ngay|bua) nao\b", t.g) and metric == "rev":
        cands.append(("day", r"\b(?:ngay|bua) nao\b"))
    if re.search(r"\b(?:nhan vien|nv|nguoi ban)\b", t.g) and metric == "rev":
        cands.append(("nv", r"\b(?:nhan vien|nv|nguoi ban)(?: nao)?\b"))
    if re.search(r"\b(?:khach(?: hang)?(?: nao)?|ai)\b", t.g) and metric in ("rev", "don"):
        cands.append(("khach", r"\bkhach(?: hang)?(?: nao)?\b|\bai\b"))
    if len(cands) != 1:
        return None
    subj = cands[0][0]
    while t.take(cands[0][1], " S "):
        pass
    t.take(r"\bthu ve\b", " Q ")
    if subj is None:
        return None
    for pat in (r"\b(?:la|nao|gi|ai)\b", r"\b(?:cua|trong)\b", r"\b(?:cua hang|quan|tiem|shop)\b", r"\bmua\b", r"\bdem theo so don\b"):
        while t.take(pat, " Q "):
            pass
    ex = {"gi", "q", "nhat", "n", "nao", "xep", "hang", "ke", "liet", "tong", "tien", "ca", "doanh", "thu", "ten", "dem", "theo", "so", "don"}
    if n is not None:
        ex |= {"nhung", "cac"}
    if not _clean(t, extra=ex):
        return None
    if n is None and re.search(r"\b(nhung|cac)\b", c.g):
        return None
    if F["group"] is not None:
        if subj != "item":
            return None
        b = b[b["nhom_hang"] == F["group"]]
        if not len(b):
            return None
    if subj == "day":
        if not _co(c.rb, "dt"):
            return None
        s = b.groupby(_day(b["ngay"]))["thanh_tien"].sum().reset_index().sort_values(["thanh_tien", "ngay"], ascending=[False, True])
        return {"type": "date", "answer": s.iloc[0]["ngay"].strftime("%Y-%m-%d")}, "Ngày có doanh thu lớn nhất trong kỳ (hoà: sớm hơn)"
    if subj == "nv":
        if not _co(c.rb, "nv", "dt"):
            return None
        col = "nhan_vien" if "nhan_vien" in b and b["nhan_vien"].notna().any() else ("ten_nhan_vien" if "ten_nhan_vien" in b else None)
        if col is None or n not in (None, 1):
            return None
        s = b[b[col].notna()].groupby(col)["thanh_tien"].sum().reset_index().sort_values(["thanh_tien", col], ascending=[False, True])
        if not len(s):
            return None
        return {"type": "string", "answer": str(s.iloc[0][col])}, "Nhân viên có doanh thu lớn nhất"
    if subj == "khach":
        if "khach" not in b or not b["khach"].notna().any() or not _co(c.rb, "khach", "khach_tien", "khach_mua"):
            return None
        k = b[b["khach"].notna()].copy()
        k["_k"] = k["khach"].astype(str)
        gg = k.groupby("_k").agg(tien=("thanh_tien", "sum"), so_don=("ma_don", "nunique"), ten=("ten_khach", "first")).reset_index()
        col = "tien" if metric == "rev" else "so_don"
        gg = gg.sort_values([col, "_k"], ascending=[False, True])
        ten = lambda r: str(r["ten"]) if pd.notna(r["ten"]) else str(r["_k"])
        if n and n >= 2:
            if len(gg) < n:
                return None
            return {"type": "list", "answer": [ten(r) for _, r in gg.head(n).iterrows()]}, f"Top {n} khách theo {col}"
        if n in (None, 1):
            return {"type": "string", "answer": ten(gg.iloc[0])}, f"Khách đứng đầu theo {col}"
        return None
    if not _co(c.rb, "phi", "chay"):
        return None
    x = b[~b["la_phi"]] if "la_phi" in b else b
    key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if "ma_hang" in x else x["ten_hang"]
    x = x.assign(_k=key.astype(str))
    if metric in ("qty", "qty_u"):
        if "so_luong_co_ban" not in x or x["so_luong_co_ban"].isna().any():
            return None
        if metric == "qty_u":
            x = x[x["don_vi_co_ban"].map(lambda v: fold(str(v)).strip() if pd.notna(v) else "") == unit]
        gg = x.groupby("_k").agg(v=("so_luong_co_ban", "sum"), ten=("ten_hang", "first")).reset_index()
    elif metric == "rev":
        if not _co(c.rb, "dt"):
            return None
        gg = x.groupby("_k").agg(v=("thanh_tien", "sum"), ten=("ten_hang", "first")).reset_index()
    else:
        if not _co(c.rb, "loi", "gv") or "gia_von" not in x or x["gia_von"].isna().any():
            return None
        x = x.assign(_lai=x["thanh_tien"] - x["so_luong"] * x["gia_von"])
        gg = x.groupby("_k").agg(v=("_lai", "sum"), ten=("ten_hang", "first")).reset_index()
    if not len(gg):
        return None
    gg = gg.sort_values(["v", "_k"], ascending=[False, True])
    if n and n >= 2:
        if len(gg) < n:
            return None
        return {"type": "list", "answer": [str(v) for v in gg["ten"].head(n)]}, f"Top {n} mặt hàng theo {metric}"
    if n == 1:
        return {"type": "list", "answer": [str(gg.iloc[0]["ten"])]}, "Mặt hàng đứng đầu"
    return {"type": "string", "answer": str(gg.iloc[0]["ten"])}, f"Mặt hàng đứng đầu theo {metric}"


def _oor(c):
    return (_unans("Kỳ nằm ngoài khoảng ngày có dữ liệu"), "ngoài dữ liệu") if _co(c.rb, "unans") else None


def _h_measure(c):
    if len(c.P) != 1:
        return None
    p = c.P[0]
    ban = c.ban
    t = T(c.g)
    F = {}
    thr = _thr(t)
    if thr == "BAD":
        return None
    cthr = _cnt_thr(t)
    if not _filters(t, ban, c.biz, F):
        return None
    nf = (F["group"] is not None) + bool(F["kenh"]) + (F["nkh"] is not None) + (F["item"] is not None) + F["kedon"] + bool(F["pay"]) + bool(F["ca"])
    nvals = max(len(F["kenh"]), len(F["pay"]))
    if len(F["kenh"]) > 1 and (F["pay"] or nf > 1):
        return None
    if len(F["pay"]) > 1 and (F["kenh"] or nf > 1):
        return None
    if re.search(r"\b(?:so voi|so vs|tang|giam|hon|kem|chenh)\b", t.g):
        return None
    _strip_time(t)
    b = _sel(ban, p)
    out_range = _outside(p, ban)

    if thr:
        if nf or cthr is not None:
            return None
        op, val = thr
        isday = t.take(r"\b(?:bao nhieu|may|so) ngay\b", " INTENT ")
        isord = t.take(r"\b(?:bao nhieu|may|so) (?:luong )?don(?: hang)?\b", " INTENT ") if not isday else None
        if bool(isday) == bool(isord):
            return None
        if not re.search(r"doanh thu|dthu|tien|gia tri|\bban\b|\bdat\b", t.g):
            return None
        for pat in (r"\bdoanh thu\b", r"\bdthu\b", r"\btien hang\b", r"\btien ban\b", r"\btien\b", r"\bgia tri\b", r"\bban(?: duoc)?\b", r"\bdat\b", r"\bca ngay\b",
                    r"\bco\b", r"\bma\b", r"\bnao\b", r"\bcua\b", r"\bdon\b", r"\bcac\b"):
            while t.take(pat, " Q "):
                pass
        if not _clean(t, extra={"q", "gi", "thr", "tro", "len", "xuong", "hang", "cua"}):
            return None
        if out_range:
            return _oor(c)
        x = b[b["tinh_dt"]]
        if isday:
            s = x.groupby(_day(x["ngay"]))["thanh_tien"].sum().round(6)
        else:
            s = x.groupby("ma_don")["thanh_tien"].sum().round(6)
        k = int((s >= val).sum()) if op == ">=" else int((s > val).sum())
        return {"type": "number", "answer": k}, f"Số {'ngày' if isday else 'đơn'} có doanh thu {op} {val:,.0f}"

    if cthr is not None:
        if nf or not _co(c.rb, "khach_mua"):
            return None
        isitem = bool(re.search(r"\b(?:mat hang|san pham|mon)\b", t.g))
        pct = bool(re.search(r"%|phan tram|ti le", t.g))
        x = b[b["tinh_dt"]]
        if isitem:
            if pct or not _co(c.rb, "phi") or "la_phi" not in x:
                return None
            t.take(r"\b(?:bao nhieu|may|dem|so)\b", " INTENT ")
            t.take(r"\b(?:mat hang|san pham|mon)\b", " S ")
            if not _clean(t, extra={"gi", "goi", "cthr", "cthrg", "duoc", "nao", "khac", "nhau", "s", "q", "intent", "dem"}):
                return None
            if out_range:
                return None
            x = x[~x["la_phi"]]
            key = "ma_hang" if x["ma_hang"].notna().any() else "ten_hang"
            cc = x.groupby(key)["ma_don"].nunique()
            return {"type": "number", "answer": int((cc >= cthr).sum())}, f"Số mặt hàng có mặt ở >= {cthr} đơn"
        if "khach" not in ban or not ban["khach"].notna().any():
            return None
        t.take(r"\b(?:bao nhieu|may|dem|so)\b", " INTENT ")
        t.take(r"%|\bphan tram\b|\bti le\b", " PCT ")
        t.take(r"\bkhach(?: hang)?\b|\bnguoi\b", " S ")
        for pat in (r"\bmua(?: hang)?\b", r"\bchiem\b", r"\btat ca\b", r"\bda\b", r"\bduoc\b", r"\bcho\b", r"\bso\b", r"\bdem\b"):
            while t.take(pat, " Q "):
                pass
        if not _clean(t, extra={"gi", "cthr", "cthrg", "s", "q", "intent", "pct", "nguoi", "khach", "tang", "qua", "set", "nguog", "ne", "muon", "quen", "qa", "mk", "tinh"}):
            return None
        if out_range:
            return None
        xx = x[x["khach"].notna()].drop_duplicates(["ma_don"])
        cc = xx.groupby("khach")["ma_don"].nunique()
        if not len(cc):
            return None
        if re.search(r"%|phan tram|ti le", c.g):
            return {"type": "percent", "answer": _pct(int((cc >= cthr).sum()), len(cc))}, f"Khách >= {cthr} đơn / khách đã mua"
        return {"type": "number", "answer": int((cc >= cthr).sum())}, f"Số khách mua >= {cthr} đơn"

    rate = _take_any(t, _RATE_PATS) if not nf else None
    huy = None
    if not rate and not nf:
        huy = _take_any(t, _HUY_PATS)
    if rate or huy:
        if not _co(c.rb, "huy") or "trang_thai" not in ban:
            return None
        for pat in (r"%", r"\bphan tram\b", r"\bbao nhieu\b", r"\bmay\b", r"\btong so\b"):
            t.take(pat, " Q ")
        if not _clean(t, extra=_XTRA | {"gi", "q", "intent"}):
            return None
        if out_range:
            return _oor(c)
        nh = int(b.loc[b["trang_thai"] == "huỷ", "ma_don"].nunique())
        if huy:
            return {"type": "number", "answer": nh}, "Số đơn huỷ trong kỳ"
        na = int(b["ma_don"].nunique())
        if not na:
            return None
        v = (Decimal(nh) * 100 / Decimal(na)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {"type": "percent", "answer": float(v)}, "Đơn huỷ / mọi đơn đặt trong kỳ"

    pct = bool(re.search(r"%|\bphan tram\b|\bti trong\b|\bchiem\b", t.g))

    mu, unit = _unit_in(t, ban)
    qty = None
    allit = r"\b(?:tat ca|het cac|moi|cac|toan bo)\s+(?:mat hang|mon|san pham)\b"
    if mu is not None and not pct and not re.search(r"\b(?:bao nhieu|may|nhieu|bn) (?:tien|dong)\b|\bdoanh thu\b|\bdthu\b|\btien (?:ban|hang)\b", t.g):
        if F["item"] is not None or F["group"] is not None or re.search(allit, t.g) \
                or (re.search(r"\bban\b", t.g) and unit not in ("lan",) and not nf and not re.search(r"\bnhan vien\b|\bkhach\b|\bsize\b", t.g)):
            qty = unit
    if qty:
        if F["kenh"] or F["nkh"] or F["pay"] or F["kedon"] or F["ca"] or not _co(c.rb, "don_vi", "phi"):
            return None
        t.g = t.g[:mu.start()] + " INTENT " + t.g[mu.end():]
        for pat in (allit + r"(?: ban)?", r"\btinh (?:tat ca|het|moi)?\b", r"\btheo " + re.escape(unit) + r"\b", r"\bban(?: (?:duoc|dc|ra))?\b", r"\btong(?: cong| so)?\b", r"\bso luong\b", r"\btheo\b"):
            while t.take(pat, " Q "):
                pass
        if not _clean(t, extra={"gi", "q", "intent", "tinh", "gom"}):
            return None
        if out_range:
            return None
        x = b[b["tinh_dt"]]
        if "la_phi" in x:
            x = x[~x["la_phi"]]
        if "so_luong_co_ban" not in x or x["so_luong_co_ban"].isna().any():
            return None
        if F["item"] is not None:
            x = x[x["ten_hang"] == F["item"]]
            sub = ban[ban["ten_hang"] == F["item"]]
            if "ma_hang" in sub and sub["ma_hang"].nunique() > 1:
                return None
            if {fold(str(v)).strip() for v in sub["don_vi_co_ban"].dropna().unique()} != {unit}:
                return None
        else:
            if F["group"] is not None:
                x = x[x["nhom_hang"] == F["group"]]
            x = x[x["don_vi_co_ban"].map(lambda v: fold(str(v)).strip() if pd.notna(v) else "") == unit]
        return {"type": "number", "answer": _sl(x["so_luong_co_ban"].sum())}, f"Cộng so_luong_co_ban đơn vị '{unit}' (item/nhóm/tất cả)"

    if pct:
        if nf != 1 or nvals > 1 or not _co(c.rb, "dt"):
            return None
        if F["item"] is not None or F["ca"]:
            return None
        if F["nkh"] is not None and not _co(c.rb, "tt_nhom"):
            return None
        if F["kedon"] and not _co(c.rb, "ke_don"):
            return None
        isrev = bool(_take_any(t, _REV_PATS))
        for pat in (r"%", r"\bphan tram\b", r"\bti trong\b", r"\bchiem\b", r"\bbao nhieu\b", r"\bmay\b", r"\bcua\b", r"\bdong gop\b", r"\btong\b",
                    r"\bdoanh thu\b", r"\bdthu\b", r"\bcua hang\b", r"\bduoc\b", r"\bla\b", r"\bmang ve\b", r"\bban\b"):
            while t.take(pat, " Q "):
                pass
        if not isrev and not re.search(r"\bdoanh thu|\bdthu", c.g):
            return None
        if not _clean(t, extra={"gi", "q", "tong", "chiem", "intent", "ban", "tren", "trong"}):
            return None
        if out_range:
            return _oor(c)
        den = _rev(b)
        x = _apply(b, F)
        if x is None:
            return None
        v = _pct(_rev(x), den)
        if v is None:
            return None
        return {"type": "percent", "answer": v}, f"Tỉ trọng: {_rev(x):,.0f} / {den:,.0f}"

    avg = bool(t.has(r"\bbinh quan\b") and t.has(r"\bdon\b"))
    kind = None
    if avg:
        for pat in (r"\bbinh quan\b", r"\b(?:moi |mot |1 )?(?:moi |mot )?don(?: hang)?\b"):
            t.take(pat, " INTENT ")
        for pat in (r"\bdoanh thu\b", r"\bdthu\b", r"\bgia tri\b", r"\bkhach(?: hang)? mua\b", r"\bmang ve\b", r"\bmang lai\b", r"\bdem (?:ve|de|lai)\b", r"\bhet\b",
                    r"\btien hang\b", r"\btien ban\b", r"\btien\b", r"\bbao nhieu\b", r"\bmay\b", r"\bcua\b", r"\btren\b", r"\bban\b"):
            while t.take(pat, " Q "):
                pass
        kind = "avg"
    else:
        isrev = bool(_take_any(t, _REV_PATS))
        iscnt = False
        if not isrev:
            iscnt = bool(t.take(_CNT_PAT, " INTENT "))
        elif t.has(_CNT_PAT):
            return None
        if isrev == iscnt:
            return None
        kind = "rev" if isrev else "cnt"
        if kind == "cnt":
            t.take(r"\bco it nhat mot (?:mat hang|san pham|mon)\b", " Q ")
            if F["item"] is not None or F["group"] is not None:
                t.take(r"\b(?:co|mua)\b", " Q ")
    ex4 = {"gi", "q", "intent", "tren", "cac"}
    if nvals >= 2:
        ex4 |= {"voi", "tung", "ben", "moi", "kenh", "lan", "luot", "va", "con", "cac", "don"}
    if not _clean(t, extra=ex4):
        return None
    if kind == "avg" and (F["group"] is not None or F["item"] is not None or F["kedon"] or not _co(c.rb, "tb")):
        return None
    if out_range:
        return _oor(c)

    def val(x):
        if kind == "rev":
            return _tien(_rev(x))
        if kind == "cnt":
            return _nord(x)
        n_ = _nord(x)
        if not n_:
            return None
        return int((Decimal(repr(round(_rev(x), 6))) / Decimal(n_)).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    if nvals >= 2:
        which = "kenh" if len(F["kenh"]) >= 2 else "pay"
        if kind == "avg":
            return None
        res = []
        for v in (F["kenh"] if which == "kenh" else F["pay"]):
            x = _apply(b, F, which=(which, v))
            if x is None:
                return None
            res.append(val(x))
        return {"type": "list", "answer": res}, f"Giá trị từng {which} theo thứ tự nêu"
    x = _apply(b, F)
    if x is None:
        return None
    v = val(x)
    if v is None:
        return None
    return {"type": "number", "answer": v}, f"{kind} kỳ {p.label} (lọc: {sorted(k for k, vv in F.items() if vv and k != 'pay_ids')})"


def _solve(q, f, biz, rulebook):
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban):
        return None
    for col in ("ma_don", "ngay", "tinh_dt", "thanh_tien"):
        if col not in ban:
            return None
    rb = _rbn(rulebook)
    if not _co(rb, "dt", "dt2", "don"):
        return None
    as_of, cal = _ctx()
    if as_of is None:
        return None
    today = pd.Timestamp(as_of).normalize()
    g = _prep(f)
    P = find_periods(g, as_of, cal)
    if not P or len(P) > 6:
        return None
    for p in P:
        if p.kind == "snapshot" or p.start is None or pd.Timestamp(p.end) > today:
            return None
    c = C()
    c.q, c.g, c.P, c.ban, c.biz, c.rb, c.today, c.cal = q, g, P, ban, biz, rb, today, cal
    for fn in (_h_bool, _h_months, _h_cmp, _h_top, _h_measure):
        r = fn(c)
        if r:
            return r
    return None


def solve(q, f, periods, biz, rulebook):
    try:
        return _solve(q, f, biz, rulebook)
    except Exception as e:  # noqa: BLE001
        WHY.append("EXC:" + repr(e)[:100])
        return None
