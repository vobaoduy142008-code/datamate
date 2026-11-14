"""Công thức CỐ ĐỊNH cho câu CƠ BẢN (một kỳ, một con số), không dùng LLM.
Nguyên tắc: mặc định TỪ CHỐI. Câu chỉ được nhận khi mọi từ còn lại (sau khi bỏ cụm kỳ, tên mặt hàng, cụm ý định)
đều nằm trong danh sách từ vô hại. Không chắc -> None để LLM giải."""
import re
import math
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from loader import fold

# ---------------------------------------------------------------- điều kiện bật theo rulebook
_LUAT = {
    "dt": "tong thanh tien cac dong hang",
    "dt2": "don khong huy va khong tra hang",
    "don": "chi tinh don khong huy khong tra",
    "tb": "gia tri trung binh mot don",
    "khach": "khach da mua trong ky la khach co it nhat mot don nhu vay",
}


def _co_luat(rulebook, *keys):
    if rulebook is None:
        return False
    t = re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook))
    t = re.sub(r"\s+", " ", t)
    return all(_LUAT[k] in t for k in keys)


# ---------------------------------------------------------------- cụm thời gian
_TIME_RE = [
    r"(?:tu |den |toi )?(?:ngay )?\b\d{1,2} ?/ ?\d{1,2} ?/ ?\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:th|t)\s?\d{1,2}\s?[/-]\s?\d{2,4}\b",
    r"\b\d{1,2}-\d{4}\b",
    r"\bthang \d{1,2} ?/ ?\d{4}\b",
    r"\b\d{1,2} ?/ ?\d{4}\b",
    r"(?:ngay )?\b\d{1,2} ?/ ?\d{1,2}\b",
    r"\bthang \d{1,2}(?: nam (?:\d{4}|nay|ngoai|truoc))?\b",
    r"\bnam \d{4}\b",
    r"\bquy \d(?: nam (?:\d{4}|nay|ngoai|truoc))?\b",
    r"\b\d+ ngay (?:gan nhat|qua|vua qua|vua roi|truoc|gan day)\b",
    r"\b\d+ ngay\b",
    r"\bngay \d{1,2}\b",
    r"\b(?:hom nay|hom qua|bua nay|bua qua|hnay|hqua)\b",
    r"\b(?:tuan|thang|nam|quy) (?:nay|truoc|roi|ngoai|vua roi|vua qua|truoc do)\b",
    r"\b(?:tu )?dau (?:nam|thang|tuan|quy)(?: (?:den|toi|cho den) (?:het )?(?:hom nay|nay|bay gio|hien tai|nay))?\b",
    r"\b(?:den|toi|tinh den|tinh toi|cho den|den tan) (?:het )?(?:hom nay|nay|bay gio|hien tai|hom qua)\b",
]
_TIME_TOK = {"thang", "nam", "tuan", "quy", "ngay", "hom", "nay", "qua", "truoc", "roi", "tu", "den", "toi",
             "het", "dau", "tinh", "trong", "vao", "luc", "trong"}

# từ vô hại còn lại
_OK = set("""cho minh toi hoi biet xem giup gium coi check kiem tra vay nhi a ha ah nha nhe oi e ak ay
la bao nhieu bn may so tong cong co duoc dc ra cua cua hang shop quan tiem tat ca da mot va
ban mua mat hang mon san pham thuoc loai vao tien dong vnd d the nao nhi vay z do ta total dem tiem
hien tai thi ma thoi ne nay a vay nhe""".split())
_OK -= {"tra", "ca", "kiem", "vao"}  # 'tra' = trả; 'ca' = ca sáng; bỏ hai chữ dễ nhầm
_OK |= {"kiem"}   # 'kiem tra' hỏi

_BLOCK = re.compile(
    r"so voi|tang|giam|hon|kem|nhat|\btop\b|cao|thap|it |nhieu nhat|lai|von|nhap|ton\b|no\b|chiu|cod|"
    r"ca (sang|chieu)|gio\b|kenh|zalo|dien thoai|quay|\bsi\b|le\b|vang lai|nhom|size|ke don|hoat chat|"
    r"tra hang|tra lai|doi tra|dang giao|ship|giao|voucher|khuyen mai|vat|thue|phi\b|tru\b|khong tinh|"
    r"ngoai|chua|rieng|tung|theo|moi ngay|moi thang|moi tuan|nhan vien|\bnv\b|tien mat|chuyen khoan|\bck\b|"
    r"the\b|momo|cuoi tuan|thu [2-7]\b|chu nhat|ngay thuong|sinh nhat|dip|tet|ky\b|\bvs\b|cung ky|lien truoc|"
    r"nao\b|ai\b|dau\b|bang|duoi|tu \d+|\d+ tro|khoang|xap xi|it nhat|toi da|toi thieu|\bgia\b(?! tri)")


WHY = []


def _no():
    import sys
    WHY.append("L%d" % sys._getframe(1).f_lineno)
    return None


def _norm(s):
    s = re.sub(r"[^a-z0-9/%\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tien_quy(v):
    return int(Decimal(repr(round(float(v), 6))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _sl_fmt(v):
    v = round(float(v), 6)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


def _words(key):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", key)).strip()


def _ten_hang(g, q, ban):
    """Tìm đúng MỘT mặt hàng (tên đầy đủ) trong câu. Trả (g đã thay tên bằng ITEM, tên) | (g, None) | (g, "AMBIG")."""
    if "ten_hang" not in ban:
        return g, None
    names = ban.loc[~ban["la_phi"], "ten_hang"].dropna().unique() if "la_phi" in ban else ban["ten_hang"].dropna().unique()
    gk = " " + _words(g) + " "
    hits = {}
    for n in names:
        k = _words(fold(n))
        if len(k) >= 3 and f" {k} " in gk:
            hits.setdefault(k, []).append(n)
    if not hits:
        return g, None
    keys = sorted(hits, key=len, reverse=True)
    top = keys[0]
    for k in keys[1:]:
        if f" {k} " not in f" {top} ":      # hai mặt hàng khác nhau cùng được nhắc
            return g, "AMBIG"
    raws = hits[top]
    if len(raws) > 1:
        ql = " " + re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", q.lower())).strip() + " "
        sel = [n for n in raws if " " + re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(n).lower())).strip() + " " in ql]
        if len(sel) != 1:
            return g, "AMBIG"
        raws = sel
    name = raws[0]
    pat = r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(t) for t in top.split()) + r"(?![a-z0-9])"
    g2, n = re.subn(pat, " ITEM ", g)
    if n != 1:
        return g, "AMBIG"
    return g2.strip(), name


def _thr(f):
    """Tìm ngưỡng 'từ X trở lên' / '≥ X' trong câu gốc (đã fold). -> (f đã thay bằng ' THR ', giá trị) | (f, None) | (f, 'BAD')."""
    f2 = f.replace("≥", " gte ").replace(">=", " gte ")
    m = re.search(r"(?:\bgte\b|\btu\b|\bdat\b)\s*(\d[\d.,]*)\s*(?:dong|vnd|d\b)?(\s*tro len)?", f2)
    if not m or not (re.search(r"\bgte\b", m.group(0)) or m.group(2)):
        return f, None
    n = m.group(1).rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", n):
        v = int(re.sub(r"[.,]", "", n))
    elif n.isdigit():
        v = int(n)
    else:
        return f, "BAD"
    return f2[:m.start()] + " thrsym " + f2[m.end():], v


def solve(q, f, periods, biz, rulebook):
    ban = getattr(biz, "ban", None)
    if ban is None or not len(ban) or len(periods) != 1:
        return _no()
    p = periods[0]
    if p.kind == "snapshot" or p.start is None:
        return _no()
    if not _co_luat(rulebook, "dt", "dt2", "don"):
        return _no()
    for c in ("ma_don", "ngay", "tinh_dt", "thanh_tien"):
        if c not in ban:
            return _no()
    # --- khoảng ngày có dữ liệu
    nmin, nmax = ban["ngay"].min().normalize(), ban["ngay"].max().normalize()
    st, en = pd.Timestamp(p.start).normalize(), pd.Timestamp(p.end).normalize()
    if en < nmin or st > nmax:
        return _no()

    f, thr = _thr(f)
    if thr == "BAD":
        return _no()
    g = _norm(f)
    # --- mặt hàng
    g, item = _ten_hang(g, q, ban)
    if item == "AMBIG":
        return _no()
    gb = " " + g + " "
    for pat in _TIME_RE:
        gb = re.sub(pat, " TIME ", gb)
    gb = re.sub(r"\s+", " ", gb)
    gb = re.sub(r"\borders?\b", " don ", gb)
    gb = re.sub(r"\brevenue\b", " doanh thu ", gb)
    mb = _BLOCK.search(re.sub(r"\b(?:TIME|ITEM|thrsym)\b", " ", gb))
    if mb:
        WHY.append("BLOCK:" + mb.group(0))
        return _no()
    DON = r"(?:don hang|don|bill|hoa don)"
    spans = {
        "AVG": r"(?:gia tri |doanh thu |tien )?(?:trung binh|binh quan)(?: (?:cua|tren|moi|mot|1))* ?(?:moi |mot |1 )?" + DON + r"(?: (?:trung binh|binh quan))?(?: (?:gia tri|doanh thu))?"
               r"|(?:moi|mot|1) " + DON + r" (?:trung binh|binh quan)",
        "REV": r"\b(?:dat (?:duoc )?)?(?:tong )?(?:doanh thu|dthu)(?: ban hang| ban ra| thu ve)?(?: tong cong| cua)?\b|\btien ban(?: duoc| ra| hang)?\b|\b(?:ban(?: ra)?|thu ve|thu duoc) (?:ITEM )?(?:duoc |dc |ra )?(?:tong cong |tat ca |total )?(?:bao nhieu|nhieu|bn|bnhiu) tien",
        "CNT": r"(?:tong )?(?:so|bao nhieu|may) (?:luong )?" + DON + r"(?: hang)?|" + DON + r" (?:ban duoc|da ban)",
        "KHACH": r"(?:so|bao nhieu|may) (?:luong )?khach(?: hang)?(?: da)?(?: mua(?: hang)?)|khach(?: hang)? (?:da )?mua(?: hang)? (?:la )?(?:bao nhieu|may)",
        "HUY": r"(?:so|bao nhieu|may) (?:luong )?" + DON + r"(?: hang)? (?:bi |da )?huy|" + DON + r" (?:bi |da )?huy (?:la )?(?:bao nhieu|may)",
        "RATE": r"(?:ti le|ty le|ti so|phan tram) (?:cac )?" + DON + r"(?: bi| da)? huy(?: la)?|" + DON + r"(?: bi| da)? huy chiem|(?:bao nhieu|nhieu|may) ?(?:%|phan tram) (?:cac )?" + DON + r"(?: hang)?(?: bi| da)? huy",
        "THR": r"(?:bao nhieu|may|so) (?:luong )?" + DON + r"\b[^\n]{0,50}?thrsym",
        "NDAY": r"(?:bao nhieu|may|so) ngay\b[^\n]{0,70}?thrsym",
        "NITEM": r"thrsym " + DON + r"(?: hang)? khac nhau",
    }
    if "thrsym" in gb and not (re.search(spans["THR"], gb) or re.search(spans["NDAY"], gb) or re.search(spans["NITEM"], gb)):
        return _no()
    found = {}
    for k, pat in spans.items():
        m = re.search(pat, gb)
        if m:
            found[k] = m
    kinds = set(found)
    extra = set()
    intent = None
    if "THR" in kinds or "NDAY" in kinds or "NITEM" in kinds:
        if thr is None:
            return _no()
        pri = [k for k in ("THR", "NDAY", "NITEM") if k in kinds]
        if len(pri) != 1:
            return _no()
        intent = pri[0]
        mm = found[intent]
        if intent in ("THR", "NDAY"):
            mid = re.sub(r"^(?:bao nhieu|may|so) (?:luong )?(?:" + DON + r"|ngay)\b", " ", mm.group(0))
            mid = re.sub(r"thrsym$", " ", mid)
            if not re.search(r"doanh thu|dthu|tien hang|gia tri|tong tien|\btien\b|\bban\b", mid):
                return _no()
            if intent == "NDAY":
                mid = re.sub(r"ca ngay", " ", mid)
            mid = re.sub(r"doanh thu|dthu|tien hang|gia tri|tong tien|\btien\b|\bTIME\b|\bban (?:duoc|dc|ra)\b|\bban\b", " ", mid)
            if set(mid.split()) - {"co", "ma", "dat", "cua", "hang", "trong", "nao", "duoc", "dc", "don", "ngay", "doanh", "thu"}:
                return _no()
        gb = re.sub(spans[intent], " INTENT ", gb, count=1)
        if intent == "NITEM":
            if not re.search(r"mat hang|san pham|\bmon\b", gb):
                return _no()
            gb = re.sub(r"(?:dem )?(?:co )?(?:bao nhieu|may|so)? ?(?:luong )?(?:mat hang|san pham|mon)(?: nao)?(?: co mat o| duoc mua o| xuat hien o| duoc ban o| co mat| o)?", " ", gb)
            gb = re.sub(r"\bINTENT\b(?: tro len)?", " INTENT ", gb)
            gb = re.sub(r"\b(?:dem|so|bao nhieu|may)\b", " ", gb)
        else:
            gb = re.sub(r"\bINTENT\b(?: tro len)?", " INTENT ", gb)
        gb = re.sub(r"\b(?:thrsym)\b", " ", gb)
    elif "RATE" in kinds:
        if len(kinds) > 1 and kinds - {"RATE", "HUY", "CNT"}:
            return _no()
        intent = "RATE"
        gb = re.sub(spans["RATE"], " INTENT ", gb, count=1)
        gb = re.sub(r"\b(?:bao nhieu|may)\b ?(?:%|phan tram)?|%|\bphan tram\b|\btong so don\b", " ", gb)
        extra = set()
    elif "AVG" in kinds:
        if kinds - {"AVG", "REV", "CNT"}:
            return _no()
        intent = "AVG"
        gb = re.sub(spans["AVG"], " INTENT ", gb, count=1)
        for k in ("REV", "CNT"):
            gb = re.sub(spans[k], " INTENT ", gb)
        gb = re.sub(r"\b(?:mang|dem) (?:ve|lai)\b(?=\s+(?:bao nhieu|bn|nhieu)\s+(?:INTENT|doanh thu|dthu|tien))", " ", gb)
    elif "HUY" in kinds:
        if kinds - {"HUY", "CNT"}:
            return _no()
        intent = "HUY"
        gb = re.sub(spans["HUY"], " INTENT ", gb, count=1)
    elif len(kinds) == 1:
        intent = next(iter(kinds))
        gb = re.sub(spans[intent], " INTENT ", gb, count=1)
    elif kinds == {"REV", "CNT"}:
        return _no()   # nhiều số trong một câu
    else:
        if item and "ITEM" in gb and re.search(r"\bban\b", gb) and not kinds:
            intent = "QTY"
        else:
            return _no()
    unit = None
    if intent == "QTY":
        mu = re.search(r"(?:bao nhieu|nhieu|may|bn|so|tong so) (?!ITEM)(\w+) ITEM", gb) or \
            re.search(r"(?:bao nhieu|may|bn|tong so|so luong) ?(?:\w+ )?(?<!ITEM )(\w+)(?= *$)|(?:bao nhieu|nhieu|may|bn) (?!ITEM|TIME)(\w+)", gb)
        if not mu:
            WHY.append("GB:" + gb)
            return _no()
        unit = [x for x in mu.groups() if x][0]
        if unit in ("ban", "mua", "duoc", "ra", "dc", "tong", "cong"):
            return _no()
        gb = gb[:mu.start()] + " INTENT " + gb[mu.end():]
        gb = re.sub(r"\b(?:ban|mua)(?: ?(?:duoc|ra|dc))?(?: tong cong)?\b", " ", gb)
    if item is not None and intent not in ("REV", "CNT", "QTY"):
        return _no()
    if item is None and intent == "QTY":
        return _no()
    # --- phần còn lại phải vô hại
    rest = [t for t in re.sub(r"\b(?:ITEM|TIME|INTENT)\b", " ", gb).split()]
    ok_extra = {"o", "dem"} if intent == "NITEM" else set()
    for t in rest:
        if t in _OK or t in _TIME_TOK or t in ok_extra:
            continue
        WHY.append("TOK:" + t)
        return _no()
    if intent == "KHACH" and not _co_luat(rulebook, "khach"):
        return _no()
    if intent == "AVG" and not _co_luat(rulebook, "tb"):
        return _no()

    d = ban["ngay"].dt.normalize()
    inp = (d >= st) & (d <= en)
    b = ban[ban["tinh_dt"] & inp]
    if item is not None:
        b = b[(b["ten_hang"] == item)]
        if "la_phi" in b and b["la_phi"].any():
            return _no()
        if "ma_hang" in ban and ban.loc[ban["ten_hang"] == item, "ma_hang"].nunique() > 1:
            return _no()
    mo = f"{intent} kỳ {p.label}" + (f" mặt hàng {item}" if item else "")
    if intent == "REV":
        return {"type": "number", "answer": _tien_quy(b["thanh_tien"].sum())}, "Doanh thu = tổng thanh_tien dòng tinh_dt. " + mo
    if intent == "CNT":
        return {"type": "number", "answer": int(b["ma_don"].nunique())}, "Số đơn tính doanh thu. " + mo
    if intent == "AVG":
        n = b["ma_don"].nunique()
        if n == 0:
            return _no()
        v = (Decimal(repr(round(float(b["thanh_tien"].sum()), 6))) / Decimal(n)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return {"type": "number", "answer": int(v)}, "Doanh thu / số đơn, làm tròn nửa lên. " + mo
    if intent == "KHACH":
        if "khach" not in b or ban["khach"].notna().sum() == 0:
            return _no()
        return {"type": "number", "answer": int(b["khach"].nunique())}, "Số khách (cột khach) có đơn tính doanh thu. " + mo
    if intent in ("HUY", "RATE"):
        if "trang_thai" not in ban:
            return _no()
        a = ban[inp]
        nh = a.loc[a["trang_thai"] == "huỷ", "ma_don"].nunique()
        if intent == "HUY":
            return {"type": "number", "answer": int(nh)}, "Số đơn huỷ trong kỳ. " + mo
        na = a["ma_don"].nunique()
        if na == 0:
            return _no()
        v = (Decimal(int(nh)) * 100 / Decimal(int(na))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {"type": "percent", "answer": float(v)}, "Đơn huỷ / mọi đơn đặt trong kỳ. " + mo
    if intent == "THR":
        s_ = b.groupby("ma_don")["thanh_tien"].sum()
        return {"type": "number", "answer": int((s_.round(6) >= thr).sum())}, f"Số đơn có doanh thu >= {thr}. " + mo
    if intent == "NDAY":
        s_ = b.groupby(b["ngay"].dt.normalize())["thanh_tien"].sum()
        return {"type": "number", "answer": int((s_.round(6) >= thr).sum())}, f"Số ngày doanh thu >= {thr}. " + mo
    if intent == "NITEM":
        x = b[~b["la_phi"]] if "la_phi" in b else b
        key = "ma_hang" if "ma_hang" in x and x["ma_hang"].notna().any() else "ten_hang"
        c = x.groupby(key)["ma_don"].nunique()
        return {"type": "number", "answer": int((c >= thr).sum())}, f"Số mặt hàng có mặt ở >= {thr} đơn. " + mo
    if intent == "QTY":
        sub = ban[ban["ten_hang"] == item]
        if "don_vi_co_ban" not in sub or sub["don_vi_co_ban"].dropna().nunique() != 1:
            return _no()
        if fold(str(sub["don_vi_co_ban"].dropna().iloc[0])) != unit:
            return _no()
        col = "so_luong_co_ban"
        if col not in b or b[col].isna().any():
            return _no()
        return {"type": "number", "answer": _sl_fmt(b[col].sum())}, "Số lượng quy về đơn vị cơ bản. " + mo
    return _no()
