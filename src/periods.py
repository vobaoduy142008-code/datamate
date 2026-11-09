"""
Giải nghĩa cách nói thời gian theo rulebook DataMate (§2), dùng ngày `as_of_date`.

Không để LLM tự tính ngày: gọi `find_periods(question, as_of)` để lấy sẵn các khoảng
ngày (start, end đều TÍNH CẢ HAI ĐẦU), rồi đưa vào prompt / dùng trực tiếp trong SQL.

    from periods import Calendar, find_periods
    cal = Calendar("calendar.csv")
    for p in find_periods("dthu tuần rồi so với tuần trước nữa", "2026-03-04", cal):
        print(p)

Mọi hàm đều thuần Python, không cần mạng.
"""
from __future__ import annotations

import calendar as _cal
import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

# --------------------------------------------------------------------------- Period


@dataclass(frozen=True)
class Period:
    start: Optional[date]   # None = từ đầu dữ liệu (câu hỏi "đến hết ngày X", số còn lại)
    end: date               # tính cả ngày end
    kind: str               # day | week | week_to_date | month | month_to_date | quarter | year |
                            # year_to_date | last_n_days | tet | before_tet | range | snapshot
    label: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1 if self.start else 0

    def to_dict(self):
        return {"start": self.start.isoformat() if self.start else None,
                "end": self.end.isoformat(), "kind": self.kind, "label": self.label}

    def __str__(self):
        s = self.start.isoformat() if self.start else "(đầu dữ liệu)"
        return f"{self.label}: {s} → {self.end.isoformat()} [{self.kind}]"


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def _month_end(y: int, m: int) -> date:
    return date(y, m, _cal.monthrange(y, m)[1])

# --------------------------------------------------------------------------- lịch âm


class Calendar:
    """Đọc calendar.csv: date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday"""

    def __init__(self, path: str):
        self.rows = {}
        self.by_lunar = {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = date.fromisoformat(r["date"])
                leap = r["lunar_leap"].strip().lower() == "true"
                key = (int(r["lunar_year"]), int(r["lunar_month"]), int(r["lunar_day"]), leap)
                self.rows[d] = r
                self.by_lunar.setdefault(key, d)

    def solar(self, lunar_year: int, lunar_month: int, lunar_day: int, leap=False) -> date:
        try:
            return self.by_lunar[(lunar_year, lunar_month, lunar_day, leap)]
        except KeyError:
            raise ValueError(f"Lịch không có ngày âm {lunar_day}/{lunar_month}/{lunar_year}")

# --------------------------------------------------------------------------- các kỳ cơ bản


def day(d, label="ngày") -> Period:
    d = _d(d)
    return Period(d, d, "day", f"{label} {d.isoformat()}")


def today(as_of) -> Period:
    return day(as_of, "hôm nay")


def yesterday(as_of) -> Period:
    return day(_d(as_of) - timedelta(days=1), "hôm qua")


def week_of(d) -> Period:
    """Cả tuần thứ Hai → Chủ nhật chứa ngày d."""
    d = _d(d)
    mon = d - timedelta(days=d.weekday())
    return Period(mon, mon + timedelta(days=6), "week", f"tuần {mon.isoformat()}")


def this_week(as_of) -> Period:
    """Tuần này: thứ Hai của tuần chứa hôm nay → hết hôm nay."""
    a = _d(as_of)
    return Period(a - timedelta(days=a.weekday()), a, "week_to_date", "tuần này")


def last_week(as_of) -> Period:
    """Tuần trước / tuần rồi: CẢ tuần liền trước, thứ Hai → Chủ nhật."""
    p = week_of(_d(as_of) - timedelta(days=7))
    return Period(p.start, p.end, "week", "tuần trước")


def month(y: int, m: int) -> Period:
    return Period(date(y, m, 1), _month_end(y, m), "month", f"tháng {m}/{y}")


def month_no_year(m: int, as_of) -> Period:
    """Tháng M không ghi năm: năm nay nếu không sau hôm nay, ngược lại năm trước."""
    a = _d(as_of)
    y = a.year if (a.year, m) <= (a.year, a.month) else a.year - 1
    return month(y, m)


def this_month(as_of) -> Period:
    """Tháng này: từ ngày 1 đến hết hôm nay."""
    a = _d(as_of)
    return Period(date(a.year, a.month, 1), a, "month_to_date", "tháng này")


def last_month(as_of) -> Period:
    a = _d(as_of)
    y, m = (a.year, a.month - 1) if a.month > 1 else (a.year - 1, 12)
    p = month(y, m)
    return Period(p.start, p.end, "month", "tháng trước")


def quarter(y: int, q: int) -> Period:
    m0 = 3 * (q - 1) + 1
    return Period(date(y, m0, 1), _month_end(y, m0 + 2), "quarter", f"quý {q}/{y}")


def this_quarter(as_of) -> Period:
    """Quý này: từ ngày đầu quý đến hết hôm nay (như "tháng này")."""
    a = _d(as_of)
    q = (a.month - 1) // 3 + 1
    return Period(date(a.year, 3 * (q - 1) + 1, 1), a, "quarter_to_date", "quý này")


def last_quarter(as_of) -> Period:
    a = _d(as_of)
    q = (a.month - 1) // 3 + 1
    y, q = (a.year, q - 1) if q > 1 else (a.year - 1, 4)
    p = quarter(y, q)
    return Period(p.start, p.end, "quarter", "quý trước")


def quarter_no_year(q: int, as_of) -> Period:
    a = _d(as_of)
    cur_q = (a.month - 1) // 3 + 1
    return quarter(a.year if q <= cur_q else a.year - 1, q)


def year(y: int) -> Period:
    return Period(date(y, 1, 1), date(y, 12, 31), "year", f"năm {y}")


def year_to_date(as_of) -> Period:
    a = _d(as_of)
    return Period(date(a.year, 1, 1), a, "year_to_date", "từ đầu năm")


def last_n_days(n: int, as_of) -> Period:
    """N ngày gần nhất, kể cả hôm nay."""
    a = _d(as_of)
    return Period(a - timedelta(days=n - 1), a, "last_n_days", f"{n} ngày gần nhất")


def next_n_days(n: int, as_of) -> Period:
    """N ngày tới (vd. hết hạn trong N ngày tới): từ hôm nay D đến hết D + N."""
    a = _d(as_of)
    return Period(a, a + timedelta(days=n), "next_n_days", f"{n} ngày tới")


def date_range(d1, d2, label="") -> Period:
    """Từ X đến Y: gồm cả X và Y."""
    d1, d2 = _d(d1), _d(d2)
    return Period(d1, d2, "range", label or f"từ {d1} đến {d2}")


def snapshot(d, label="đến hết ngày") -> Period:
    """Số còn lại (nợ, tồn…): mọi phát sinh từ đầu dữ liệu đến hết ngày d."""
    d = _d(d)
    return Period(None, d, "snapshot", f"{label} {d.isoformat()}")


def tet(y: int, cal: Calendar) -> Period:
    """Tết năm Y: 23 tháng Chạp (năm âm Y-1) → hết mùng 10 tháng Giêng (năm âm Y)."""
    s = cal.solar(y - 1, 12, 23)
    e = cal.solar(y, 1, 10)
    return Period(s, e, "tet", f"Tết {y}", {"year": y})


def before_tet(y: int, cal: Calendar, n: Optional[int] = None) -> Period:
    """Trước Tết = cùng số ngày (mặc định bằng độ dài dịp Tết) ngay trước 23 tháng Chạp."""
    t = tet(y, cal)
    n = n or t.days
    return Period(t.start - timedelta(days=n), t.start - timedelta(days=1),
                  "before_tet", f"{n} ngày trước Tết {y}", {"year": y})

# --------------------------------------------------------------------------- kỳ so sánh


def previous(p: Period, cal: Optional[Calendar] = None) -> Period:
    """'Kỳ trước' của p, dùng cho % thay đổi / tăng trưởng (rulebook §1, §2)."""
    if p.kind == "day":
        return day(p.start - timedelta(days=1), "ngày trước")
    if p.kind in ("week", "week_to_date"):
        s = p.start - timedelta(days=7)
        return Period(s, s + timedelta(days=p.days - 1), p.kind, "tuần trước đó")
    if p.kind == "month":
        y, m = (p.start.year, p.start.month - 1) if p.start.month > 1 else (p.start.year - 1, 12)
        return month(y, m)
    if p.kind == "month_to_date":
        # cùng số ngày của tháng trước; tháng trước không có ngày đó thì đến hết tháng trước
        y, m = (p.start.year, p.start.month - 1) if p.start.month > 1 else (p.start.year - 1, 12)
        e = min(p.end.day, _cal.monthrange(y, m)[1])
        return Period(date(y, m, 1), date(y, m, e), "month_to_date", "cùng kỳ tháng trước")
    if p.kind == "quarter":
        q = (p.start.month - 1) // 3 + 1
        return quarter(p.start.year, q - 1) if q > 1 else quarter(p.start.year - 1, 4)
    if p.kind == "year":
        return year(p.start.year - 1)
    if p.kind == "tet":
        if cal is None:
            raise ValueError("Cần Calendar để lấy Tết năm trước")
        return tet(p.meta["year"] - 1, cal)
    # last_n_days, range, …: N ngày ngay trước
    n = p.days
    return Period(p.start - timedelta(days=n), p.start - timedelta(days=1), p.kind,
                  f"{n} ngày ngay trước")


def same_period_last_year(p: Period, cal: Optional[Calendar] = None) -> Period:
    """'Cùng kỳ năm trước / năm ngoái'."""
    if p.kind == "tet":
        return tet(p.meta["year"] - 1, cal)
    if p.kind == "month":
        return month(p.start.year - 1, p.start.month)
    if p.kind == "quarter":
        return quarter(p.start.year - 1, (p.start.month - 1) // 3 + 1)

    def back(d):
        try:
            return d.replace(year=d.year - 1)
        except ValueError:          # 29/02
            return d.replace(year=d.year - 1, day=28)
    return Period(back(p.start) if p.start else None, back(p.end), p.kind, "cùng kỳ năm trước")


def clip_to_data(p: Period, data_min, data_max) -> Optional[Period]:
    """Rulebook §7: kỳ nằm HOÀN TOÀN ngoài khoảng dữ liệu → None (unanswerable).
    Kỳ chỉ một phần nằm trong dữ liệu → tính trên phần có dữ liệu."""
    lo, hi = _d(data_min), _d(data_max)
    s = p.start or lo
    if p.end < lo or s > hi:
        return None
    return Period(max(s, lo) if p.start else None, min(p.end, hi), p.kind, p.label, p.meta)

# --------------------------------------------------------------------------- chuẩn hoá câu hỏi


_TELEX_TONE = re.compile(r"(?<=[a-z])[sfrxj]$")


def fold(text: str) -> str:
    """Bỏ dấu + sửa lỗi gõ Telex phổ biến ('nawm'→'nam', 'thangs'→'thang', 'ddos'→'do').
    CHỈ dùng để dò mẫu câu, không dùng để so tên hàng."""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    def fix(m):
        w = m[0]
        if len(w) <= 2 or w in ("bus", "max"):
            return w
        w = re.sub(r"dd", "d", w)
        w = re.sub(r"([aeo])\1", r"\1", w)
        w = re.sub(r"(?<=[aou])w", "", w)
        return _TELEX_TONE.sub("", w)
    s = re.sub(r"[a-z]+", fix, s)
    # viết tắt, tiếng địa phương hay gặp
    for a, b in [(r"\bth\b(?= (truoc|roi|nay|ni|sau|lien))", "thang"), (r"\btrc\b", "truoc"),
                 (r"\bhnay\b", "hom nay"), (r"\bhqua\b", "hom qua"), (r"\bhkia\b", "hom kia"),
                 (r"\bbua (nay|ni)\b", "hom nay"), (r"\bbua qua\b", "hom qua"),
                 (r"\b(hom|thang|tuan|nam|quy) ni\b", r"\1 nay"),
                 (r"\bvs\b", "voi"), (r"\bdthu\b", "doanh thu"),
                 (r"\bq\s?([1-4])\b", r"quy \1"),                      # Q2/2025, q3 -> quý
                 (r"\b(toi|den) chu\b", r"\1 nay")]:                     # "tới chừ" = tới giờ
        s = re.sub(a, b, s)
    # tháng viết tắt: T8/2026, th3/2025, th 4/25, Th3
    s = re.sub(r"\b(?:th|t)\s?(?=\d{1,2}\s*[/-]\s*\d{2,4}\b)", "thang ", s)
    s = re.sub(r"\bth\s?(?=\d{1,2}\b)", "thang ", s)
    s = re.sub(r"\b(thang|quy) (\d{1,2})\s*[/-]\s*(\d{2})\b(?![/\d])", r"\1 \2/20\3", s)
    # tháng không có chữ "tháng": 8-2025, 4/2026 (không phải phần đuôi của dd/mm/yyyy)
    s = re.sub(r"(?<![\d/])(?<!thang )(?<!quy )(\d{1,2})\s*[/-]\s*(20\d{2})\b", r"thang \1/\2", s)
    # M/YY với YY > 12 (8/26, 12/25) → tháng; dd/mm thì giữ nguyên
    s = re.sub(r"(?<![\d/])(?<!thang )(?<!quy )(\d{1,2})/(1[3-9]|[2-3]\d)\b(?![/\d])",
               lambda m: f"thang {m[1]}/20{m[2]}" if 1 <= int(m[1]) <= 12 else m[0], s)
    # tháng bằng chữ
    words = [("muoi mot", 11), ("muoi hai", 12), ("chap", 12), ("gieng", 1), ("mot", 1), ("hai", 2),
             ("ba", 3), ("tu", 4), ("bon", 4), ("nam", 5), ("bay", 7), ("tam", 8),
             ("chin", 9), ("muoi", 10)]
    for w, n in words:
        # "3 tháng từ tháng 7" / "tháng từ ngày..." : "tu" là "từ", không phải tháng tư
        s = re.sub(rf"(?<!\d )\bthang {w}\b(?! \d)(?! (nay|truoc|roi|ngoai|thang|ngay|tuan|hom|dau|cuoi|quy|nam \d)\b)", f"thang {n}", s)
    return re.sub(r"\s+", " ", s).strip()

# --------------------------------------------------------------------------- dò kỳ trong câu


_NUM = r"(\d{1,2})"
_Y = r"(\d{4})"
_SEP = r"\s*(?:/|-|nam)\s*"


def _rel_prev(p: Period) -> Period:
    return Period(p.start, p.end, p.kind, p.label, {**p.meta, "rel_prev": True})


def find_periods(question: str, as_of, cal: Optional[Calendar] = None) -> list[Period]:
    """Trả về các kỳ nhắc tới trong câu theo thứ tự xuất hiện.
    Ngoài ra, nếu câu có 'so với/cùng kỳ/tăng giảm', thêm kỳ so sánh vào meta['compare']."""
    a = _d(as_of)
    s = fold(question)
    # "N ngày kể từ ngày D (tính cả D) so với N ngày ngay trước D" (hiệu ứng đổi giá): kỳ hỏi + kỳ so sánh riêng
    mn = re.search(r"(\d+) ngay (?:ke |tinh )?tu (?:ngay )?(?:do|nay|dd|\d)", s)
    md = re.search(rf"(?:ngay )?{_NUM}/{_NUM}(?:/{_Y})?", s)
    if mn and md and re.search(r"truoc", s[mn.end():] + s[:mn.start()]):
        try:
            n = int(mn[1])
            dd, mm, yy = int(md[1]), int(md[2]), md[3]
            d0 = date(int(yy), mm, dd) if yy else (date(a.year, mm, dd) if date(a.year, mm, dd) <= a else date(a.year - 1, mm, dd))
            if 1 <= n <= 366:
                cmp = Period(d0 - timedelta(days=n), d0 - timedelta(days=1), "range", f"{n} ngày ngay trước {d0:%d/%m/%Y}")
                return [Period(d0, d0 + timedelta(days=n - 1), "range", f"{n} ngày kể từ {d0:%d/%m/%Y} (tính cả ngày đó)",
                               {"compare": cmp})]
        except ValueError:
            pass
    found: list[tuple[int, Period]] = []
    taken = []

    def add(m, fn):
        if any(m.start() < e and m.end() > b for b, e in taken):
            return
        try:
            p = fn() if callable(fn) else fn
        except ValueError:
            return
        taken.append((m.start(), m.end()))
        found.append((m.start(), p))

    def ddmm(dd, mm, yy):
        dd, mm = int(dd), int(mm)
        if yy:
            return date(int(yy), mm, dd)
        d = date(a.year, mm, dd)
        return d if d <= a else date(a.year - 1, mm, dd)

    # từ ngày X đến (hết) ngày Y
    for m in re.finditer(rf"tu (?:ngay )?{_NUM}/{_NUM}(?:/{_Y})? (?:den|toi) (?:het )?(?:ngay )?{_NUM}/{_NUM}(?:/{_Y})?", s):
        add(m, lambda: date_range(ddmm(m[1], m[2], m[3]), ddmm(m[4], m[5], m[6] or m[3])))
    # từ ngày X đến (hết) hôm nay / đến nay
    for m in re.finditer(rf"tu (?:ngay )?{_NUM}/{_NUM}(?:/{_Y})?(?: nam nay)? (?:den|toi) (?:het )?(?:ngay )?(?:hom nay|nay|gio|bay gio)", s):
        add(m, lambda: date_range(ddmm(m[1], m[2], m[3]), a, m[0]))
    # từ tháng A/Y đến tháng B/Y
    for m in re.finditer(rf"tu thang {_NUM}{_SEP}{_Y} (?:den|toi) (?:het )?thang {_NUM}{_SEP}{_Y}", s):
        add(m, lambda: Period(month(int(m[2]), int(m[1])).start, month(int(m[4]), int(m[3])).end,
                              "range", m[0], {"months": True}))
    # đến hết ngày X / tính đến ngày X  (số còn lại)
    for m in re.finditer(rf"(?:den het|tinh den|den cuoi) (?:ngay )?{_NUM}/{_NUM}(?:/{_Y})?", s):
        add(m, lambda: snapshot(ddmm(m[1], m[2], m[3])))
    for m in re.finditer(r"(?:den het|tinh den|den cuoi) (?:ngay )?hom nay", s):
        if not re.search(r"dau (thang|nam|tuan) (nay )?den het (ngay )?hom nay", s):
            add(m, lambda: snapshot(a, "đến hết hôm nay"))
    # tháng M/Y, tháng M năm Y
    for m in re.finditer(rf"thang {_NUM}{_SEP}{_Y}", s):
        add(m, lambda: month(int(m[2]), int(m[1])))
    # quý Q/Y, quý Q năm Y
    for m in re.finditer(rf"quy {_NUM}{_SEP}{_Y}", s):
        add(m, lambda: quarter(int(m[2]), int(m[1])))
    for m in re.finditer(r"quy ([1-4]) (20\d\d)\b", s):          # "quý 2 2025"
        add(m, lambda: quarter(int(m[2]), int(m[1])))
    # ngày dd/mm/yyyy, dd/mm
    for m in re.finditer(rf"(?:ngay )?{_NUM}/{_NUM}(?:/{_Y})?", s):
        try:
            add(m, lambda: day(ddmm(m[1], m[2], m[3])))
        except ValueError:
            pass
    # Tết
    for m in re.finditer(r"tet(?: am lich)?(?: nam)? ?(\d{4}|nay|ngoai|truoc|roi)?", s):
        if cal is None:
            continue
        w = m[1]
        y = int(w) if w and w.isdigit() else (a.year - 1 if w in ("ngoai", "truoc", "roi") else a.year)
        if re.match(r"tet (nam )?(truoc|roi|ngoai)", m[0]) and re.search(r"tet nam nay", s):
            y = a.year - 1
        mm = re.search(r"(\d+) ngay truoc tet", s)
        if re.search(r"truoc tet", s[max(0, m.start() - 12):m.start() + 4]):
            add(m, lambda: before_tet(y, cal, int(mm[1]) if mm else None))
        else:
            add(m, lambda: tet(y, cal))
    # tương đối
    rel = [
        (r"tu dau thang (nay )?den het (ngay )?hom nay|tu dau thang( nay)?|thang nay", lambda: this_month(a)),
        (r"(tu )?dau nam( nay)?( (den|toi) (het )?(ngay )?(hom nay|nay|gio|chu))?", lambda: year_to_date(a)),
        (r"tuan nay", lambda: this_week(a)),
        (r"quy nay", lambda: this_quarter(a)),
        (r"(?<!cung ky )(?<!hon )(?<!voi )quy (truoc|roi|vua roi)", lambda: _rel_prev(last_quarter(a))),
        (r"tuan (truoc|roi|vua roi)", lambda: _rel_prev(last_week(a))),
        (r"(?<!cung ky )(?<!hon )(?<!voi )thang (truoc|roi|vua roi)(?! do)", lambda: _rel_prev(last_month(a))),
        (r"hom qua", lambda: yesterday(a)),
        (r"hom kia", lambda: day(a - timedelta(days=2), "hôm kia")),
        (r"hom nay", lambda: today(a)),
        (r"nam ngoai|nam truoc(?! do)", lambda: year(a.year - 1)),
    ]
    for pat, fn in rel:
        for m in re.finditer(pat, s):
            add(m, fn())
    for m in re.finditer(r"(\d+) ngay (gan nhat|gan day|vua qua|qua|cuoi)", s):
        add(m, lambda: last_n_days(int(m[1]), a))
    for m in re.finditer(r"(\d+) ngay (toi|sap toi|nua|ke tiep)", s):
        add(m, lambda: next_n_days(int(m[1]), a))
    for m in re.finditer(rf"thang {_NUM}(?![/\d]| nam)", s):
        mo = int(m[1])
        if 1 <= mo <= 12:
            add(m, lambda: month_no_year(mo, a))
    for m in re.finditer(rf"quy {_NUM}(?![/\d]| nam)", s):
        if 1 <= int(m[1]) <= 4:
            add(m, lambda: quarter_no_year(int(m[1]), a))
    for m in re.finditer(rf"nam {_Y}", s):
        add(m, lambda: year(int(m[1])))

    periods = [p for _, p in sorted(found, key=lambda t: t[0])]
    # "tháng 8/2026 hơn hay kém THÁNG TRƯỚC": tháng trước ở đây là kỳ liền trước tháng 8/2026 (kỳ so sánh),
    # không phải tháng trước của hôm nay -> bỏ, để phần dưới tự tính kỳ so sánh từ kỳ đầu tiên.
    if (len(periods) >= 2 and re.search(r"so (voi )?|tang|giam|thay doi|hon hay kem|hon|kem|chenh", s)
            and any(p.meta.get("rel_prev") for p in periods[1:]) and not periods[0].meta.get("rel_prev")):
        periods = [periods[0]] + [p for p in periods[1:] if not p.meta.get("rel_prev")]

    # kỳ so sánh ngầm
    if len(periods) == 1:
        p = periods[0]
        cmp = None
        if re.search(r"cung ky nam (truoc|ngoai)|so voi nam (truoc|ngoai)", s):
            cmp = same_period_last_year(p, cal)
        elif re.search(r"so (voi )?|cung ky|tang|giam|thay doi|hon hay kem|chenh", s) and p.kind != "snapshot":
            cmp = previous(p, cal)
        if cmp is not None:
            periods[0] = Period(p.start, p.end, p.kind, p.label, {**p.meta, "compare": cmp})
    return periods


def describe(periods: list[Period]) -> str:
    """Chuỗi ngắn để chèn vào prompt LLM."""
    lines = []
    for p in periods:
        lines.append(f"- {p}")
        if "compare" in p.meta:
            lines.append(f"  · kỳ so sánh: {p.meta['compare']}")
    return "\n".join(lines) or "- (không có mốc thời gian; xem rulebook)"
