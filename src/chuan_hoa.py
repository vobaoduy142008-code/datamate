"""
Diễn giải từ viết tắt / từ lóng / tiếng địa phương trong câu hỏi (vd "dthu th ni bán đc bn đơn z?").

Nguồn: docs/tu_dien_viet_tat.md (từ điển đội tổng hợp). Chỉ lấy những từ AN TOÀN trong ngữ cảnh hỏi số liệu
cửa hàng: từ nào có nhiều nghĩa dễ nhầm (k, m, t, s, ck, tt, chi, dt...) thì KHÔNG đổi, để model tự hiểu.
Câu gốc vẫn được đưa cho model; câu diễn giải chỉ đi kèm để model đỡ hiểu sai.

    chuan_hoa("Dthu th ni bán đc bn z?")  ->  "doanh thu tháng này bán được bao nhiêu vậy?"
"""
import re

# (mẫu, thay bằng, phân biệt hoa thường?)  -- mẫu khớp TRỌN TỪ
_MAP = [
    # không / được / gì / chưa / vậy
    (r"ko|khum|hok|hông|hong|hổng", "không", False),
    (r"kh", "không", True),                       # "kh" thường = không; "KH" hoa = khách hàng (dưới)
    (r"đc|dc|đk", "được", False),
    (r"j|gj", "gì", False),
    (r"chx", "chưa", False),
    (r"z|zậy|zay|dzậy|dzay", "vậy", False),
    (r"rứa", "vậy", False),
    (r"làm răng|mần răng", "làm sao", False),              # "răng" một mình còn là "răng" (kem đánh răng)
    (r"chừng mô|khi mô|hồi mô|bao h|bh", "khi nào", False),
    (r"(?<=ở )mô|(?<=đi )mô|(?<=tới )mô|(?<=đến )mô|(?<=về )mô", "đâu", False),
    (r"mô", "nào", False),                                  # "mặt hàng mô" = mặt hàng nào
    (r"ni", "này", False),
    (r"chừ", "giờ", False),
    (r"mơi", "ngày mai", False),
    (r"(?<=chi nhánh )tê|(?<=bên )tê|(?<=xưởng )tê|(?<=kho )tê|(?<=shop )tê|(?<=cửa hàng )tê", "kia", False),
    (r"chộ", "thấy", False),
    (r"(?<=tới )h|(?<=đến )h|(?<=toi )h|(?<=den )h", "giờ", False),
    (r"cái đồ (?:ni|này)|món (?:ni|này)|con hàng (?:ni|này)|đồ (?:ni|này)", "mặt hàng này", False),
    (r"bữa ni|bữa nay|hnay", "hôm nay", False),
    (r"bữa qua|hqua", "hôm qua", False),
    (r"hkia", "hôm kia", False),
    (r"bữa hổm|hôm bữa", "hôm trước", False),
    (r"hổm rày|bữa giờ", "mấy hôm nay", False),
    # bao nhiêu
    (r"bn|bnh|bnhiu|bao nhiu|bao nhiều", "bao nhiêu", False),
    (r"nhiu|nhìu", "nhiều", False),
    (r"mấy %|bao nhiêu %|mấy phần trăm", "bao nhiêu phần trăm", False),
    # từ nối hay viết tắt
    (r"trc", "trước", False),
    (r"vs", "với", False),
    (r"cx", "cũng", False),
    (r"ms", "mới", False),
    (r"lm", "làm", False),
    (r"ntn", "như thế nào", False),
    (r"th(?= (?:này|ni|nay|trước|trc|rồi|sau|liền|tới))", "tháng", False),
    # thuật ngữ cửa hàng
    (r"dthu|d\.thu|doanh thủ", "doanh thu", False),
    # "dt" vừa là doanh thu vừa là điện thoại: chỉ đổi khi đứng cạnh từ chỉ thời gian / số tiền
    (r"(?<!qua )(?<!bằng )(?<!điện )dt(?= (?:tháng|th|tuần|hôm|ngày|năm|quý|bên|của|cả|là|bao|bn|đc|được|đạt|trung|bình|tb)\b)|(?<=tổng )dt", "doanh thu", False),
    (r"DS|ds(?= (?:tháng|th|tuần|hôm|ngày|năm|quý|của|là|bao|bn|đc|được))", "doanh số", True),
    (r"mấy củ|bao nhiêu củ|bn củ", "bao nhiêu triệu đồng", False),
    (r"mấy tỏi|bn tỏi", "bao nhiêu tỷ đồng", False),
    (r"bán ế nhất|ế nhất", "bán chậm nhất", False),
    (r"lời", "lãi", False),
    (r"ktra|kt", "kiểm tra", False),
    (r"p\.mềm|pmềm", "phần mềm", False),
    (r"sđt|sdt", "số điện thoại", False),
    (r"đvvc", "đơn vị vận chuyển", False),
    (r"sz", "cỡ", False),
    (r"KH", "khách hàng", True),
    (r"NV|nv", "nhân viên", True),
    (r"NCC|ncc", "nhà cung cấp", True),
    (r"HSD|hsd", "hạn sử dụng", True),
    (r"SL|sl|slg", "số lượng", True),
    (r"ĐG|đg", "đơn giá", True),
    (r"sp", "sản phẩm", False),
    (r"mh", "mặt hàng", False),
    (r"đh|dh", "đơn hàng", False),
    (r"LN|ln", "lợi nhuận", True),
    (r"PNK", "phiếu nhập kho", True),
    (r"PXK", "phiếu xuất kho", True),
]
_COMPILED = [(re.compile(rf"(?<!\w)(?:{p})(?!\w)", 0 if cs else re.IGNORECASE), b) for p, b, cs in _MAP]

# tiền viết tắt: 500k -> 500.000 đồng; 2tr / 2 triệu / 2 củ -> 2.000.000 đồng; 1tỷ / 1 tỏi -> 1.000.000.000 đồng
_MONEY = [
    (re.compile(r"(?<![\w.,/])(\d+(?:[.,]\d+)?)\s?(k|nghìn|ngàn|cành)(?!\w)", re.IGNORECASE), 1_000),
    (re.compile(r"(?<![\w.,/])(\d+(?:[.,]\d+)?)\s?(tr|triệu|củ)(?!\w)", re.IGNORECASE), 1_000_000),
    (re.compile(r"(?<![\w.,/])(\d+(?:[.,]\d+)?)\s?(tỷ|tỏi)(?!\w)", re.IGNORECASE), 1_000_000_000),
]


def _money(m, unit):
    v = float(m[1].replace(",", "."))
    n = round(v * unit)
    return f"{n:,}".replace(",", ".") + " đồng"


# ----------------------------------------------------------------------------- sửa lỗi gõ Telex
# "naof" -> "nào", "hangf" -> "hàng", "nawm" -> "năm", "bij" -> "bị", "tir leej" -> "tỉ lệ", "chiuj" -> "chịu"
# Chỉ sửa khi kết quả là một âm tiết tiếng Việt có thật (am_tiet.txt: các âm tiết có dấu gặp trong đề train,
# rulebook, README), nên tên hàng tiếng Anh (Tiger, Oreo, Jotun...) không bị đụng tới.
import os
import unicodedata

_VOCAB = set()
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "am_tiet.txt"), encoding="utf-8") as _f:
        _VOCAB = {w.strip() for w in _f if w.strip()}
except OSError:
    pass
_TONES = {"s": "\u0301", "f": "\u0300", "r": "\u0309", "x": "\u0303", "j": "\u0323"}
_VOWELS = "aăâeêioôơuưy"


def _telex_base(w):
    for a, b in (("uow", "ươ"), ("dd", "đ"), ("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"), ("ow", "ơ"),
                 ("uw", "ư")):
        w = w.replace(a, b)
    return w.replace("w", "ư")


_EN = {"max", "mix", "fax", "tax", "box", "sales", "ads", "its", "yes", "bus", "plus", "ps", "kids", "cups", "sets",
       "bags", "pcs", "kits", "tops", "hits", "fix", "six", "wax"}


def _telex_word(w):
    lw = w.lower()
    if lw in _VOCAB or lw in _EN or not re.fullmatch(r"[a-zđ]+", lw) or len(lw) < 2:
        return w
    tone = None
    if lw[-1] in _TONES and any(c in "aeiouyw" for c in lw[:-1]):
        tone, lw = _TONES[lw[-1]], lw[:-1]
    base = _telex_base(lw)
    if base == lw and tone is None:
        return w
    cands = [base] if tone is None else []
    if tone:
        for i, ch in enumerate(base):
            if ch in _VOWELS:
                cands.append(unicodedata.normalize("NFC", base[:i + 1] + tone + base[i + 1:]))
    hits = [c for c in cands if c in _VOCAB]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else w)


def sua_telex(text: str) -> str:
    if not _VOCAB:
        return text
    def fix(m):
        w = m[0]
        if w.islower():
            return _telex_word(w)
        if w[0].isupper() and (len(w) == 1 or w[1:].islower()):     # "Thangs" đầu câu
            t = _telex_word(w.lower())
            return t[0].upper() + t[1:] if t != w.lower() else w
        return w
    return re.sub(r"[A-Za-zđĐ]+", fix, text)


def chuan_hoa(text: str) -> str:
    """Trả về câu đã diễn giải từ viết tắt. Không đổi gì thì trả lại đúng câu cũ."""
    if not text:
        return text
    s = sua_telex(text)
    for rx, unit in _MONEY:
        s = rx.sub(lambda m, u=unit: _money(m, u), s)
    for rx, b in _COMPILED:
        s = rx.sub(b, s)
    s = re.sub(r"\b(l[aà]) chi(\s*\??\s*)$", r"\1 gì\2", s)      # "... là chi?" (miền Trung) = "là gì?"
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    for q in ["Dthu th ni bán đc bn z?", "đơn trên 500k tháng trc có mấy đơn", "KH nào mua nhiều nhất th này",
              "NV nào bán đc nhiều nhất", "Cà phê sữa (M) bán đc bn ly", "tháng 8/2026 dthu bn"]:
        print(q, "->", chuan_hoa(q))
