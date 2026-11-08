"""
Agent v1: LLM viết code pandas -> chạy -> xem kết quả -> (sửa) -> gán ANSWER.

Luồng cho mỗi câu hỏi:
    1. Đọc dữ liệu shop bằng loader (có cache), lấy bản mô tả bảng (schema_text).
    2. Tính sẵn các khoảng thời gian trong câu hỏi bằng periods.py (LLM không phải tự tính ngày).
    3. Gửi cho LLM: rulebook + mô tả bảng + câu hỏi. LLM trả về 1 khối code Python.
    4. Chạy code. Code in ra (print) để xem dữ liệu, hoặc gán biến ANSWER khi đã có đáp án.
       Kết quả in ra / lỗi được gửi lại cho LLM; tối đa MAX_STEPS lượt.
    5. Kiểm tra và chuẩn hoá ANSWER đúng format §9 của rulebook. Không ra được đáp án hợp lệ -> abstain.

Hợp đồng (interface) cả đội dùng chung — KHÔNG đổi tên/kiểu trả về khi chưa báo cả nhóm:
    solve(question: dict, data_dir: str) -> dict
        question = {"id", "business_id", "question", "as_of_date"}
        trả về   = {"answer": {...}, "thinking": ..., "code": ..., "output": ...}
Luật thi: KHÔNG viết logic theo id câu hỏi hay business_id. Mọi thứ ở đây chung cho mọi shop.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import re
import signal
import time
import traceback
import unicodedata
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd

import llm
from loader import fold, load_business
from sales import build_sales
from periods import Calendar, find_periods

MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "4"))      # số lượt LLM tối đa cho 1 câu
CODE_TIMEOUT = int(os.environ.get("AGENT_CODE_TIMEOUT", "30"))  # giây chạy tối đa cho 1 khối code
OUT_LIMIT = 3500                                               # số ký tự kết quả in ra gửi lại cho LLM
TIME_BUDGET = int(os.environ.get("AGENT_TIME_BUDGET", "300"))  # giây tối đa cho 1 câu; quá thì bỏ qua (abstain)
VERBOSE = os.environ.get("AGENT_VERBOSE", "1") == "1"          # in tiến độ từng lượt ra màn hình
# Đáp án bằng 0 / danh sách rỗng gần như luôn là do code lọc sai (train: chỉ 4/3000 đáp án là 0, không có list rỗng).
# Sai bị -1 điểm, bỏ qua 0 điểm -> gặp kết quả này thì bỏ qua (abstain).
ABSTAIN_ZERO = os.environ.get("AGENT_ABSTAIN_ZERO", "1") == "1"

# --------------------------------------------------------------------------- bộ nhớ đệm


_BIZ = OrderedDict()            # giữ tối đa vài shop trong RAM (câu hỏi được chạy theo nhóm shop)
_STATIC = {}                    # rulebook, calendar theo data_dir


def _business(data_dir, business_id):
    key = (data_dir, business_id)
    if key in _BIZ:
        _BIZ.move_to_end(key)
        return _BIZ[key]
    biz = load_business(os.path.join(data_dir, "data", business_id))
    _BIZ[key] = biz
    while len(_BIZ) > 3:
        _BIZ.popitem(last=False)
    return biz


def _static(data_dir):
    if data_dir not in _STATIC:
        rb = os.path.join(data_dir, "rulebook.md")
        cal = os.path.join(data_dir, "calendar.csv")
        _STATIC[data_dir] = (open(rb, encoding="utf-8").read() if os.path.exists(rb) else "",
                             Calendar(cal) if os.path.exists(cal) else None)
    return _STATIC[data_dir]

# --------------------------------------------------------------------------- prompt


SYSTEM_TEMPLATE = """Bạn là trợ lý phân tích dữ liệu bán hàng của một cửa hàng nhỏ ở Việt Nam.
Bạn trả lời câu hỏi của chủ shop bằng cách VIẾT CODE PYTHON (pandas) chạy trên dữ liệu thật của shop.

# LUẬT TÍNH (rulebook) — đáp án chuẩn tính theo các luật này
{rulebook}

# CÁCH LÀM VIỆC
- Mỗi lượt, trả lời bằng ĐÚNG MỘT khối ```python ... ```. Code sẽ được chạy; mọi thứ bạn print() sẽ được gửi lại cho bạn.
- Có tối đa {max_steps} lượt. Lượt đầu nên print để xem dữ liệu (các giá trị trạng thái, tên hàng gần đúng...) nếu chưa chắc.
- Khi đã chắc đáp án, gán biến ANSWER theo đúng format §9, ví dụ:
    ANSWER = {{"type": "number", "answer": 4820000}}
    ANSWER = {{"type": "percent", "answer": 12.4}}
    ANSWER = {{"type": "list", "answer": ["Cà phê sữa (M)", "Bạc xỉu (M)"]}}
    ANSWER = {{"type": "table", "answer": {{"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000]]}}}}
    ANSWER = {{"type": "unanswerable", "reason": "shop không ghi giờ bán"}}
- Biến có sẵn trong code (KHÔNG cần đọc file):
    tables  : dict tên bảng -> DataFrame đã làm sạch (cột ngày là datetime64, cột tiền/số lượng là số).
              Đã bỏ dòng tổng/cộng, đã điền ô trống theo nhóm (§8), đã bỏ dòng trùng y hệt trong cùng đơn.
              Cột `_nguon` cho biết dòng đến từ file/sheet nào (thường không cần dùng).
    pd, np, re, math
    TODAY   : pd.Timestamp ngày hôm nay (as_of_date)
    PERIODS : list các khoảng thời gian trong câu hỏi, ĐÃ TÍNH SẴN theo §2 (dùng luôn, đừng tự tính lại).
              Danh sách PERIODS[0], PERIODS[1]... được ghi rõ ở phần CÂU HỎI (kỳ so sánh cũng là một phần tử riêng).
              Mỗi phần tử là dict {{"label", "start", "end", "kind"}}; start/end là pd.Timestamp,
              TÍNH CẢ HAI ĐẦU; start = None nghĩa là từ đầu dữ liệu.
    DATA_END: pd.Timestamp ngày cuối có dữ liệu (theo README), hoặc None
    norm(s) : chữ thường, bỏ dấu, gộp khoảng trắng — dùng để so khớp tên/trạng thái gần đúng
    in_period(series, p): Series True/False, ngày (bỏ giờ) nằm trong khoảng p (cả hai đầu)
    status_of(series): xếp cột trạng thái đơn về 4 nhóm, trả về đúng các chữ "hoàn thành" / "huỷ" / "trả hàng" / "đang giao"
    valid_orders(df, "cột trạng thái", "cột mã đơn"): Series True/False, dòng thuộc đơn TÍNH DOANH THU (§1):
              bỏ đơn huỷ và đơn có trả hàng (bỏ CẢ ĐƠN nếu một dòng trả). Bảng không có mã đơn thì bỏ tham số thứ 3.
    match(series, "tên"): Series True/False, ô khớp tên hàng/khách (không phân biệt hoa thường, dấu câu)
- Lưu ý dữ liệu:
    * Trạng thái đơn ghi nhiều kiểu khác nhau (vd "xong", "ok", "đã lấy", "completed" đều là hoàn thành;
      "hủy", "khách hủy", "cancelled" là huỷ; "trả lại", "Hoàn trả", "Đổi trả", "returned" là trả hàng).
      Hãy print các giá trị khác nhau rồi tự xếp về 4 trạng thái: hoàn thành, huỷ, trả hàng, đang giao.
    * Cột của CẢ ĐƠN (tổng đơn, khách trả, giảm giá cả đơn, phí ship...) có thể lặp lại ở mọi dòng hàng của đơn:
      phải gộp theo mã đơn (drop_duplicates / groupby().first()) trước khi cộng.
    * Đơn có trả hàng kể cả một phần là đơn trả hàng: bỏ CẢ ĐƠN khỏi doanh thu (§1).
    * So khớp tên hàng/khách không phân biệt hoa thường; tên khách có thể kèm số điện thoại.
    * Tiền là số nguyên VND (làm tròn về đồng). Phần trăm là số phần trăm, làm tròn 2 chữ số.
    * Nếu kỳ hỏi nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu, hoặc shop không ghi loại thông tin cần → unanswerable.
- Tên bảng và tên cột phải chép ĐÚNG như trong mô tả dữ liệu (giữ dấu tiếng Việt, hoa thường, khoảng trắng).
  Mỗi bảng chỉ có các cột được liệt kê dưới bảng đó; cần cột của bảng khác thì merge theo mã đơn / mã hàng.
- Dùng PERIODS có sẵn, KHÔNG tự viết lại ngày tháng, KHÔNG tự tạo list/dict khoảng ngày mới.
  Câu so sánh với kỳ trước: kỳ so sánh đã có sẵn trong PERIODS (xem phần CÂU HỎI).
- Chỉ dùng dữ liệu trong `tables`. Không đoán.

# BẢNG CHUẨN `ban_hang_chuan` — DÙNG BẢNG NÀY CHO MỌI CÂU VỀ BÁN HÀNG
Nếu shop có bảng `tables['ban_hang_chuan']` (xem mô tả ở phần DỮ LIỆU), nó đã được dựng sẵn ĐÚNG rulebook §1 và §8
từ các bảng bán hàng gốc: đã ghép đơn + chi tiết, chuẩn hoá trạng thái, đánh dấu đơn trả hàng (kể cả trả một phần
ghi ở bảng khác hoặc ghi số âm), quy số lượng về đơn vị cơ bản. Tên cột cố định:
  ma_don, ngay (ngày bán của đơn), thoi_diem (ngày giờ tạo đơn), trang_thai ('hoàn thành'/'huỷ'/'trả hàng'/'đang giao'),
  tra_hang (đơn có trả hàng), tinh_dt (True = dòng TÍNH DOANH THU: đơn không huỷ, không trả),
  ma_hang, ten_hang, la_phi (True = dòng phí, không phải mặt hàng), nhom_hang, don_vi, so_luong, so_luong_co_ban,
  don_gia, thanh_tien (thành tiền dòng sau giảm = doanh thu), gia_von (giá vốn 1 đơn vị bán, ghi trên dòng bán),
  ma_khach, ten_khach, nhom_khach, kenh, nhan_vien (mã), ten_nhan_vien.
Chỉ những cột có dữ liệu mới có mặt. Câu hỏi về nhập hàng, tồn kho, lô/hạn dùng, công nợ, thanh toán, bảng giá...
thì dùng các bảng gốc.

Công thức (B = tables['ban_hang_chuan'], p = PERIODS[0]):
- DOANH THU / bán được bao nhiêu tiền:
    b = B[in_period(B['ngay'], p) & B['tinh_dt']]
    ANSWER = {{"type": "number", "answer": round(b['thanh_tien'].sum())}}
- SỐ ĐƠN (bán được mấy đơn): b['ma_don'].nunique() với b như trên (đơn tính doanh thu). KHÔNG đếm số dòng.
- SỐ ĐƠN HUỶ: B[in_period(B['ngay'], p) & (B['trang_thai'] == 'huỷ')]['ma_don'].nunique() → "number".
  TỈ LỆ HUỶ = số đơn huỷ / B[in_period(B['ngay'], p)]['ma_don'].nunique() × 100 → "percent".
  SỐ ĐƠN TRẢ HÀNG: lọc B['tra_hang'].
- GIÁ TRỊ TRUNG BÌNH MỖI ĐƠN: doanh thu / số đơn tính doanh thu (cùng b).
- NGÀY BÁN NHIỀU TIỀN NHẤT: b.groupby('ngay')['thanh_tien'].sum().idxmax() → "date" dạng YYYY-MM-DD.
- BÁN CHẠY / top k / hot nhất / bán nhiều nhất (theo số lượng):
    g = b[~b['la_phi']].groupby('ten_hang').agg(sl=('so_luong_co_ban', 'sum'), ma=('ma_hang', 'first')).reset_index()
    g = g.sort_values(['sl', 'ma'], ascending=[False, True])     # bằng nhau: mã hàng tăng dần
    ANSWER = {{"type": "list", "answer": g['ten_hang'].head(k).tolist()}}
  (không có cột so_luong_co_ban thì dùng so_luong; không có ma_hang thì bỏ cột 'ma')
- THEO MẶT HÀNG / NHÓM / KHÁCH / KÊNH / NHÂN VIÊN: lọc thêm bằng match(), KHÔNG so sánh == với chữ tự gõ:
    b = b[match(b['ten_hang'], 'Sữa đặc Ông Thọ')]      # hoặc match(b['nhom_hang'], 'điện'), match(b['kenh'], 'zalo')
  THUỐC KÊ ĐƠN: b[b['ke_don']]; theo HOẠT CHẤT: b[b['hoat_chat'].fillna('').str.contains('Cetirizin', case=False)]
- CA SÁNG / CA CHIỀU (giờ tạo đơn, xem rulebook; train: sáng 06:00–13:59, chiều 14:00–21:59):
    h = b['thoi_diem'].dt.hour
    sang = b[(h >= 6) & (h <= 13)];   chieu = b[(h >= 14) & (h <= 21)]
- ĐƠN CÓ DOANH THU ≥ X (hoặc từ X trở lên): cộng theo đơn TRƯỚC rồi mới so:
    d = b.groupby('ma_don')['thanh_tien'].sum();   ANSWER = {{"type": "number", "answer": int((d >= X).sum())}}
- KHÁCH chi nhiều nhất / top khách (§3): có ma_khach thì gộp theo ma_khach (bỏ đơn không có mã = khách lẻ),
  trả về TÊN khách (ten_khach của mã đó), không trả số tiền:
    k = b[b['ma_khach'].notna()].groupby('ma_khach').agg(tien=('thanh_tien', 'sum'), ten=('ten_khach', 'first'))
    k = k.sort_values('tien', ascending=False);   ANSWER = {{"type": "string", "answer": k['ten'].iloc[0]}}
- GIÁ BÁN của X vào ngày D: đơn giá trên dòng bán X các đơn đặt ngày D (KỂ CẢ đơn huỷ/trả), cho MỘT ĐƠN VỊ CƠ BẢN,
  trước chiết khấu:   r = B[match(B['ten_hang'], 'X') & (B['ngay'] == pd.Timestamp('YYYY-MM-DD'))]
    gia = (r['don_gia'] * r['so_luong'] / r['so_luong_co_ban']).round().mode().iloc[0]
- HIỆU ỨNG TĂNG/ĐỔI GIÁ ngày D, so N ngày: sau = [D, D+N−1], trước = [D−N, D−1] (đọc D, N trong câu);
  cộng so_luong_co_ban của mặt hàng trên dòng tinh_dt mỗi khoảng; (sau − trước) / trước × 100 → "percent".
- % THAY ĐỔI so với kỳ trước: tính cùng một cách cho PERIODS[0] và kỳ so sánh, (kỳ hỏi − kỳ so) / kỳ so × 100,
  làm tròn 2 chữ số, type "percent" (giảm thì âm).
- TỈ TRỌNG (chiếm bao nhiêu %): phần / tổng doanh thu CẢ SHOP cùng kỳ (kể cả dòng phí) × 100 → "percent".
- LÃI GỘP (§5): doanh thu − (so_luong × gia_von) trên các dòng tính doanh thu.
- "THÁNG NÀO": type "date", dạng "YYYY-MM". "NGÀY NÀO": "YYYY-MM-DD".
- Trả tên (hàng, khách...) đúng như trong dữ liệu, không bỏ dấu, không xoá khoảng trắng.

# KHI PHẢI DÙNG BẢNG GỐC
- Trạng thái đơn ghi nhiều kiểu: dùng status_of(...) / valid_orders(df, "cột trạng thái", "cột mã đơn").
- Cột của CẢ ĐƠN (tổng đơn, khách trả, giảm giá cả đơn...) lặp lại ở mọi dòng hàng: gộp theo mã đơn trước khi cộng.
- Ngày/trạng thái ở bảng đơn còn chi tiết ở bảng khác: lọc bảng đơn trước, rồi lọc bảng chi tiết bằng isin(mã đơn).
"""

USER_TEMPLATE = """# DỮ LIỆU CỦA SHOP
{schema}

# CÂU HỎI
Hôm nay (as_of_date): {today}
Dữ liệu có đến hết (README): {data_end}
Khoảng thời gian trong câu hỏi (đã tính sẵn, biến PERIODS):
{periods}

Câu hỏi: {question}
"""

# --------------------------------------------------------------------------- chạy code


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.S)


def _extract_code(text: str):
    blocks = _CODE_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    if "ANSWER" in (text or "") or "print(" in (text or ""):
        return text.strip()                   # model quên bọc ```, thử chạy nguyên văn
    return None


def _p_dict(p):
    if p is None:
        return None
    return {"label": p.label, "kind": p.kind,
            "start": pd.Timestamp(p.start) if p.start else None, "end": pd.Timestamp(p.end)}


_BAN_DESC = {
    "ma_don": "mã đơn", "ngay": "ngày bán của đơn", "thoi_diem": "ngày giờ tạo đơn", "trang_thai": "trạng thái chuẩn",
    "tra_hang": "đơn có trả hàng", "tinh_dt": "dòng tính doanh thu (§1)", "dong_tra_hang": "dòng ghi hàng trả (số âm)",
    "ma_hang": "mã hàng", "ten_hang": "tên hàng", "la_phi": "dòng phí, không phải mặt hàng", "nhom_hang": "nhóm hàng",
    "don_vi": "đơn vị của dòng", "so_luong": "số lượng theo đơn vị của dòng", "so_luong_co_ban": "số lượng quy về đơn vị cơ bản",
    "don_gia": "đơn giá", "thanh_tien": "thành tiền dòng sau giảm (doanh thu)", "gia_von": "giá vốn 1 đơn vị bán",
    "ma_khach": "mã khách", "ten_khach": "tên khách", "nhom_khach": "nhóm khách", "kenh": "kênh bán",
    "nhan_vien": "mã nhân viên bán", "ten_nhan_vien": "tên nhân viên",
}


def _ban_schema(df, ban):
    if df is None:
        return ""
    src = ban.attrs.get("nguon", {}) if ban is not None else {}
    lines = [f"## Bảng CHUẨN `ban_hang_chuan`: {len(df):,} dòng, mỗi dòng là {src.get('moi_dong_la', 'một dòng hàng')} "
             f"(dựng từ `{src.get('dong')}`" + (f" + `{src.get('don')}`" if src.get("don") else "") + "). DÙNG BẢNG NÀY TRƯỚC."]
    for c in df.columns:
        s = df[c].dropna()
        if c in ("trang_thai", "kenh", "nhom_khach", "nhom_hang", "don_vi") and s.nunique() <= 12:
            ex = ", ".join(map(str, s.value_counts().index[:12]))
        elif pd.api.types.is_bool_dtype(df[c]):
            ex = f"True ở {int(s.sum()):,} dòng"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            ex = f"{s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}" if c == "ngay" else f"vd {s.iloc[0]}"
        else:
            ex = "; ".join(map(str, s.astype(str).unique()[:3]))
        lines.append(f"- `{c}`: {_BAN_DESC.get(c, '')} | {ex[:120]}")
    return "\n".join(lines) + "\n\n# Các bảng GỐC của shop\n"


def _in_period(series, p):
    s = pd.to_datetime(series, errors="coerce").dt.normalize()
    ok = s <= p["end"]
    if p.get("start") is not None:
        ok &= s >= p["start"]
    return ok


_ST_HUY = re.compile(r"\bhuy\b|cancel|khong giao duoc|giao that bai|\bthat bai")   # \b: "chuyen" có chứa "huy"!
_ST_TRA = re.compile(r"\btra hang|\btra lai|hoan tra|hoan hang|doi tra|return|hang tra ve|khach tra\b|mang tra|\btra ve")
_ST_GIAO = re.compile(r"\bdang\b|cho giao|chua giao|shipping|pending|processing|van chuyen")


def _status_one(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return "hoàn thành"          # §8: sổ không ghi trạng thái / ghi chú không nhắc trạng thái = hoàn thành
    t = fold(v)
    if _ST_HUY.search(t):
        return "huỷ"
    if _ST_TRA.search(t):
        return "trả hàng"
    if _ST_GIAO.search(t):
        return "đang giao"
    return "hoàn thành"


def _valid_orders(df, status_col, id_col=None):
    """True ở các dòng thuộc đơn tính doanh thu: bỏ đơn huỷ và đơn có trả hàng (bỏ cả đơn)."""
    st = _status_of(df[status_col])
    bad = st.isin(["huỷ", "trả hàng"])
    if id_col is not None and id_col in df.columns:
        bad_ids = set(df.loc[bad, id_col].dropna())
        return ~df[id_col].isin(bad_ids) & ~bad
    return ~bad


def _status_of(series):
    """Xếp trạng thái đơn (ghi nhiều kiểu) về 4 nhóm: hoàn thành / huỷ / trả hàng / đang giao."""
    s = pd.Series(series)
    m = {v: _status_one(v) for v in s.dropna().unique()}
    return s.map(lambda v: m.get(v, _status_one(v)) if not (isinstance(v, float) and math.isnan(v)) else "hoàn thành")


def _canon(v, keep_accents=True):
    t = unicodedata.normalize("NFC", str(v)).lower() if keep_accents else fold(v)
    return re.sub(r"[^\w]+", " ", t).strip()


def _match(series, text):
    """Series True/False: ô khớp tên `text` (không phân biệt hoa thường, dấu câu, khoảng trắng).
    Thứ tự ưu tiên: khớp đúng cả tên (giữ dấu) -> khớp đúng (bỏ dấu) -> chứa (giữ dấu) -> chứa (bỏ dấu).
    Giữ dấu trước để "Sting dâu" không lẫn với "Sting dầu"."""
    s = pd.Series(series)
    if not str(text).strip():
        return pd.Series(False, index=s.index)
    raw = s.astype(str)
    for keep in (True, False):
        key, f = _canon(text, keep), raw.map(lambda v: _canon(v, keep))
        exact = f == key
        if exact.any():
            return exact
    for keep in (True, False):
        key, f = _canon(text, keep), raw.map(lambda v: _canon(v, keep))
        hit = f.str.contains(re.escape(key), regex=True)
        if hit.any():
            return hit
    return pd.Series(False, index=s.index)


def _run(code: str, env: dict):
    """Chạy code trong env (dict dùng lại giữa các lượt). -> (kết quả in ra, lỗi hoặc None)."""
    buf = io.StringIO()
    err = None
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(CODE_TIMEOUT)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<code>", "exec"), env)
    except _Timeout:
        err = f"Code chạy quá {CODE_TIMEOUT} giây, hãy viết cách nhanh hơn (tránh vòng lặp từng dòng)."
    except Exception as e:
        err = _explain_error(e, code, env)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    out = buf.getvalue()
    if len(out) > OUT_LIMIT:
        out = out[:OUT_LIMIT] + f"\n... (cắt bớt, tổng {len(out)} ký tự; hãy in ít hơn)"
    return out, err

def _explain_error(e, code, env):
    """Thông báo lỗi ngắn + gợi ý sửa (tên cột/bảng gần đúng) để model nhỏ sửa được."""
    import difflib
    tb = traceback.extract_tb(e.__traceback__)
    line_no = next((fr.lineno for fr in reversed(tb) if fr.filename == "<code>"), None)
    lines = code.splitlines()
    msg = f"{type(e).__name__}: {e}"
    if line_no and 0 < line_no <= len(lines):
        msg = f"Dòng {line_no}: {lines[line_no - 1].strip()}\n{msg}"
    tables = env.get("tables", {})
    hints = []
    if isinstance(e, KeyError):
        key = str(e.args[0]) if e.args else ""
        keys = key.strip("[]").replace("'", "").split(", ") if key.startswith("[") else [key]
        for k in keys:
            if k in tables:
                continue
            tnames = difflib.get_close_matches(k, list(tables), n=3, cutoff=0.6)
            if tnames:
                hints.append(f"Không có bảng '{k}'. Bảng gần giống: {tnames}")
                continue
            cands = []
            for tn, df in tables.items():
                cols = [c for c in df.columns if c != "_nguon"]
                fmap = {fold(c): c for c in cols}
                for fc in difflib.get_close_matches(fold(k), list(fmap), n=3, cutoff=0.55):
                    cands.append(f"tables['{tn}']['{fmap[fc]}']")
            hints.append(f"Không có cột '{k}' trong bảng đang dùng." +
                         (f" Cột gần giống: {', '.join(cands[:6])}" if cands else ""))
            # gợi ý merge: bảng nào có đúng cột k, và có cột chung với bảng đang dùng
            line = lines[line_no - 1] if line_no and 0 < line_no <= len(lines) else code
            cur = [t for t in re.findall(r"tables\[['\"]([^'\"]+)['\"]\]", code) if t in tables]
            for tn, df in tables.items():
                if fold(k) in {fold(c) for c in df.columns} and cur and tn not in cur:
                    common = [c for c in df.columns if c != "_nguon" and c in set(tables[cur[-1]].columns)]
                    if common:
                        hints.append(f"Cột '{k}' có ở tables['{tn}']; merge với tables['{cur[-1]}'] qua cột chung {common[:2]}"
                                     f" (vd: x.merge(tables['{tn}'][{common[:1] + [k]}].drop_duplicates(), on='{common[0]}', how='left')).")
            _ = line
    if isinstance(e, NameError):
        name = str(e).split("'")[1] if "'" in str(e) else ""
        if name in tables:
            hints.append(f"Bảng phải lấy qua dict: tables['{name}'] (không có biến tên {name}).")
    used = [t for t in dict.fromkeys(re.findall(r"tables\[['\"]([^'\"]+)['\"]\]", code)) if t in tables]
    for t in used[:3]:
        cols = [c for c in tables[t].columns if c != "_nguon"]
        hints.append(f"Cột của tables['{t}']: {cols}")
    return msg + ("\nGỢI Ý:\n- " + "\n- ".join(hints) if hints else "")


_STATUS_COL = re.compile(r"^(trang thai|tinh trang|status|tien do|ket qua|xu ly|tt don|ghi chu tt)")


def _self_check(question, code, tables):
    """Bắt lỗi hay gặp nhất của model nhỏ: tính doanh thu/số đơn/bán chạy mà quên bỏ đơn huỷ, đơn trả.
    Trả về câu nhắc (str) hoặc None. Quy tắc chung cho mọi shop, không theo business_id."""
    q = fold(question)
    # (a) tự viết ngày tháng thay vì dùng PERIODS
    if re.search(r"""["']start["']\s*:|pd\.Timestamp\(\s*["']20\d\d-\d\d-01["']\s*\)""", code) and "PERIODS" in code \
            or re.search(r"""["']start["']\s*:\s*pd\.Timestamp""", code):
        return ("Code đang TỰ VIẾT khoảng ngày (dict 'start'/'end' hoặc pd.Timestamp ngày đầu tháng). Hãy dùng đúng "
                "PERIODS[0], PERIODS[1]... đã tính sẵn ở phần CÂU HỎI (kỳ so sánh cũng có sẵn trong PERIODS), không tự tạo.")
    # (b) đếm đơn bằng số dòng
    if "ban_hang_chuan" in code or re.search(r"\bB\[", code):
        if (re.search(r"\bdon\b|bill|hoa don", q) and "nunique" not in code
                and re.search(r"\.shape\[0\]|\blen\(|\.count\(\)|\.size\b", code)):
            return ("Đếm SỐ ĐƠN phải đếm mã đơn khác nhau: df['ma_don'].nunique(), không đếm số dòng "
                    "(len / shape[0] / count), vì một đơn có nhiều dòng hàng.")
    if re.search(r"\bhuy\b|tra hang|tra lai|\bno\b|cong no|nhap|ton kho|ty le|ti le|gia ban|don gia", q):
        return None                    # câu về huỷ/trả/nợ/nhập/tồn/giá bán: luật khác, không nhắc
    if "ban_hang_chuan" in code or re.search(r"\bB\[", code):
        if "tinh_dt" not in code and "trang_thai" not in code and "tra_hang" not in code:
            return ("Code dùng tables['ban_hang_chuan'] nhưng CHƯA lọc b['tinh_dt'] (bỏ đơn huỷ, đơn trả hàng theo §1). "
                    "Doanh thu, số đơn, bán chạy, chi tiêu của khách... chỉ tính dòng có tinh_dt == True: "
                    "thêm  b = b[b['tinh_dt']]  rồi tính lại.")
        return None
    if "valid_orders" in code or "status_of" in code:
        return None
    used = [t for t in dict.fromkeys(re.findall(r"tables\[['\"]([^'\"]+)['\"]\]", code)) if t in tables]
    for t in used:
        df = tables[t]
        for c in df.columns:
            if _STATUS_COL.match(fold(c)) and 2 <= df[c].nunique() <= 15:
                from loader import _ID_NAME, _fname
                ids = [x for x in df.columns if _ID_NAME.match(_fname(x)) and fold(x) != "stt"]
                call = f"valid_orders(df, '{c}', '{ids[0]}')" if ids else f"valid_orders(df, '{c}')"
                return (f"Bảng tables['{t}'] có cột trạng thái '{c}' nhưng code chưa bỏ đơn huỷ / đơn trả hàng (§1). "
                        f"Doanh thu, số đơn, bán chạy... chỉ tính đơn không huỷ, không trả. "
                        f"Ngay sau khi lấy bảng đó (gọi là df), thêm đúng dòng: df = df[{call}]  "
                        f"(KHÔNG so sánh trực tiếp với chữ 'huỷ'/'trả hàng', vì cột ghi nhiều kiểu khác nhau).")
    return None


def _fix_date_type(question, ans):
    """Câu hỏi "tháng nào"/"ngày nào" mà model trả chuỗi kiểu "tháng 7/2025", "07/2025", "2025-07-05" -> type date."""
    q = fold(question)
    if ans.get("type") not in ("string", "number") or not re.search(r"(thang|ngay) nao", q):
        return ans
    v = fold(ans.get("answer"))
    if re.search(r"thang nao", q):
        m = re.search(r"(\d{4})-(\d{1,2})\b", v) or None
        if m:
            return {"type": "date", "answer": f"{int(m[1]):04d}-{int(m[2]):02d}"}
        m = re.search(r"(\d{1,2})\s*[/-]\s*(\d{4})", v)
        if m and 1 <= int(m[1]) <= 12:
            return {"type": "date", "answer": f"{int(m[2]):04d}-{int(m[1]):02d}"}
    else:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", v)
        if m:
            return {"type": "date", "answer": f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"}
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", v)
        if m:
            return {"type": "date", "answer": f"{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}"}
    return ans


def _restore_names(ans, tables):
    """Model đôi khi trả tên đã bị chuẩn hoá ("càphêsữa(m)", "PANADOL EXTRA"). Nếu tên không có nguyên văn trong
    dữ liệu nhưng khớp đúng một cách viết khi bỏ hoa thường / khoảng trắng / dấu câu, thay bằng cách viết trong dữ liệu."""
    def key(v):
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFC", str(v)).lower())
    vals = set()
    for df in tables.values():
        for c in df.columns:
            if c != "_nguon" and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
                u = df[c].dropna().unique()
                if len(u) <= 5000:
                    vals.update(x for x in u if isinstance(x, str))
    exact = vals
    by_key = {}
    for v in vals:
        by_key.setdefault(key(v), set()).add(v)

    def fix(x):
        if not isinstance(x, str) or x in exact:
            return x
        cands = by_key.get(key(x))
        return next(iter(cands)) if cands and len(cands) == 1 else x
    t = ans.get("type")
    if t == "string":
        ans = {**ans, "answer": fix(ans["answer"])}
    elif t == "list":
        ans = {**ans, "answer": [fix(x) for x in ans["answer"]]}
    elif t == "table":
        rows = [[fix(r[0])] + list(r[1:]) if r else r for r in ans["answer"]["rows"]]
        ans = {**ans, "answer": {**ans["answer"], "rows": rows}}
    return ans


# --------------------------------------------------------------------------- kiểm tra đáp án


def _py(v):
    """numpy/pandas -> kiểu JSON thuần."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [_py(x) for x in list(v)]
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    return v


def _num(x, ndigits=None):
    if isinstance(x, bool) or x is None:
        raise ValueError("không phải số")
    if isinstance(x, str):
        x = float(x.replace(",", "").replace("%", "").strip())
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("số không hợp lệ (NaN/inf)")
    if ndigits is not None:
        x = round(x, ndigits)
    return int(round(x)) if abs(x - round(x)) < 1e-9 else x


def normalize_answer(ans):
    """Kiểm tra + chuẩn hoá ANSWER. -> (dict đáp án, None) hoặc (None, thông báo lỗi để gửi lại LLM)."""
    if not isinstance(ans, dict) or "type" not in ans:
        return None, 'ANSWER phải là dict có khoá "type" (và "answer").'
    ans = _py(ans)
    t = str(ans["type"]).strip().lower()
    v = ans.get("answer")
    try:
        if t == "unanswerable":
            return {"type": "unanswerable", "reason": str(ans.get("reason") or "")}, None
        if v is None:
            return None, 'Thiếu "answer". Nếu không trả lời được thì dùng type "unanswerable".'
        if t == "number":
            return {"type": t, "answer": _num(v, 2)}, None
        if t == "percent":
            return {"type": t, "answer": _num(v, 2)}, None
        if t == "boolean":
            if not isinstance(v, bool):
                return None, 'boolean phải là True hoặc False.'
            return {"type": t, "answer": v}, None
        if t == "date":
            v = str(v)[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", str(v)) else str(v)
            if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v):
                return None, 'date phải là "YYYY-MM-DD" (hoặc "YYYY-MM" nếu hỏi theo tháng).'
            return {"type": t, "answer": v}, None
        if t == "string":
            if isinstance(v, (list, dict)):
                return None, "string phải là một chuỗi, không phải list."
            return {"type": t, "answer": str(v).strip()}, None
        if t == "list":
            if not isinstance(v, list):
                return None, "list phải là mảng (Python list), kể cả khi chỉ có 1 phần tử."
            return {"type": t, "answer": [x if isinstance(x, str) else _num(x) for x in v]}, None
        if t == "table":
            if not isinstance(v, dict) or "columns" not in v or "rows" not in v:
                return None, 'table phải là {"columns": [...], "rows": [[...], ...]}.'
            rows = [[c if isinstance(c, str) else _num(c) for c in r] for r in v["rows"]]
            return {"type": t, "answer": {"columns": [str(c) for c in v["columns"]], "rows": rows}}, None
    except (ValueError, TypeError) as e:
        return None, f"Giá trị không đúng kiểu {t}: {e}"
    return None, f'type "{t}" không hợp lệ. Chọn một trong: number, percent, date, boolean, string, list, table, unanswerable.'

# --------------------------------------------------------------------------- solve


def solve(question: dict, data_dir: str) -> dict:
    t0 = time.time()
    rulebook, cal = _static(data_dir)
    biz = _business(data_dir, question["business_id"])
    today = pd.Timestamp(question["as_of_date"])
    data_end = pd.Timestamp(biz.data_end) if biz.data_end else None

    periods = find_periods(question["question"], question["as_of_date"], cal)
    plist, p_lines = [], []
    for p in periods:
        plist.append(_p_dict(p))
        cmp = p.meta.get("compare")
        if cmp is not None:
            d = _p_dict(cmp)
            d["label"] = f"kỳ so sánh của {p.label}: {cmp.label}"
            plist.append(d)
    for i, d in enumerate(plist):
        start = d["start"].strftime("%Y-%m-%d") if d["start"] is not None else "(đầu dữ liệu)"
        p_lines.append(f"- PERIODS[{i}] = {d['label']}: {start} → {d['end']:%Y-%m-%d} (cả hai đầu)")
    p_text = "\n".join(p_lines) or "- (không có mốc thời gian; PERIODS rỗng — đọc câu hỏi và rulebook)"

    ban = getattr(biz, "ban", None)
    if ban is None and not getattr(biz, "_ban_tried", False):
        biz.ban = ban = build_sales(biz)
        biz._ban_tried = True
    tables = {}
    if ban is not None:
        tables["ban_hang_chuan"] = ban[[c for c in ban.columns if ban[c].notna().any()]].copy()
    tables.update({k: v.copy() for k, v in biz.tables.items()})          # bản sao: code của LLM không làm hỏng cache
    env = {"tables": tables, "B": tables.get("ban_hang_chuan"),
           "pd": pd, "np": np, "re": re, "math": math,
           "TODAY": today, "DATA_END": data_end,
           "PERIODS": plist,
           "norm": fold, "in_period": _in_period, "status_of": _status_of, "match": _match,
           "valid_orders": _valid_orders}

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(rulebook=rulebook, max_steps=MAX_STEPS)},
        {"role": "user", "content": USER_TEMPLATE.format(
            schema=_ban_schema(tables.get("ban_hang_chuan"), ban) + biz.schema_text(), today=question["as_of_date"],
            data_end=biz.data_end or "không ghi", periods=p_text, question=question["question"])},
    ]
    thinking_log, codes, outputs = [], [], []
    critiqued = 0
    for step in range(1, MAX_STEPS + 1):
        if time.time() - t0 > TIME_BUDGET:
            outputs.append(f"Hết thời gian {TIME_BUDGET}s cho câu này")
            break
        t_call = time.time()
        text, thinking = llm.chat(messages)
        if VERBOSE:
            print(f"    lượt {step}: model trả lời sau {time.time() - t_call:.0f}s "
                  f"({len(text)} ký tự{', có suy nghĩ' if thinking else ''})", flush=True)
        if thinking:
            thinking_log.append(thinking)
        code = _extract_code(text)
        messages.append({"role": "assistant", "content": text})
        if not code:
            feedback = "Không thấy khối ```python```. Hãy trả lời bằng đúng một khối code Python."
            outputs.append(feedback)
            messages.append({"role": "user", "content": feedback})
            continue
        if codes and code.strip() == codes[-1].strip():
            feedback = ("Code này GIỐNG HỆT lần trước nên sẽ lỗi/ra kết quả y như cũ. "
                        "Đọc kỹ thông báo lỗi và gợi ý ở trên rồi viết code KHÁC.")
            outputs.append(feedback)
            messages.append({"role": "user", "content": feedback})
            continue
        env.pop("ANSWER", None)
        out, err = _run(code, env)
        codes.append(code)
        outputs.append(out + (f"\nLỖI:\n{err}" if err else ""))
        if VERBOSE:
            first = (err.splitlines()[0] if err.splitlines() else "") if err else ""
            state = f"lỗi: {first[:90]}" if err else ("đã gán ANSWER" if "ANSWER" in env else "in kết quả, chưa có ANSWER")
            print(f"           chạy code: {state}", flush=True)
        if err:
            feedback = f"Kết quả in ra:\n```\n{out}\n```\nCode bị LỖI:\n```\n{err}\n```\nHãy sửa và chạy lại."
        elif "ANSWER" in env:
            ans, problem = normalize_answer(env["ANSWER"])
            if ans is not None and critiqued < 2 and step < MAX_STEPS:
                warn = _self_check(question["question"], "\n".join(codes), env["tables"])
                if warn:
                    critiqued += 1
                    messages.append({"role": "user", "content":
                                     f"Kết quả in ra:\n```\n{out}\n```\nANSWER hiện tại: {ans}\n"
                                     f"KIỂM TRA LẠI: {warn}\nNếu cần sửa thì viết code mới và gán lại ANSWER; "
                                     f"nếu đã đúng thì gán lại đúng ANSWER đó."})
                    if VERBOSE:
                        print(f"           tự kiểm: {warn[:90]}", flush=True)
                    continue
            if ans is not None:
                ans = _restore_names(ans, env["tables"])
                ans = _fix_date_type(question["question"], ans)
                if ABSTAIN_ZERO and ((ans["type"] in ("number", "percent") and ans["answer"] == 0)
                                     or (ans["type"] in ("list",) and not ans["answer"])):
                    outputs.append(f"Bỏ qua vì đáp án rỗng/bằng 0: {ans}")
                    ans = {"type": "abstain"}
                return {"answer": ans, "thinking": thinking_log or None, "code": "\n\n# ---\n".join(codes),
                        "output": outputs, "steps": step, "seconds": round(time.time() - t0, 1)}
            feedback = f"Kết quả in ra:\n```\n{out}\n```\nANSWER chưa đúng format: {problem}"
        else:
            left = MAX_STEPS - step
            feedback = (f"Kết quả in ra:\n```\n{out or '(không in gì)'}\n```\n"
                        + (f"Còn {left} lượt. Nếu đã đủ thông tin, hãy tính và gán ANSWER." if left > 1 else
                           "Đây là LƯỢT CUỐI: bắt buộc gán ANSWER (hoặc unanswerable nếu dữ liệu không đủ)."))
        messages.append({"role": "user", "content": feedback})

    return {"answer": {"type": "abstain"}, "thinking": thinking_log or None, "code": "\n\n# ---\n".join(codes),
            "output": outputs, "steps": len(codes), "seconds": round(time.time() - t0, 1)}
