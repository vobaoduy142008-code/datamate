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
import json
import math
import os
import re
import signal
import sys
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
from chuan_hoa import chuan_hoa
from them_dau import them_dau
import cong_thuc
from sales import ban_health
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
  ma_hang, ten_hang, la_phi (True = dòng phí, không phải mặt hàng), nhom_hang, don_vi (đơn vị bán của dòng),
  don_vi_co_ban (đơn vị cơ bản của mặt hàng), so_luong, so_luong_co_ban (số lượng quy về đơn vị cơ bản),
  don_gia, thanh_tien (thành tiền dòng sau giảm = doanh thu), gia_von (giá vốn 1 đơn vị bán, ghi trên dòng bán),
  ma_khach, ten_khach, nhom_khach, kenh, pttt (hình thức thanh toán), nhan_vien (mã), ten_nhan_vien.
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
    b = b[match(b['ten_hang'], 'Sữa đặc Ông Thọ')]      # hoặc match(b['nhom_hang'], 'điện'), match(b['kenh'], 'zalo'), match(b['pttt'], 'chuyển khoản')
  THUỐC KÊ ĐƠN: b[b['ke_don']]; theo HOẠT CHẤT (ghi ĐÚNG như vậy, thuốc phối hợp nhiều hoạt chất là loại riêng):
    b[b['hoat_chat'].map(norm) == norm('Cetirizin')]
- CA SÁNG / CA CHIỀU (giờ tạo đơn, xem rulebook; train: sáng 06:00–13:59, chiều 14:00–21:59):
    h = b['thoi_diem'].dt.hour
    sang = b[(h >= 6) & (h <= 13)];   chieu = b[(h >= 14) & (h <= 21)]
  "Ca chiều tăng/giảm bao nhiêu % SO VỚI CA SÁNG" (cùng một kỳ): (chiều − sáng) / sáng × 100, MỘT số "percent".
- ĐƠN CÓ DOANH THU ≥ X (hoặc từ X trở lên): cộng theo đơn TRƯỚC rồi mới so:
    d = b.groupby('ma_don')['thanh_tien'].sum();   ANSWER = {{"type": "number", "answer": int((d >= X).sum())}}
- KHÁCH (§3): luôn dùng cột `khach` = một khách (đã bỏ khách lẻ, đã gom cách viết tên). Đếm khách, khách mua nhiều,
  tỉ lệ khách... đều theo `khach`, trả về TÊN (ten_khach), không trả số tiền:
    k = b[b['khach'].notna()].groupby('khach').agg(tien=('thanh_tien', 'sum'), so_don=('ma_don', 'nunique'),
                                                   ten=('ten_khach', 'first')).sort_values('tien', ascending=False)
    ANSWER = {{"type": "string", "answer": k['ten'].iloc[0]}}        # top k: k['ten'].head(n).tolist()
  Số khách mua: b['khach'].nunique(); khách mua từ N đơn: (k['so_don'] >= N).sum().
- GIÁ BÁN của X vào ngày D: đơn giá trên dòng bán X các đơn đặt ngày D (KỂ CẢ đơn huỷ/trả), cho MỘT ĐƠN VỊ CƠ BẢN,
  trước chiết khấu:   r = B[match(B['ten_hang'], 'X') & (B['ngay'] == pd.Timestamp('YYYY-MM-DD'))]
    gia = (r['don_gia'] * r['so_luong'] / r['so_luong_co_ban']).round().mode().iloc[0]
- HIỆU ỨNG TĂNG/ĐỔI GIÁ ngày D, so N ngày: sau = [D, D+N−1], trước = [D−N, D−1] (đọc D, N trong câu);
  cộng so_luong_co_ban của mặt hàng trên dòng tinh_dt mỗi khoảng; (sau − trước) / trước × 100 → "percent".
- % THAY ĐỔI so với kỳ trước: tính cùng một cách cho PERIODS[0] và kỳ so sánh, (kỳ hỏi − kỳ so) / kỳ so × 100,
  làm tròn 2 chữ số, type "percent" (giảm thì âm).
- TỈ TRỌNG (chiếm bao nhiêu %): phần / tổng doanh thu CẢ SHOP cùng kỳ (kể cả dòng phí) × 100 → "percent".
- LÃI GỘP (§5): doanh thu − (so_luong × gia_von) trên các dòng tính doanh thu.
- MẶT HÀNG ĐƯỢC MUA Ở ≥ N ĐƠN khác nhau: x = b[~b['la_phi']];  key = 'ma_hang' if 'ma_hang' in x else 'ten_hang'
    c = x.groupby(key)['ma_don'].nunique();   ANSWER = {{"type": "number", "answer": int((c >= N).sum())}}
- DOANH THU THEO KÊNH / NHÓM KHÁCH (bán tại quầy, qua điện thoại, Zalo OA; khách sỉ, thân thiết...): MỘT SỐ tổng,
  b[match(b['kenh'], 'điện thoại')]['thanh_tien'].sum()   /   b[match(b['nhom_khach'], 'sỉ')]['thanh_tien'].sum()
- "X CHIẾM BAO NHIÊU % DOANH THU" (nhóm hàng, thuốc kê đơn, kênh, khách...): phần của X / TỔNG doanh thu CẢ CỬA HÀNG cùng kỳ × 100:
    b[match(b['nhom_hang'], 'hoàn thiện')]['thanh_tien'].sum() / b['thanh_tien'].sum() * 100    (kê đơn: b[b['ke_don'] == True])
- HỎI NHIỀU SỐ trong một câu (doanh thu A và doanh thu B): type "list", mỗi phần tử là MỘT SỐ, đúng thứ tự hỏi: [so_A, so_B].
- MẶT HÀNG Ế / BÁN CHẬM / KHÔNG BÁN ĐƯỢC CÁI NÀO trong kỳ p (đếm mặt hàng đã từng có trên dòng bán, nhưng trong kỳ không bán được):
    x = B[~B['la_phi']];  key = 'ma_hang' if 'ma_hang' in x else 'ten_hang'
    tat_ca = set(x[x['ngay'] <= p['end']][key].dropna());  da_ban = set(x[x['tinh_dt'] & in_period(x['ngay'], p)][key].dropna())
    ANSWER = {{"type": "number", "answer": len(tat_ca - da_ban)}}
- BÁN ĐƯỢC BAO NHIÊU <ĐƠN VỊ> (ly, cái, gói, m3...) mà không nêu tên hàng: chỉ các mặt hàng có ĐƠN VỊ CƠ BẢN là đơn vị đó,
  cộng số lượng đã quy về đơn vị cơ bản:   b[b['don_vi_co_ban'].map(norm) == norm('cái')]['so_luong_co_ban'].sum()
- SIZE M / SIZE L: món có tên KẾT THÚC bằng (M) / (L), không phân biệt hoa thường:
    b[b['ten_hang'].str.strip().str.upper().str.endswith('(M)')]
- "THÁNG NÀO": type "date", dạng "YYYY-MM". "NGÀY NÀO": "YYYY-MM-DD".
- Trả tên (hàng, khách...) đúng như trong dữ liệu, không bỏ dấu, không xoá khoảng trắng.

# BẢNG CHUẨN NHẬP HÀNG `nhap_hang_chuan` (N = tables['nhap_hang_chuan'], nếu có)
- TIỀN NHẬP HÀNG trong kỳ: x = N[N['tinh'] & in_period(N['ngay'], p)]
    có cột tong_phieu: x.drop_duplicates('ma_phieu')['tong_phieu'].sum(); không có: x['thanh_tien'].sum()
- NHÀ CUNG CẤP nhập nhiều tiền nhất: x.groupby('ncc')['thanh_tien'].sum().idxmax() (trả tên ncc)
- GIÁ NHẬP BÌNH QUÂN của X trong tháng (theo đơn vị nhập, bình quân theo số lượng):
    y = x[match(x['ten_hang'], 'X')];   round(y['thanh_tien'].sum() / y['so_luong'].sum())
  (shop không ghi giá nhập → unanswerable)

# CÔNG NỢ (§4): `cong_no_don` (O) và `thu_tien` (T), nếu có
- Chỉ đơn ghi_no == True và huy == False. COD / bán qua sàn không phải nợ (đã loại trong ghi_no).
- THU NỢ / khách trả nợ trong kỳ: T[T['ghi_no'] & ~T['huy'] & T['sau_ngay_mua'] & in_period(T['ngay_tra'], p)]['so_tien'].sum()
- NỢ THÊM trong kỳ: với đơn o ghi nợ đặt trong kỳ: tong_don − tiền trả CÙNG ngày mua (T['ngay_tra'] == T['ngay_don']), cộng lại.
- CÒN NỢ đến hết ngày X (X = p['end']):
    o = O[O['ghi_no'] & ~O['huy'] & (O['ngay'] <= X)].copy()
    da_tra = T[T['ngay_tra'] <= X].groupby('ma_don')['so_tien'].sum()
    o['no'] = o['tong_don'] - o['ma_don'].map(da_tra).fillna(0);   o = o[o['no'] > 0]
  Tổng nợ: o['no'].sum(). Số khách còn nợ: o['ma_khach'].dropna().nunique(). Top khách nợ: groupby('ma_khach') (bỏ khách lẻ).
  Đơn đã đặt quá N ngày: o[(X - o['ngay']).dt.days > N].
- Một khách cụ thể: lọc O / T theo mã đơn của khách đó (match(O['ten_khach'], 'tên')).

# GIAO HÀNG NHIỀU ĐỢT: `giao_hang_chuan` (G), nếu có
- SỐ ĐỢT GIAO (số đợt hàng xe chở / đã giao) trong kỳ p: mỗi đợt của mỗi đơn tính một đợt, kể cả đơn sau đó trả hàng:
    g = G[in_period(G['ngay_giao'], p)];   len(g.drop_duplicates(['ma_don', 'dot']))

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
    "ma_khach": "mã khách", "ten_khach": "tên khách", "khach": "MỘT KHÁCH theo §3 (đã bỏ khách lẻ) – dùng để đếm/xếp hạng", "nhom_khach": "nhóm khách", "kenh": "kênh bán", "don_vi_co_ban": "đơn vị cơ bản của mặt hàng", "pttt": "hình thức thanh toán của đơn (tiền mặt, chuyển khoản, ghi nợ...)",
    "nhan_vien": "mã nhân viên bán", "ten_nhan_vien": "tên nhân viên",
    "ke_don": "True = thuốc kê đơn", "hoat_chat": "hoạt chất của thuốc (so khớp đúng cả chuỗi bằng norm)",
}


def _ban_schema(df, ban):
    if df is None:
        return ""
    src = ban.attrs.get("nguon", {}) if ban is not None else {}
    lines = [f"## Bảng CHUẨN `ban_hang_chuan`: {len(df):,} dòng, mỗi dòng là {src.get('moi_dong_la', 'một dòng hàng')} "
             f"(dựng từ `{src.get('dong')}`" + (f" + `{src.get('don')}`" if src.get("don") else "") + "). DÙNG BẢNG NÀY TRƯỚC."]
    for c in df.columns:
        s = df[c].dropna()
        if c in ("trang_thai", "kenh", "nhom_khach", "nhom_hang", "don_vi", "pttt") and s.nunique() <= 12:
            ex = ", ".join(map(str, s.value_counts().index[:12]))
        elif pd.api.types.is_bool_dtype(df[c]):
            ex = f"True ở {int(s.sum()):,} dòng"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            ex = f"{s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}" if c == "ngay" else f"vd {s.iloc[0]}"
        else:
            ex = "; ".join(map(str, s.astype(str).unique()[:3]))
        lines.append(f"- `{c}`: {_BAN_DESC.get(c, '')} | {ex[:120]}")
    return "\n".join(lines) + "\n"


_STD_DESC = {
    "nhap_hang_chuan": ("mỗi dòng là một dòng hàng trong phiếu nhập", {
        "ma_phieu": "mã phiếu nhập", "ngay": "ngày phiếu", "huy": "phiếu huỷ", "tinh": "True = phiếu được tính (không huỷ)",
        "ncc": "nhà cung cấp (tên)", "ncc_ma": "mã nhà cung cấp", "ma_hang": "mã hàng", "ten_hang": "tên hàng",
        "don_vi": "đơn vị nhập", "so_luong": "số lượng nhập (theo đơn vị nhập)", "don_gia": "giá nhập 1 đơn vị nhập",
        "thanh_tien": "tiền nhập của dòng", "tong_phieu": "tổng tiền CẢ PHIẾU (lặp lại ở mọi dòng của phiếu)"}),
    "cong_no_don": ("mỗi dòng là một ĐƠN bán (để tính công nợ §4)", {
        "ma_don": "mã đơn", "ngay": "ngày đặt", "tong_don": "số khách phải trả cả đơn", "pttt": "hình thức thanh toán",
        "ghi_no": "True = đơn bán nợ (mua chịu)", "cod_san": "đơn COD / bán qua sàn (không phải nợ)", "huy": "đơn huỷ",
        "ma_khach": "mã khách", "ten_khach": "tên khách", "khach": "MỘT KHÁCH theo §3 (đã bỏ khách lẻ) – dùng để đếm/xếp hạng", "kenh": "kênh", "trang_thai": "trạng thái chuẩn",
        "tra_hang": "đơn có trả hàng"}),
    "thu_tien": ("mỗi dòng là MỘT LẦN khách trả tiền cho một đơn", {
        "ma_don": "mã đơn", "ngay_tra": "ngày trả", "so_tien": "số tiền trả lần đó", "ngay_don": "ngày đặt của đơn",
        "sau_ngay_mua": "True = trả SAU ngày mua (trả nợ)", "ghi_no": "đơn đó là đơn bán nợ", "huy": "đơn đó bị huỷ"}),
    "giao_hang_chuan": ("mỗi dòng là một mặt hàng giao trong MỘT ĐỢT giao của một đơn (sổ giao hàng)", {
        "ma_don": "mã đơn bán", "dot": "số thứ tự đợt / lần / lượt giao của đơn đó", "ngay_giao": "ngày giao (chở) của đợt",
        "ten_hang": "tên hàng giao", "so_luong": "số lượng giao", "don_vi": "đơn vị", "xe": "biển số xe",
        "nguoi_giao": "người giao / chở"}),
}


def _std_schema(name, df):
    if df is None:
        return ""
    what, desc = _STD_DESC[name]
    lines = [f"## Bảng CHUẨN `{name}`: {len(df):,} dòng, {what}."]
    for c in df.columns:
        s = df[c].dropna()
        if pd.api.types.is_bool_dtype(df[c]):
            ex = f"True ở {int(s.sum()):,} dòng"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            ex = f"{s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}" if len(s) else ""
        elif c in ("pttt", "don_vi", "ncc") and s.nunique() <= 10:
            ex = ", ".join(map(str, s.value_counts().index[:10]))
        else:
            ex = "; ".join(map(str, s.astype(str).unique()[:3]))
        lines.append(f"- `{c}`: {desc.get(c, '')} | {ex[:110]}")
    return "\n".join(lines) + "\n"


def _patch_str_contains():
    """str.contains('Sữa tươi (L)') mặc định là regex: '(L)' thành nhóm regex, khớp nhầm mọi tên có chữ L.
    Model gần như luôn muốn tìm NGUYÊN VĂN -> chuỗi có ngoặc mà không có ký hiệu regex khác thì tìm nguyên văn."""
    try:
        from pandas.core.strings.accessor import StringMethods
    except Exception:
        return
    orig = StringMethods.contains
    if getattr(orig, "_datamate", False):
        return

    def contains(self, pat, *args, **kw):
        regex = kw.get("regex", args[3] if len(args) > 3 else True)
        if regex and isinstance(pat, str) and re.search(r"[()]", pat) and not re.search(r"[|\\\[\]^$]|\.\*|\.\+", pat):
            if len(args) > 3:
                args = args[:3] + (False,) + args[4:]
            else:
                kw["regex"] = False
        return orig(self, pat, *args, **kw)

    contains._datamate = True
    StringMethods.contains = contains


_patch_str_contains()


def _in_period(series, p):
    """series nằm trong kỳ p. p: một phần tử PERIODS (dict start/end), hoặc một NGÀY (Timestamp / 'YYYY-MM-DD'),
    hoặc cặp (từ, đến) — model hay truyền các dạng này, nhận hết cho đỡ lỗi."""
    s = pd.to_datetime(series, errors="coerce").dt.normalize()
    if isinstance(p, (tuple, list)) and len(p) == 2:
        p = {"start": p[0], "end": p[1]}
    if not isinstance(p, dict):
        if hasattr(p, "start") and hasattr(p, "end"):
            p = {"start": p.start, "end": p.end}
        else:
            d = pd.Timestamp(p).normalize()
            p = {"start": d, "end": d}
    end = pd.Timestamp(p["end"]).normalize()
    ok = s <= end
    if p.get("start") is not None:
        ok &= s >= pd.Timestamp(p["start"]).normalize()
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


_FILLER = {"so", "thung", "dia", "ly", "cai", "hop", "chai", "goi", "lon", "bao", "cay", "vien", "tuyp", "mat", "hang",
           "mon", "san", "pham", "thuoc", "tien", "ban", "loai", "cuon", "bich", "kg", "m3", "xe", "tam", "vi"}


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
    # model hay chép cả chữ thừa ở đầu ("số thùng Chống thấm Kova...", "số đĩa má heo..."): bỏ dần từ đầu, giữ ≥ 2 từ
    words = str(text).split()
    if len(words) >= 3 and fold(words[0]) in _FILLER:      # chỉ bỏ chữ đệm (số, thùng, đĩa, ly, món...), không bỏ tên
        return _match(series, " ".join(words[1:]))
    return pd.Series(False, index=s.index)


def _run(code: str, env: dict):
    """Chạy code trong env (dict dùng lại giữa các lượt). -> (kết quả in ra, lỗi hoặc None)."""
    if re.search(r"\bos\s*\.\s*(_exit|kill|abort|system)\b|\bsignal\s*\.|\bsubprocess\b", code):
        return "", "Không được dùng os._exit / os.kill / os.system / subprocess / signal. Chỉ tính bằng pandas rồi gán ANSWER."
    buf = io.StringIO()
    err = None
    use_alarm = hasattr(signal, "SIGALRM")
    old_trace = None
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(CODE_TIMEOUT)
    else:
        # Windows không có SIGALRM: theo dõi từng dòng CỦA CODE MODEL VIẾT (không theo dõi bên trong pandas),
        # quá giờ thì ném _Timeout -> vòng lặp vô hạn không làm treo cả lần chạy.
        deadline = time.time() + CODE_TIMEOUT

        def _local(frame, event, arg):
            if event == "line" and time.time() > deadline:
                raise _Timeout()
            return _local

        def _global(frame, event, arg):
            return _local if frame.f_code.co_filename == "<code>" else None

        old_trace = sys.gettrace()
        sys.settrace(_global)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<code>", "exec"), env)
    except _Timeout:
        err = f"Code chạy quá {CODE_TIMEOUT} giây, hãy viết cách nhanh hơn (tránh vòng lặp từng dòng)."
    except SystemExit:           # code gọi exit()/sys.exit(): KHÔNG được để dừng cả chương trình
        err = "Không được gọi exit() / quit() / sys.exit(). Chỉ cần gán ANSWER = {...} rồi kết thúc."
    except Exception as e:
        err = _explain_error(e, code, env)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        else:
            sys.settrace(old_trace)
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


_COUNT_Q = re.compile(r"\b(dem|bao nhieu|may|tong so|so luong)\s+(so (luong )?)?(khach|mat hang|don|lo|san pham|nhan vien|ngay|loai|mon|nha cung cap)")
_NOT_COUNT = re.compile(r"liet ke|\bnhung\b|\bnao\b|ke ten|danh sach|\btop\b|chan|\ble\b|\bva\b|moi (khach|nguoi)|tung|lan luot")


_ONE_Q = re.compile(r"bao nhieu|\bmay\b|%|phan tram")
_MANY_Q = re.compile(r"\bva\b|lan luot|\btung\b|\bmoi\b|\bnhung\b|\bnao\b|liet ke|\btop\b|chan|\ble\b|hai so|2 so|ba so|"
                     r"cac (thang|ngay|tuan|quy)|danh sach|\bvs\b.*\bvs\b|,.*,|tra (duoc|dc)")


def _type_check(question, ans):
    """Câu hỏi đếm SỐ LƯỢNG (bao nhiêu khách / mấy mặt hàng / đếm số đơn...) mà model trả cả danh sách."""
    q = fold(question)
    if ans.get("type") in ("list", "table") and _COUNT_Q.search(q) and not _NOT_COUNT.search(q):
        return ("Câu hỏi hỏi SỐ LƯỢNG (đếm), không hỏi danh sách: đáp án phải là MỘT SỐ, "
                "ANSWER = {\"type\": \"number\", \"answer\": len(danh_sach)} (đếm phần tử khác nhau).")
    # hỏi MỘT giá trị (bao nhiêu / mấy / %) mà trả danh sách nhiều số (train: 1918 câu dạng này, không câu nào là list)
    if ans.get("type") in ("list", "table") and _ONE_Q.search(q) and not _MANY_Q.search(q):
        return ("Câu hỏi chỉ hỏi MỘT giá trị: đáp án là MỘT số (type \"number\", hoặc \"percent\" nếu hỏi %), "
                "không trả danh sách. Tính đúng con số được hỏi rồi gán ANSWER.")
    # trả % trong khi câu không hỏi % / tăng giảm / tỉ lệ (train: 2380 câu dạng này, không câu nào đáp án là percent)
    if ans.get("type") == "percent" and not _PCT_Q.search(q):
        return ("Câu hỏi KHÔNG hỏi phần trăm hay tăng/giảm: đáp án là con số được hỏi (tiền, số lượng...), "
                "type \"number\", không tính phần trăm.")
    return None


_PCT_Q = re.compile(r"%|phan tram|tang|giam|so voi|so vs|ty le|ti le|chiem|ty trong|ti trong|thay doi|chenh")


def _self_check(question, code, tables, plist=None, last=None):
    """Bắt lỗi hay gặp nhất của model nhỏ: tính doanh thu/số đơn/bán chạy mà quên bỏ đơn huỷ, đơn trả.
    Trả về câu nhắc (str) hoặc None. Quy tắc chung cho mọi shop, không theo business_id."""
    q = fold(question)
    last = code if last is None else last          # các lỗi "mẫu code" chỉ xét code mới nhất (code cũ đã sửa thì thôi)
    # (a) tự viết ngày tháng thay vì dùng PERIODS
    lits = set(re.findall(r"""pd\.Timestamp\(\s*["'](20\d\d-\d\d-\d\d)["']\s*\)""", last))
    allowed = set()
    for p0 in plist or []:
        for p in (p0, p0.get("compare") if isinstance(p0.get("compare"), dict) else None):
            if not p:
                continue
            for k in ("start", "end"):
                if p.get(k) is not None:
                    d = pd.Timestamp(p[k])
                    allowed |= {d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}
    # chỉ nhắc khi model tự viết khoảng ngày KHÁC với PERIODS (viết lại đúng ngày thì không sao)
    lits |= set(re.findall(r"""["'](20\d\d-\d\d-\d\d)["']""", last))
    if plist and lits and not lits <= allowed:
        return ("Code đang TỰ VIẾT khoảng ngày (dict 'start'/'end' hoặc pd.Timestamp ngày đầu tháng). Hãy dùng đúng "
                "PERIODS[0], PERIODS[1]... đã tính sẵn ở phần CÂU HỎI (kỳ so sánh cũng có sẵn trong PERIODS), không tự tạo.")
    if re.search(r"read_csv|read_excel|read_sql|open\(", code):
        return ("Dữ liệu ĐÃ nạp sẵn trong biến tables (và các bảng chuẩn), không có file nào để đọc. "
                "Bỏ read_csv/read_excel/open, dùng tables['tên bảng'].")
    if "tong_phieu" in code and "drop_duplicates" not in code and re.search(r"tong_phieu'\]\.sum|tong_phieu\.sum", code):
        return ("Cột tong_phieu là tổng CẢ PHIẾU, lặp lại ở mọi dòng của phiếu: phải x.drop_duplicates('ma_phieu')['tong_phieu'].sum().")
    if "ban_hang_chuan" in code and re.search(r"khach", q) and "'khach'" not in code and '"khach"' not in code \
            and re.search(r"groupby\(\s*['\"](ma_khach|ten_khach)", code):
        return ("Đếm / xếp hạng KHÁCH phải dùng cột 'khach' của ban_hang_chuan (đã bỏ khách lẻ theo §3), "
                "không groupby ma_khach / ten_khach trực tiếp.")
    # (d) số đợt giao hàng: phải đếm cặp (mã đơn, đợt) trong bảng chuẩn giao_hang_chuan
    if "giao_hang_chuan" in tables and re.search(r"\bdot\b", q) and re.search(r"giao|\bcho\b|\bxe\b", q) \
            and not ("drop_duplicates" in code and "'dot'" in code.replace('"', "'")):
        return ("Số ĐỢT GIAO = số cặp (mã đơn, đợt) khác nhau có ngày giao trong kỳ (mỗi đợt của mỗi đơn tính một đợt): "
                "g = G[in_period(G['ngay_giao'], PERIODS[0])];  len(g.drop_duplicates(['ma_don', 'dot'])). Không đếm nunique của cột đợt.")
    # (e) công nợ còn lại: phải theo đúng công thức CÒN NỢ (chỉ đơn ghi_no, đơn đặt đến hết ngày X, không in_period)
    if "cong_no_don" in tables and re.search(r"con (cong )?no|con thieu|thieu tien|no qua han|chua tra( het)? no|dang no|chua thanh toan", q) \
            and re.search(r"cong_no_don|\bO\[", last) and ("ghi_no" not in last or re.search(r"in_period\([^)]*\['ngay'\]", last)):
        return ("Tính CÒN NỢ đến hết ngày X phải theo đúng công thức §CÔNG NỢ: X = PERIODS[0]['end'];  "
                "o = O[O['ghi_no'] & ~O['huy'] & (O['ngay'] <= X)].copy();  da_tra = T[T['ngay_tra'] <= X].groupby('ma_don')['so_tien'].sum();  "
                "o['no'] = o['tong_don'] - o['ma_don'].map(da_tra).fillna(0);  o = o[o['no'] > 0]  "
                "(không dùng in_period cho ngày đặt đơn; đơn quá N ngày: o[(X - o['ngay']).dt.days > N]).")
    # (e2) khách trả nợ / thu nợ trong kỳ: phải dùng bảng thu_tien (các lần trả SAU ngày mua)
    if "thu_tien" in tables and re.search(r"tra (bot |them |lai )?no|thu (duoc |hoi )?no|tra bot|tra dan", q) \
            and "sau_ngay_mua" not in code:
        return ("Tiền khách TRẢ NỢ / thu nợ trong kỳ p phải tính từ bảng thu_tien (các lần trả SAU ngày mua của đơn ghi nợ), "
                "không tính từ ban_hang_chuan: T[T['ghi_no'] & ~T['huy'] & T['sau_ngay_mua'] & in_period(T['ngay_tra'], PERIODS[0])]['so_tien'].sum()")
    # (g) lọc ngày bằng so sánh tự viết thay vì in_period(..., PERIODS[i])
    if plist and "in_period" not in last and not re.search(r"cong_no_don|thu_tien|\bO\[|\bT\[", last) \
            and re.search(r"""\[['"]ngay['"]\]\s*(<=|>=|<|>)""", last):
        return ("Lọc thời gian phải dùng in_period(cột ngày, PERIODS[i]) với các kỳ đã tính sẵn ở phần CÂU HỎI "
                "(vd b[in_period(b['ngay'], PERIODS[0])]), không tự so sánh ngày bằng <, <=, ==.")
    # (h) lãi gộp: giá vốn ghi cho MỘT ĐƠN VỊ BÁN -> nhân so_luong, không nhân so_luong_co_ban
    if re.search(r"lai gop|loi nhuan|tien lai|\blai\b", q) and "gia_von" in last \
            and re.search(r"so_luong_co_ban['\"]?\]?\s*\*\s*\w*\[?['\"]?gia_von|gia_von['\"]?\]?\s*\*\s*\w*\[?['\"]?so_luong_co_ban", last):
        return ("Lãi gộp (§5): gia_von là giá vốn của MỘT ĐƠN VỊ BÁN (cùng đơn vị với don_gia), nên giá vốn dòng = "
                "b['so_luong'] * b['gia_von'], KHÔNG nhân so_luong_co_ban. Lãi gộp = b['thanh_tien'].sum() - (b['so_luong'] * b['gia_von']).sum().")
    # (i) câu về MỘT mặt hàng (tăng giá, giá bán...) mà code lọc theo đơn vị thay vì theo tên hàng
    if "don_vi_co_ban" in last and "ten_hang" not in last and "ma_hang" not in last \
            and re.search(r"tang gia|giam gia|doi gia|len gia|gia ban|mat hang|\bmon\b", q):
        return ("Câu hỏi về MỘT mặt hàng cụ thể: phải lọc đúng mặt hàng đó bằng match(b['ten_hang'], 'tên hàng trong câu'), "
                "rồi cộng so_luong_co_ban. Không lọc theo don_vi_co_ban (công thức đó chỉ dành cho câu KHÔNG nêu tên hàng).")
    # (j) lọc thuốc kê đơn khi câu hỏi không hỏi kê đơn
    if re.search(r"""\[['"]ke_don['"]\]""", last) and not re.search(r"ke don|ke toa|theo don|khong can don|otc", q):
        return ("Câu hỏi KHÔNG hỏi thuốc kê đơn: bỏ điều kiện ke_don, chỉ lọc theo đúng điều kiện câu hỏi nêu "
                "(vd hoạt chất: b[b['hoat_chat'].map(norm) == norm('Vitamin C')]).")
    # (k) lấy giá trị nhóm/tên bằng .get('chữ gõ tay') / .loc['chữ'] mà chữ đó KHÔNG có nguyên văn trong dữ liệu
    try:
        b0 = tables.get("ban_hang_chuan")
        vals = set()
        if b0 is not None:
            for c in ("nhom_hang", "ten_hang", "kenh", "nhom_khach", "hoat_chat", "pttt"):
                if c in b0:
                    vals |= set(b0[c].dropna().astype(str))
        for m in re.finditer(r"""(?:\.get|\.loc|\.at)\s*[\(\[]\s*(['"])(.+?)\1""", last):
            if vals and m[2] not in vals and (" " in m[2] or not m[2].isascii()):
                return (f"'{m[2]}' không có nguyên văn trong dữ liệu (sai dấu / hoa thường), nên .get / .loc ra 0. "
                        f"Hãy lọc dòng bằng match: b[match(b['nhom_hang'], '{m[2]}')] (hoặc cột ten_hang / kenh tương ứng).")
    except Exception:
        pass
    # (l) câu về MỘT khách có tên riêng mà code không lọc theo khách
    if re.search(r"\bkh[aá]ch\s+(?!h[aà]ng\b|l[eẻ]\b|n[aà]o\b|nh[oó]m\b|quen\b|th[aâ]n\b|s[iỉ]\b|m[oớ]i\b|c[uũ]\b|v[aã]ng\b|đo[aà]n\b|[oở]\b|đ?d?[uư][oờ]ng\b|t[aạ]i\b|khu\b)"
                 r"[A-ZĐÀ-Ỹ][\wÀ-ỹ]*\s+[A-ZĐÀ-Ỹ]", question) and not re.search(r"khach|ma_don\b.*isin", last):
        return ("Câu hỏi về MỘT khách cụ thể (có tên riêng): phải lọc theo khách đó, "
                "vd o = O[match(O['ten_khach'], 'tên khách')], các lần trả: T[T['ma_don'].isin(o['ma_don'])], "
                "hoặc b[match(b['ten_khach'], 'tên khách')] với bảng bán.")
    # (m) "cộng mọi mặt hàng tính theo <đơn vị>": phải dùng đơn vị cơ bản
    mu = re.search(r"tinh theo (?!(so|thang|ngay|tuan|quy|nam|don|khach|ca|gio|kenh|nhom|gia|tien)\b)\w+|"
                   r"(cong|tat ca|moi|het) (cac )?(mat hang|mon)\b.*(bao nhieu|may) (?!(don|khach|mat|loai|ngay|lan|phan)\b)\w+", q)
    if mu and "don_vi_co_ban" not in last and "ban_hang_chuan" in tables and not re.search(r"gia|tien|doanh thu|dthu", q):
        return ("Câu hỏi cộng MỌI mặt hàng theo một đơn vị: chỉ lấy các mặt hàng có ĐƠN VỊ CƠ BẢN là đơn vị đó, "
                "cộng số lượng đã quy đổi: b[b['don_vi_co_ban'].map(norm) == norm('<đơn vị>')]['so_luong_co_ban'].sum() "
                "(không lọc theo tên hàng).")
    # (p) nhóm KHÁCH (thân thiết, sỉ, vip...) mà code lọc theo nhóm HÀNG
    if re.search(r"nhom khach|khach (hang )?nhom|khach (than thiet|si|vip|quen)", q) and "ban_hang_chuan" in tables \
            and "nhom_khach" not in last:
        return ("Câu hỏi về NHÓM KHÁCH: lọc theo cột nhom_khach (không phải nhom_hang): "
                "b[match(b['nhom_khach'], 'thân thiết')].")
    # (n) ca sáng / ca chiều mà code không lọc theo giờ
    if re.search(r"\bca (sang|chieu|toi)\b", q) and "ban_hang_chuan" in tables \
            and not re.search(r"thoi_diem|\.hour\b|\bgio\b", last):
        return ("Câu hỏi theo CA: phải lọc theo giờ của thoi_diem (rulebook): h = b['thoi_diem'].dt.hour; "
                "ca sáng = b[(h >= 6) & (h <= 13)], ca chiều = b[(h >= 14) & (h <= 21)].")
    # (o) hoạt chất: so khớp ĐÚNG hoạt chất, không contains (thuốc phối hợp nhiều hoạt chất sẽ bị tính nhầm)
    if re.search(r"""\[['"]hoat_chat['"]\]\.str\.contains""", last):
        return ("Hoạt chất phải so ĐÚNG tên (không dùng str.contains vì khớp nhầm thuốc phối hợp): "
                "b[b['hoat_chat'].map(norm) == norm('Ibuprofen')].")
    # (f) lọc tên bằng chuỗi quá ngắn (vd str.contains('M')) khớp nhầm rất nhiều tên
    short = re.search(r"""str\.contains\(\s*r?(['"])(.{1,2})\1""", last)
    if short and short[2].strip():
        return (f"str.contains('{short[2]}') quá ngắn, khớp nhầm cả các tên chỉ tình cờ chứa chữ đó. "
                f"Hãy in ra các tên hàng khác nhau (b['ten_hang'].unique()) để xem cách ghi, rồi lọc bằng đúng phần ghi đó "
                f"(vd size ghi '(M)' ở cuối tên: b['ten_hang'].str.contains('(M)', regex=False)).")
    # (c) câu hỏi theo hoạt chất / thuốc kê đơn mà code không dùng cột tương ứng
    if "ban_hang_chuan" in tables:
        cols = tables["ban_hang_chuan"].columns
        if re.search(r"hoat chat", q) and "hoat_chat" in cols and "hoat_chat" not in code:
            return ("Câu hỏi theo HOẠT CHẤT: phải lọc cột hoat_chat của ban_hang_chuan, ghi đúng như vậy (không lọc theo tên hàng):"
                    " b[b['hoat_chat'].map(norm) == norm('tên hoạt chất')]")
        if re.search(r"ke don", q) and "ke_don" in cols and "ke_don" not in code:
            return "Câu hỏi về thuốc kê đơn: lọc bằng cột ke_don của ban_hang_chuan (b[b['ke_don']] hoặc b[~b['ke_don']])."
    # (b) đếm đơn bằng số dòng
    uses_ban = "ban_hang_chuan" in code or re.search(r"\bB\[", code)
    skip_q = re.search(r"\bhuy\b|tra hang|tra lai|\bno\b|cong no|nhap|ton kho|gia ban|don gia", q)
    if (uses_ban and not skip_q and "tinh_dt" not in code and "trang_thai" not in code and "tra_hang" not in code):
        return ("Code dùng tables['ban_hang_chuan'] nhưng CHƯA lọc b['tinh_dt'] (bỏ đơn huỷ, đơn trả hàng theo §1). "
                "Doanh thu, số đơn, bán chạy, chi tiêu của khách... chỉ tính dòng có tinh_dt == True: "
                "thêm  b = b[b['tinh_dt']]  rồi tính lại.")
    if uses_ban:
        lit = None
        for m in re.finditer(r"""\[['"](ten_hang|nhom_hang|nhom_khach|ten_khach|kenh|hoat_chat|pttt)['"]\]\s*==\s*(['"])(.*?)\2""", last):
            try:
                vals = set(tables["ban_hang_chuan"][m[1]].dropna().astype(str))
            except Exception:
                vals = set()
            if m[3] not in vals:          # chữ gõ tay KHÔNG có nguyên văn trong dữ liệu -> mới nhắc
                lit = m
                break
        if lit:
            return (f"Đang so sánh cột '{lit[1]}' bằng == với chữ tự gõ (dễ sai hoa thường, dấu, khoảng trắng). "
                    f"Hãy dùng match(df['{lit[1]}'], 'tên') thay cho ==, rồi in ra các giá trị khớp để kiểm tra.")
        if (re.search(r"\bdon\b|bill|hoa don", q)
                and re.search(r"\.shape\[0\]|\blen\(|\.count\(\)|\.size\b", last)):
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


VOTE = os.environ.get("AGENT_VOTE", "0") == "1"   # giải 2 lần (2 cách), khác nhau thì bỏ qua câu
VOTE_TEMP = float(os.environ.get("AGENT_VOTE_TEMP", "0.3"))
_VARIANT_NOTE = ("\n\n(LẦN GIẢI ĐỘC LẬP THỨ HAI: tự làm lại từ đầu; trước khi gán ANSWER hãy in ra và kiểm tra lại "
                 "bảng/cột đã dùng, bộ lọc kỳ PERIODS, bộ lọc trạng thái (tinh_dt) và đơn vị của đáp án.)")


# Công thức cho các luật CHỈ có ở một số rulebook (vd rulebook public: khách trung thành, giờ cao điểm...).
# Mỗi dòng chỉ đưa vào prompt khi rulebook đang chạy có đúng định nghĩa đó -> rulebook khác (train) prompt không đổi.
# Hoà nhau: xếp theo giá trị giảm dần rồi theo mã / tên tăng dần (§ thứ tự của rulebook).
_LUAT_THEM = [
    ("khach trung thanh nhat", "KHÁCH TRUNG THÀNH NHẤT (nhiều NGÀY có mua nhất): g = b[b['khach'].notna()].groupby('khach')['ngay'].nunique()"),
    ("khach sop nhat", "KHÁCH SỘP NHẤT (trung bình một đơn cao nhất, chỉ khách ≥ 3 đơn): x = b[b['khach'].notna()]; "
                       "g = x.groupby('khach').agg(tien=('thanh_tien', 'sum'), don=('ma_don', 'nunique')); g = g[g['don'] >= 3]; g = g['tien'] / g['don']"),
    ("khach hay doi y nhat", "KHÁCH HAY ĐỔI Ý NHẤT (nhiều đơn HUỶ nhất, dùng B gồm cả đơn huỷ): x = B[in_period(B['ngay'], p) & (B['trang_thai'] == 'huỷ') & B['khach'].notna()]; "
                             "g = x.groupby('khach')['ma_don'].nunique()"),
    ("mat hang duoc ua chuong nhat", "MẶT HÀNG ĐƯỢC ƯA CHUỘNG NHẤT (có mặt trong nhiều ĐƠN nhất): x = b[~b['la_phi']]; g = x.groupby('ten_hang')['ma_don'].nunique()"),
    ("cham chi nhat", "NHÂN VIÊN / THỢ CHĂM CHỈ NHẤT (nhiều NGÀY làm việc nhất): g = b.groupby('nhan_vien')['ngay'].nunique()  "
                      "(shop ghi tên ở ten_nhan_vien thì groupby cột đó; sổ sửa xe: cột thợ của bảng gốc, kể cả phiếu bảo hành)"),
    ("ngay dat khach nhat", "NGÀY ĐẮT KHÁCH NHẤT (nhiều ĐƠN nhất): g = b.groupby(b['ngay'].dt.normalize())['ma_don'].nunique()  -> type date YYYY-MM-DD"),
    ("gio cao diem", "GIỜ CAO ĐIỂM (khung 1 giờ có nhiều đơn nhất, đáp án là GIỜ BẮT ĐẦU, type number): "
                     "x = b.dropna(subset=['thoi_diem']); g = x.groupby(x['thoi_diem'].dt.hour)['ma_don'].nunique()"),
    ("nha cung cap chinh", "NHÀ CUNG CẤP CHÍNH (nhiều PHIẾU nhập nhất, bỏ phiếu huỷ; khác nhà cung cấp nhập nhiều tiền nhất): "
                           "x = N[N['tinh'] & in_period(N['ngay'], p)]; g = x.groupby('ncc')['ma_phieu'].nunique()"),
    ("trung binh moi ngay thu", "TRUNG BÌNH MỖI NGÀY THỨ X của kỳ: chia cho SỐ NGÀY THỨ X trong kỳ (kể cả ngày không có đơn): "
                                "d = pd.date_range(p['start'], p['end']); so_ngay = (d.weekday == w).sum()  (thứ 2 → w=0 … chủ nhật → w=6); "
                                "tong = b[b['ngay'].dt.weekday == w]['thanh_tien'].sum(); tong / so_ngay"),
]


def _luat_them(rulebook):
    """Các dòng công thức của _LUAT_THEM mà rulebook đang chạy có định nghĩa."""
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", fold(rulebook or "")))
    rows = [f"- {f}" for k, f in _LUAT_THEM if k in t]
    if not rows:
        return ""
    return ("\n# CÔNG THỨC CHO CÁC LUẬT RIÊNG CỦA RULEBOOK NÀY (b = B đã lọc kỳ p và tinh_dt; "
            "xếp hạng: g.sort_values(ascending=False), hoà nhau thì theo mã / tên tăng dần; trả TÊN / giá trị đứng đầu)\n"
            + "\n".join(rows) + "\n")


def _shop_texts(biz):
    """Chữ có dấu của chính shop (tên hàng, nhóm hàng, README) để thêm dấu đúng cho câu hỏi gõ không dấu."""
    out = [getattr(biz, "notes", "") or ""]
    ban = getattr(biz, "ban", None)
    if ban is not None:
        for c in ("ten_hang", "nhom_hang", "hoat_chat", "kenh", "nhom_khach"):
            if c in ban.columns:
                out += [str(v) for v in ban[c].dropna().unique()[:3000]]
    return out


def _q_with_note(q, biz=None):
    """Câu hỏi gốc + (nếu gõ không dấu / có viết tắt / từ địa phương) một dòng diễn giải để model đỡ hiểu sai."""
    try:
        r = them_dau(q, _shop_texts(biz) if biz is not None else ())
    except Exception:
        r = q
    n = chuan_hoa(r)
    return q if n == q else f"{q}\n(Diễn giải câu hỏi – đã thêm dấu / bỏ viết tắt: {n})"


def _ans_key(ans):
    """Dạng so sánh của một đáp án (để biết 2 lần giải có ra cùng kết quả không)."""
    t, v = ans.get("type"), ans.get("answer")
    if t in ("number",):
        try:
            return (t, round(float(v)))
        except (TypeError, ValueError):
            return (t, str(v))
    if t == "percent":
        try:
            return (t, round(float(v), 2))
        except (TypeError, ValueError):
            return (t, str(v))
    if t == "list":
        return (t, tuple(fold(x) if isinstance(x, str) else x for x in (v or [])))
    if t in ("string", "date"):
        return (t, fold(v))
    return (t, json.dumps(v, ensure_ascii=False, sort_keys=True, default=str))


def solve(question: dict, data_dir: str) -> dict:
    """Giải 1 câu. AGENT_VOTE=1: giải thêm lần 2 (cách khác); 2 lần khác đáp án thì abstain."""
    r1 = _solve_once(question, data_dir, 0)
    a1 = r1.get("answer") or {"type": "abstain"}
    if not VOTE or a1.get("type") in ("abstain", "unanswerable"):
        return r1
    r2 = _solve_once(question, data_dir, 1)
    a2 = r2.get("answer") or {"type": "abstain"}
    same = a2.get("type") not in ("abstain",) and _ans_key(a1) == _ans_key(a2)
    out = dict(r1)
    out["votes"] = [a1, a2]
    out["seconds"] = round((r1.get("seconds") or 0) + (r2.get("seconds") or 0), 1)
    out["code"] = (r1.get("code") or "") + "\n\n# ===== LẦN GIẢI THỨ 2 =====\n" + (r2.get("code") or "")
    out["output"] = list(r1.get("output") or []) + ["===== LẦN GIẢI THỨ 2 ====="] + list(r2.get("output") or [])
    if (r1.get("thinking") or r2.get("thinking")):
        out["thinking"] = list(r1.get("thinking") or []) + list(r2.get("thinking") or [])
    if not same:
        out["answer"] = {"type": "abstain"}
        out["output"].append(f"Hai lần giải khác nhau ({a1} / {a2}) -> bỏ qua câu")
    return out


def _solve_once(question: dict, data_dir: str, variant: int = 0) -> dict:
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
    # CỔNG CHẤT LƯỢNG: bảng chuẩn không đáng tin cho shop này (sổ kiểu lạ) -> model đọc bảng gốc, tắt công thức cố định
    try:
        ban_ok, ban_why = ban_health(biz)
    except Exception as e:
        ban_ok, ban_why = False, [f"lỗi kiểm tra bảng chuẩn: {e!r}"]
    if not ban_ok:
        ban = None
    # kỳ hỏi nằm TRƯỚC ngày đầu tiên có dữ liệu bán của shop -> không có dữ liệu để trả lời (train: 14/14 là unanswerable)
    try:
        lo = ban["ngay"].min() if ban is not None and len(ban) else None
        if lo is not None and pd.notna(lo) and periods and all(
                p.end is not None and pd.Timestamp(p.end) < lo for p in periods):
            why = f"Kỳ hỏi kết thúc trước ngày đầu tiên có dữ liệu bán ({lo:%Y-%m-%d})"
            return {"answer": {"type": "unanswerable", "reason": why}, "thinking": None,
                    "code": "agent.py::_solve_once (kiểm tra kỳ hỏi so với ngày đầu dữ liệu)",
                    "output": [why], "steps": 0, "seconds": round(time.time() - t0, 1)}
    except Exception:
        pass
    # loại câu có công thức cố định trong rulebook (đã kiểm trên train) -> giải thẳng, không cần LLM
    try:
        r = None if not ban_ok else cong_thuc.try_solve(chuan_hoa(question["question"]), periods, biz, rulebook)
        if r is not None:
            return {"answer": r[0], "thinking": None, "code": f"cong_thuc.py: {r[1]}",
                    "output": [f"Giải bằng công thức cố định: {r[1]} -> {r[0].get('answer')}"],
                    "steps": 0, "seconds": round(time.time() - t0, 1)}
    except Exception:
        pass
    tables = {}
    if ban is not None:
        tables["ban_hang_chuan"] = ban[[c for c in ban.columns if ban[c].notna().any()]].copy()
    for name, df in (("nhap_hang_chuan", getattr(biz, "nhap", None)), ("cong_no_don", getattr(biz, "no_don", None)),
                     ("thu_tien", getattr(biz, "thu_tien", None)), ("giao_hang_chuan", getattr(biz, "giao", None))):
        if df is not None and len(df):
            tables[name] = df[[c for c in df.columns if df[c].notna().any()]].copy()
    tables.update({k: v.copy() for k, v in biz.tables.items()})          # bản sao: code của LLM không làm hỏng cache
    env = {"tables": tables, "B": tables.get("ban_hang_chuan"),
           # model hay gọi thẳng tên bảng chuẩn như một biến: cho phép luôn
           **{n: tables[n] for n in ("ban_hang_chuan", "nhap_hang_chuan", "cong_no_don", "thu_tien", "giao_hang_chuan") if n in tables},
           "N": tables.get("nhap_hang_chuan"), "O": tables.get("cong_no_don"), "T": tables.get("thu_tien"),
           "G": tables.get("giao_hang_chuan"),
           "pd": pd, "np": np, "re": re, "math": math,
           "TODAY": today, "DATA_END": data_end,
           "PERIODS": plist,
           "norm": fold, "in_period": _in_period, "status_of": _status_of, "match": _match,
           "valid_orders": _valid_orders}

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(rulebook=rulebook, max_steps=MAX_STEPS)},
        {"role": "user", "content": USER_TEMPLATE.format(
            schema=(("" if ban_ok else
                     "## (KHÔNG có bảng chuẩn ban_hang_chuan cho shop này: " + "; ".join(ban_why) + ". "
                     "Hãy đọc README và dùng trực tiếp các bảng GỐC bên dưới; tự lọc trạng thái huỷ/trả theo rulebook.)\n")
                    + _ban_schema(tables.get("ban_hang_chuan"), ban)
                    + (_luat_them(rulebook) if ban_ok else "")
                    + "".join(_std_schema(n, tables.get(n)) for n in ("nhap_hang_chuan", "cong_no_don", "thu_tien", "giao_hang_chuan"))
                    + "\n# Các bảng GỐC của shop\n" + biz.schema_text()), today=question["as_of_date"],
            data_end=biz.data_end or "không ghi", periods=p_text, question=_q_with_note(question["question"], biz))
            + (_VARIANT_NOTE if variant else "")},
    ]
    thinking_log, codes, outputs = [], [], []
    critiqued = 0
    pending = None          # đáp án đã có trước lần "tự kiểm" (giữ lại nếu model không đưa được đáp án mới)

    def finalize(ans, step):
        ans = _restore_names(ans, env["tables"])
        ans = _fix_date_type(question["question"], ans)
        # model ghi "Không có dữ liệu / không tìm thấy thông tin" thành CHUỖI: đúng ý là "không trả lời được"
        if ans.get("type") == "string" and re.match(
                r"\s*(khong|chua) (co|tim thay|du|ghi)( du)? (du lieu|thong tin)", fold(str(ans.get("answer", "")))):
            ans = {"type": "unanswerable", "reason": str(ans.get("answer"))}
        if ABSTAIN_ZERO and ((ans["type"] in ("number", "percent") and ans["answer"] == 0)
                             or (ans["type"] in ("list",) and not ans["answer"])):
            outputs.append(f"Bỏ qua vì đáp án rỗng/bằng 0: {ans}")
            ans = {"type": "abstain"}
        return {"answer": ans, "thinking": thinking_log or None, "code": "\n\n# ---\n".join(codes),
                "output": outputs, "steps": step, "seconds": round(time.time() - t0, 1)}
    hot = False
    for step in range(1, MAX_STEPS + 1):
        if time.time() - t0 > TIME_BUDGET:
            outputs.append(f"Hết thời gian {TIME_BUDGET}s cho câu này")
            break
        t_call = time.time()
        old_timeout = llm.TIMEOUT          # một lần gọi model không được vượt thời gian còn lại của câu
        llm.TIMEOUT = max(20, min(old_timeout, TIME_BUDGET - (time.time() - t0) + 10))
        try:
            if hot:                        # lần trước model lặp y hệt code cũ: lấy mẫu "nóng" hơn (seed cố định) để thoát vòng lặp
                text, thinking = llm.chat(messages, temperature=0.7, seed=100 + step)
            else:
                text, thinking = (llm.chat(messages, temperature=VOTE_TEMP, seed=1) if variant else llm.chat(messages))
        finally:
            llm.TIMEOUT = old_timeout
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
            if pending is not None:          # model giữ nguyên code sau lần tự kiểm: dùng luôn đáp án đó
                outputs.append("Model giữ nguyên code sau khi tự kiểm -> dùng đáp án trước đó")
                return finalize(pending, step)
            # không đưa câu trả lời lặp vào hội thoại (model sẽ càng lặp); hỏi lại với nhiệt độ cao hơn
            messages.pop()
            outputs.append("Model lặp y hệt code trước -> hỏi lại (lấy mẫu khác)")
            hot = True
            continue
        hot = False
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
                warn = (_type_check(chuan_hoa(question["question"]), ans)
                        or _self_check(chuan_hoa(question["question"]), "\n".join(codes), env["tables"], plist, last=codes[-1]))
                if warn:
                    critiqued += 1
                    # nhắc "yếu" (có thể nhắc nhầm): nếu model giữ nguyên thì vẫn dùng đáp án cũ.
                    # nhắc "chắc" (thiếu lọc tinh_dt, sai khoảng ngày, thiếu cột hoạt chất...): giữ nguyên thì bỏ qua câu.
                    pending = ans if warn.startswith(("Đang so sánh cột", "Đếm SỐ ĐƠN", "str.contains(", "Lọc thời gian phải dùng", "Code đang TỰ VIẾT",
                                       "Hoạt chất phải so")) else None
                    messages.append({"role": "user", "content":
                                     f"Kết quả in ra:\n```\n{out}\n```\nANSWER hiện tại: {ans}\n"
                                     f"KIỂM TRA LẠI: {warn}\nNếu cần sửa thì viết code mới và gán lại ANSWER; "
                                     f"nếu đã đúng thì gán lại đúng ANSWER đó."})
                    if VERBOSE:
                        print(f"           tự kiểm: {warn[:90]}", flush=True)
                    continue
            if ans is not None:
                return finalize(ans, step)
            feedback = f"Kết quả in ra:\n```\n{out}\n```\nANSWER chưa đúng format: {problem}"
        else:
            left = MAX_STEPS - step
            feedback = (f"Kết quả in ra:\n```\n{out or '(không in gì)'}\n```\n"
                        + (f"Còn {left} lượt. Nếu đã đủ thông tin, hãy tính và gán ANSWER." if left > 1 else
                           "Đây là LƯỢT CUỐI: bắt buộc gán ANSWER (hoặc unanswerable nếu dữ liệu không đủ)."))
        messages.append({"role": "user", "content": feedback})

    if pending is not None:                  # hết lượt / hết giờ sau lần tự kiểm: dùng đáp án đã có
        outputs.append("Hết lượt sau khi tự kiểm -> dùng đáp án trước đó")
        return finalize(pending, len(codes))
    return {"answer": {"type": "abstain"}, "thinking": thinking_log or None, "code": "\n\n# ---\n".join(codes),
            "output": outputs, "steps": len(codes), "seconds": round(time.time() - t0, 1)}
