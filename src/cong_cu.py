"""Chế độ ĐIỀU PHỐI (AGENT_MODE=dieu_phoi): model ngôn ngữ là bộ não, các module công thức là công cụ chỉ để tính.

Luồng cho mỗi câu (tối đa 2 lần gọi model):
    Bước 1  model đọc câu hỏi -> chọn <= 2 công cụ + viết lại câu hỏi chuẩn (JSON).
    Bước 2  code chạy các module của công cụ đã chọn (đúng thứ tự ưu tiên cũ) trên câu gốc, rồi trên câu chuẩn;
            không ra -> chạy các module còn lại (dự phòng, theo thứ tự cũ).
    Bước 3  model kiểm duyệt: đáp án + cách tính có đúng điều câu hỏi hỏi không; rõ ràng sai -> bỏ qua (abstain).
LLM_BACKEND=mock: bước 1 chọn tất cả công cụ, câu chuẩn = câu gốc; bước 3 luôn khớp -> kết quả y hệt chế độ cũ.

Không sửa các module ct_*.py / cong_thuc.py; file này chỉ gọi solve() của chúng. Không logic theo id / business_id / tên shop.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import pandas as pd

import llm
from chuan_hoa import chuan_hoa
from loader import fold
from periods import find_periods
import cong_thuc
import ct_coban
import ct_tyle
import ct_top
import ct_morong
import ct_luatrieng
import ct_noton
import ct_phuthem
import ct_suaxe
import ct_mevabe
import ct_dienthoai

# --------------------------------------------------------------------------- cấu hình
# all = kiểm duyệt mọi đáp án (mặc định) | fallback = chỉ kiểm duyệt đáp án của đường dự phòng | off = không kiểm duyệt
KIEM = os.environ.get("DIEUPHOI_KIEM", "all").lower()
# 1 = nếu dữ liệu shop nhận diện được là một ngành riêng (sửa xe / mẹ và bé / điện thoại) thì luôn chạy module ngành đó
# (cũ: chạy trước mọi công thức khác). Đặc điểm lấy từ dữ liệu, không từ id / tên shop.
TU_THEM_NGANH = os.environ.get("DIEUPHOI_TU_THEM_NGANH", "1") == "1"
SO_VI_DU = int(os.environ.get("DIEUPHOI_SO_VI_DU", "6"))
FILE_VI_DU = os.environ.get("DIEUPHOI_VI_DU", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vi_du_cong_cu.jsonl"))

# thứ tự ưu tiên của các module — GIỐNG HỆT agent._solve_once (chế độ cũ)
THU_TU = ["suaxe", "mevabe", "dienthoai", "cong_thuc", "coban", "tyle", "top", "morong", "luatrieng", "noton", "phuthem"]
NGANH = {"suaxe": ct_suaxe, "mevabe": ct_mevabe, "dienthoai": ct_dienthoai}
CT_GOC = {"coban": ct_coban, "tyle": ct_tyle, "top": ct_top, "morong": ct_morong}
CT_THEM = {"luatrieng": ct_luatrieng, "noton": ct_noton, "phuthem": ct_phuthem}
TEN_NGANH_HIEN = {"suaxe": "sửa xe", "mevabe": "mẹ và bé", "dienthoai": "cửa hàng điện thoại"}

# --------------------------------------------------------------------------- danh mục công cụ
# stages = các module (xem THU_TU) công cụ này được phép chạy. Thứ tự chạy luôn theo THU_TU.
CONG_CU = {
    "doanh_thu_don_hang": {
        "mo_ta": "Một con số cơ bản của MỘT kỳ: tổng doanh thu, số đơn, số lượng bán, giá bán, giá trị trung bình mỗi đơn; "
                 "của cả shop hoặc của một mặt hàng / một ngày cụ thể.",
        "vi_du": ["Tháng 4/2026 bán Bò lá lốt nướng được bao nhiêu tiền?", "Hôm nay shop có bao nhiêu đơn?"],
        "stages": ["cong_thuc", "coban", "phuthem"]},
    "doanh_thu_dieu_kien": {
        "mo_ta": "Doanh thu / số đơn / danh sách CÓ ĐIỀU KIỆN PHỤ: nhóm hàng, hoạt chất, thuốc kê đơn, kênh bán, nhóm khách, "
                 "nhân viên, ca làm, hình thức thanh toán, size, lãi gộp, giá vốn, đơn huỷ, ngày hoặc tháng doanh thu cao nhất.",
        "vi_du": ["Tháng trước doanh thu thanh toán chuyển khoản là bao nhiêu?", "Ca chiều hôm qua nhân viên Lan bán được mấy đơn?"],
        "stages": ["cong_thuc", "coban", "tyle", "morong", "phuthem"]},
    "ti_le_so_sanh": {
        "mo_ta": "Tăng / giảm bao nhiêu phần trăm so với kỳ trước, so sánh hai kỳ, tỉ lệ, tỉ trọng (một nhóm chiếm bao nhiêu % tổng), % đơn huỷ.",
        "vi_du": ["Doanh thu tháng này tăng hay giảm bao nhiêu phần trăm so với tháng trước?", "Nhóm đồ uống chiếm bao nhiêu % doanh thu tuần qua?"],
        "stages": ["tyle", "phuthem", "cong_thuc"]},
    "xep_hang": {
        "mo_ta": "Top / xếp hạng / nhiều nhất / ít nhất: mặt hàng bán chạy nhất, nhân viên bán giỏi nhất, ngày hoặc giờ cao điểm, "
                 "nhà cung cấp chính, top N mặt hàng.",
        "vi_du": ["Top 3 mặt hàng bán chạy nhất tháng này là gì?", "Giờ nào trong ngày shop đông khách nhất tuần qua?"],
        "stages": ["cong_thuc", "top", "coban", "luatrieng", "morong", "phuthem"]},
    "khach_hang": {
        "mo_ta": "Câu về khách hàng: khách mua nhiều tiền nhất, khách trung thành, khách sộp, khách hay đổi ý, số khách, khách mới, khách mua từ k đơn.",
        "vi_du": ["Khách nào mua nhiều tiền nhất quý này?", "Tháng này có bao nhiêu khách mua từ 3 đơn trở lên?"],
        "stages": ["cong_thuc", "top", "coban", "tyle", "luatrieng", "morong", "noton", "phuthem"]},
    "cong_no": {
        "mo_ta": "Công nợ: còn nợ bao nhiêu, ai còn nợ, thu nợ, ghi nợ thêm, nợ quá hạn / tuổi nợ, tiền phải trả, trả góp.",
        "vi_du": ["Hiện tại khách Hoa còn nợ shop bao nhiêu?", "Tháng này thu hồi được bao nhiêu tiền nợ?"],
        "stages": ["cong_thuc", "noton", "dienthoai"]},
    "ton_kho_lo": {
        "mo_ta": "Tồn kho, giá trị hàng tồn, lô hàng, hạn dùng: còn tồn bao nhiêu, lô nào sắp hết hạn, mặt hàng hết hạn.",
        "vi_du": ["Hiện còn tồn bao nhiêu hộp Enervon?", "Lô nào sắp hết hạn dùng nhất?"],
        "stages": ["noton"]},
    "nhap_hang": {
        "mo_ta": "Nhập hàng: tổng tiền các phiếu nhập, số phiếu nhập, nhà cung cấp nhập nhiều nhất / nhà cung cấp chính.",
        "vi_du": ["Tháng này tổng tiền nhập hàng là bao nhiêu?", "Nhà cung cấp nào shop nhập nhiều tiền nhất quý trước?"],
        "stages": ["cong_thuc", "tyle", "morong", "luatrieng", "noton"]},
    "giao_hang": {
        "mo_ta": "Giao hàng: số đợt giao, đơn giao mấy đợt, hàng đã giao hay chưa giao.",
        "vi_du": ["Tuần này giao hàng tổng cộng mấy đợt?", "Có bao nhiêu đợt giao hàng trong tháng 3?"],
        "stages": ["cong_thuc"]},
    "nganh_sua_xe": {
        "mo_ta": "Shop SỬA XE (có biển số xe, thợ, việc làm): lượt xe, xe quay lại, bảo hành, thay nhớt, thợ làm nhiều nhất, tiền công.",
        "vi_du": ["Tháng này thợ nào sửa nhiều lượt xe nhất?", "Có bao nhiêu phiếu bảo hành trong quý này?"],
        "stages": ["suaxe"]},
    "nganh_dien_thoai": {
        "mo_ta": "Shop ĐIỆN THOẠI (có IMEI, bảo hành máy, thu máy cũ, trả góp): máy bán ra, công nợ trả góp, bảo hành máy.",
        "vi_du": ["Tháng này bán được bao nhiêu máy?", "Còn bao nhiêu khoản trả góp chưa thu xong?"],
        "stages": ["dienthoai"]},
    "nganh_me_va_be": {
        "mo_ta": "Shop MẸ VÀ BÉ (sữa, bỉm, size / số, voucher, phí ship, điểm): doanh thu và đơn kiểu shop này, bỉm theo size, khách mới đăng ký.",
        "vi_du": ["Tháng này bỉm size M chiếm bao nhiêu phần trăm?", "Tháng trước có bao nhiêu khách đăng ký mới?"],
        "stages": ["mevabe"]},
    "khong_du_du_lieu": {
        "mo_ta": "Câu hỏi về thông tin mà dữ liệu shop KHÔNG ghi (giới tính / địa chỉ khách, đánh giá nhân viên, liều dùng thuốc, "
                 "kỳ hỏi trước khi shop có dữ liệu). Công cụ này không tính gì; chỉ chọn khi chắc chắn dữ liệu không có.",
        "vi_du": ["Khách nữ nào mua nhiều nhất tháng này?", "Nhân viên nào tận tình nhất?"],
        "stages": []},
    "khac": {
        "mo_ta": "Không thuộc nhóm nào ở trên hoặc không chắc chắn.",
        "vi_du": ["Cho mình xem tổng quan tình hình shop", "Nên nhập thêm hàng gì?"],
        "stages": []},
}
TEN_CONG_CU = list(CONG_CU)
# công cụ nào được chọn ở chế độ mock (tất cả)
TAT_CA = TEN_CONG_CU[:]


def stage_cua(ten_cc):
    return {s for s in CONG_CU[ten_cc]["stages"]}


def cong_cu_cua_stage(stage):
    return [t for t in CONG_CU if stage in CONG_CU[t]["stages"]]


# --------------------------------------------------------------------------- nhãn công cụ cho ví dụ (dùng ở script tạo ví dụ)
_NHAN_STAGE = {"suaxe": "nganh_sua_xe", "mevabe": "nganh_me_va_be", "dienthoai": "nganh_dien_thoai"}


def nhan_cong_cu(stage, q):
    """Công cụ 'đúng' của 1 câu train đã được module `stage` trả lời đúng. Module ngành ứng thẳng 1 công cụ;
    các module dùng chung thì phân loại theo chữ trong câu hỏi, rồi kiểm tra công cụ đó có chứa module (nếu không -> công cụ đầu chứa module)."""
    if stage in _NHAN_STAGE:
        return _NHAN_STAGE[stage]
    f = " " + fold(chuan_hoa(q)) + " "
    if re.search(r"\blo\b|han dung|\bhet han|sap het han|ton kho|con ton|\bton\b", f):
        t = "ton_kho_lo"
    elif re.search(r"\bno\b|cong no|tra gop|phai tra|thu hoi no|con thieu", f):
        t = "cong_no"
    elif re.search(r"nhap hang|phieu nhap|nha cung cap|\bncc\b|tien nhap|\bnhap\b", f):
        t = "nhap_hang"
    elif re.search(r"\bdot\b.*giao|giao hang|dot giao|giao\b", f):
        t = "giao_hang"
    elif re.search(r"tang|giam|ti le|ty le|ti trong|ty trong|phan tram|%|so voi|so sanh|gap may lan", f):
        t = "ti_le_so_sanh"
    elif re.search(r"\bkhach\b", f):
        t = "khach_hang"
    elif re.search(r"\btop\b|nhieu nhat|it nhat|cao nhat|thap nhat|ban chay|xep hang|dung dau|lon nhat|nho nhat|gio nao|ngay nao|nhat\b", f):
        t = "xep_hang"
    elif re.search(r"nhom hang|hoat chat|ke don|\bkenh\b|nhan vien|\bca (sang|chieu|toi)|thanh toan|chuyen khoan|tien mat|\bsize\b|"
                   r"lai gop|loi nhuan|gia von|\bhuy\b|\bm\b|\bl\b", f):
        t = "doanh_thu_dieu_kien"
    else:
        t = "doanh_thu_don_hang"
    if stage not in CONG_CU[t]["stages"]:
        t = cong_cu_cua_stage(stage)[0]
    return t


# --------------------------------------------------------------------------- ví dụ gần nhất
_VD = None
_TU_BO = re.compile(r"^\d")


def _tu(q):
    ws = [w for w in fold(chuan_hoa(q)).split() if len(w) > 1 and not _TU_BO.match(w)]
    return set(ws) | {a + "_" + b for a, b in zip(ws, ws[1:])}


def _nap_vi_du():
    global _VD
    if _VD is not None:
        return _VD
    rows = []
    try:
        with open(FILE_VI_DU, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("q") and d.get("cong_cu") in CONG_CU:
                    rows.append((d["q"], d["cong_cu"], _tu(d["q"])))
    except OSError:
        pass
    df = {}
    for _, _, ts in rows:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(rows))
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    _VD = (rows, idf)
    return _VD


def vi_du_gan_nhat(q, k=SO_VI_DU):
    """k ví dụ train có nhiều từ chung (theo trọng số idf) nhất với câu hỏi; mỗi công cụ tối đa ceil(k/2) ví dụ để đa dạng."""
    rows, idf = _nap_vi_du()
    if not rows or k <= 0:
        return []
    qs = _tu(q)
    sc = []
    for i, (qq, cc, ts) in enumerate(rows):
        common = qs & ts
        if not common:
            continue
        s = sum(idf.get(t, 1.0) for t in common) / math.sqrt(sum(idf.get(t, 1.0) for t in ts) + 1.0)
        sc.append((-s, i))
    sc.sort()
    out, dem, seen = [], {}, set()
    for _, i in sc:
        qq, cc, _ = rows[i]
        if qq in seen or qq.strip() == q.strip() or dem.get(cc, 0) >= max(1, (k + 1) // 2):
            continue
        seen.add(qq)
        dem[cc] = dem.get(cc, 0) + 1
        out.append((qq, cc))
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------- bối cảnh một câu
class BoiCanh:
    """Mọi thứ agent đã tính sẵn cho câu hỏi + dữ liệu shop."""

    def __init__(self, biz, ban, ban_ok, rulebook, cal, as_of_date):
        self.biz, self.ban, self.ban_ok = biz, ban, ban_ok
        self.rulebook, self.cal, self.as_of = rulebook, cal, as_of_date
        self.today = pd.Timestamp(as_of_date)
        try:
            self.so_tay = any("lấy từ cột ghi chú" in str(w) for w in (getattr(biz, "warnings", None) or [])) or \
                (ban is not None and "ma_don" in ban and ban["ma_don"].astype(str).str.contains("#").any())
        except Exception:
            self.so_tay = False
        self._cache = {}
        self.nganh = nganh_cua_shop(biz)

    def tinh(self, q):
        k = self._cache.get(q)
        if k is None:
            periods = find_periods(q, self.as_of, self.cal)
            try:
                xong = all(pd.Timestamp(p.end) <= self.today for p in periods)
            except Exception:
                xong = False
            k = {"q": q, "periods": periods, "f": fold(chuan_hoa(q)), "xong": xong}
            self._cache[q] = k
        return k


def nganh_cua_shop(biz):
    """Các ngành riêng mà DỮ LIỆU shop khớp (dùng chính hàm nhận diện của module ngành, đã cache trên biz). Không dùng id / tên shop."""
    out = []
    for ten, fn in (("suaxe", lambda: ct_suaxe._prep(biz)), ("mevabe", lambda: ct_mevabe._ctx(biz)),
                    ("dienthoai", lambda: ct_dienthoai._get(biz))):
        try:
            if fn() is not None:
                out.append(ten)
        except Exception:
            pass
    return out


def chay_stage(stage, q, bc):
    """Chạy 1 module trên câu q (đúng điều kiện bật như agent cũ). -> (answer_dict, mo_ta) hoặc None."""
    k = bc.tinh(q)
    periods, f0 = k["periods"], k["f"]
    try:
        if stage in NGANH:
            if not k["xong"]:
                return None
            r = NGANH[stage].solve(q, f0, periods, bc.biz, bc.rulebook)
            if r is not None and r[0].get("type") not in (None, "error"):
                return r[0], f"{NGANH[stage].__name__}: {r[1]}"
            return None
        if stage == "cong_thuc":
            if not bc.ban_ok:
                return None
            r = cong_thuc.try_solve(chuan_hoa(q), periods, bc.biz, bc.rulebook)
            return (r[0], f"cong_thuc: {r[1]}") if r is not None else None
        if stage in CT_GOC:
            if not (bc.ban_ok and k["xong"]):
                return None
            r = CT_GOC[stage].solve(q, f0, periods, bc.biz, bc.rulebook)
            if r is not None and r[0].get("type") not in (None, "error"):
                return r[0], f"{CT_GOC[stage].__name__}: {r[1]}"
            return None
        if stage in CT_THEM:
            if not k["xong"]:
                return None
            if stage != "noton" and not bc.ban_ok:
                return None
            if stage == "phuthem" and bc.so_tay:
                return None
            ct_phuthem.CTX = (bc.as_of, bc.cal)
            r = CT_THEM[stage].solve(q, f0, periods, bc.biz, bc.rulebook)
            if r is not None and r[0].get("type") not in (None, "error"):
                return r[0], f"{CT_THEM[stage].__name__}: {r[1]}"
            return None
    except Exception:
        return None
    return None


def chay_cac(stages, q, bc):
    """Chạy lần lượt các module (đã sắp theo THU_TU) -> (stage, (answer, mo_ta)) đầu tiên ra kết quả, hoặc (None, None)."""
    for s in stages:
        r = chay_stage(s, q, bc)
        if r is not None:
            return s, r
    return None, None


# --------------------------------------------------------------------------- JSON chịu lỗi
def doc_json(text):
    """Lấy khối {...} cân ngoặc đầu tiên trong text rồi json.loads; lỗi -> None."""
    if not text:
        return None
    s = text.find("{")
    while s != -1:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[s:i + 1])
                        if isinstance(d, dict):
                            return d
                    except Exception:
                        pass
                    break
        s = text.find("{", s + 1)
    return None


# --------------------------------------------------------------------------- bước 1: chọn công cụ + chuẩn hoá câu
SYS1 = ("Bạn là bộ điều phối của trợ lý phân tích dữ liệu bán hàng cho cửa hàng nhỏ ở Việt Nam. "
        "Nhiệm vụ: đọc câu hỏi của chủ shop, chọn công cụ phù hợp nhất và viết lại câu hỏi cho chuẩn. Chỉ trả về JSON, không giải thích.")


def _danh_muc():
    dong = []
    for ten, c in CONG_CU.items():
        vd = " | ".join(f"\"{v}\"" for v in c["vi_du"])
        dong.append(f"- {ten}: {c['mo_ta']} Ví dụ: {vd}")
    return "\n".join(dong)


def prompt_buoc1(q, bc):
    vd = vi_du_gan_nhat(q)
    khoi_vd = ("\n# VÍ DỤ ĐÃ BIẾT (câu hỏi -> công cụ đúng)\n" + "\n".join(f"- \"{a}\" -> {b}" for a, b in vd) + "\n") if vd else ""
    nganh = ""
    if bc.nganh:
        nganh = "Dữ liệu shop này thuộc kiểu: " + ", ".join(TEN_NGANH_HIEN[n] for n in bc.nganh) + ".\n"
    return (f"# CÔNG CỤ\n{_danh_muc()}\n{khoi_vd}\n# CÂU CẦN XỬ LÝ\nHôm nay: {bc.as_of}.\n{nganh}Câu hỏi: \"{q}\"\n\n"
            "Việc cần làm:\n"
            "1. Chọn tối đa 2 công cụ, công cụ chắc nhất đứng trước.\n"
            "2. Viết lại câu hỏi bằng tiếng Việt chuẩn có dấu: sửa lỗi gõ, bỏ viết tắt. GIỮ NGUYÊN mọi số, ngày, tên hàng, tên khách; "
            "KHÔNG thêm, bớt hay suy diễn ý. Câu đã chuẩn thì chép lại y nguyên.\n"
            "Trả về đúng một JSON: {\"cong_cu\": [\"ten_cong_cu\"], \"cau_hoi_chuan\": \"...\"}")


def _so_trong(q):
    return re.findall(r"\d+", q or "")


def cau_chuan_hop_le(goc, chuan):
    """Chấp nhận câu chuẩn chỉ khi không rỗng, độ dài hợp lý và còn đủ mọi số của câu gốc."""
    if not isinstance(chuan, str):
        return None
    chuan = " ".join(chuan.split())
    if not chuan or len(chuan) > 3 * len(goc) + 60:
        return None
    thieu = [s for s in _so_trong(goc) if s not in _so_trong(chuan)]
    return None if thieu else chuan


def buoc1(q, bc):
    """-> (danh sách tên công cụ, câu chuẩn hoặc None, ghi chú log)"""
    if llm.BACKEND == "mock":
        return TAT_CA[:], None, "mock: chọn tất cả công cụ, câu chuẩn = câu gốc"
    try:
        text, _ = llm.chat([{"role": "system", "content": SYS1}, {"role": "user", "content": prompt_buoc1(q, bc)}],
                           temperature=0, think=False)
    except Exception as e:
        # model không gọi được (Ollama lỗi / hết giờ): an toàn = mọi công cụ là "chính" -> giống hệt bản chỉ công thức (v4.1)
        return TAT_CA[:], None, f"lỗi gọi model bước 1: {e!r} -> dùng mọi công cụ"
    d = doc_json(text)
    if not d:
        return TAT_CA[:], None, f"bước 1 không đọc được JSON: {str(text)[:200]!r} -> dùng mọi công cụ"
    cc = d.get("cong_cu")
    if isinstance(cc, str):
        cc = [cc]
    ds = []
    for x in (cc if isinstance(cc, list) else []):
        x = str(x).strip().strip("\"'")
        if x in CONG_CU and x not in ds:
            ds.append(x)
    return ds[:2], cau_chuan_hop_le(q, d.get("cau_hoi_chuan")), f"model: {json.dumps(d, ensure_ascii=False)[:400]}"


# --------------------------------------------------------------------------- bước 3: kiểm duyệt
SYS3 = "Bạn là người kiểm tra kết quả của công cụ tính toán trên dữ liệu bán hàng. Chỉ trả về JSON, không giải thích thêm."


def _mo_ta_ky(periods):
    out = []
    for p in periods:
        try:
            st = f"{pd.Timestamp(p.start):%Y-%m-%d}" if p.start is not None else "đầu dữ liệu"
            out.append(f"- {p.label}: {st} đến {pd.Timestamp(p.end):%Y-%m-%d}")
        except Exception:
            pass
    return "\n".join(out) or "- (câu hỏi không có mốc thời gian)"


def prompt_buoc3(q, stage, ans, mo_ta, bc, q_ky=None):
    a = json.dumps(ans, ensure_ascii=False, default=str)
    if len(a) > 700:
        a = a[:700] + "..."
    ky = _mo_ta_ky(bc.tinh(q_ky or q)["periods"])
    return (f"Câu hỏi của chủ shop: \"{q}\"\nHôm nay: {bc.as_of}.\n"
            f"Kỳ thời gian công cụ đã hiểu:\n{ky}\n"
            f"Công cụ: {stage}\nCách tính của công cụ: {mo_ta}\nĐáp án công cụ trả về: {a}\n\n"
            "Hãy kiểm tra đáp án có trả lời đúng điều câu hỏi hỏi không:\n"
            "(1) đúng kỳ thời gian; (2) đúng đối tượng (mặt hàng, khách, nhóm, nhân viên...); "
            "(3) đúng chỉ số (doanh thu, số đơn, số lượng, phần trăm...); "
            "(4) đúng kiểu đáp án (một số, phần trăm, danh sách, bảng, ngày, tên).\n"
            "CHỈ trả khop=false khi RÕ RÀNG sai một trong các điểm trên (ví dụ: hỏi doanh thu mà cách tính là số đơn; "
            "hỏi tháng 3 mà kỳ là tháng 4; hỏi 'ai' mà đáp án là một con số). "
            "Nếu chỉ khác cách diễn đạt, không chắc, hoặc là quy ước tính của shop thì khop=true. "
            "Không đánh giá con số nghe có hợp lý hay không.\n"
            "Trả về đúng một JSON: {\"khop\": true, \"ly_do\": \"...\"}")


def buoc3(q, stage, ans, mo_ta, bc, bat_buoc, q_ky=None):
    """-> (khop: bool, ghi chú). Lỗi model / JSON hỏng: đáp án chính -> coi là khớp; đáp án dự phòng (bat_buoc) -> không khớp."""
    if llm.BACKEND == "mock":
        return True, "mock: khop=true"
    try:
        text, _ = llm.chat([{"role": "system", "content": SYS3}, {"role": "user", "content": prompt_buoc3(q, stage, ans, mo_ta, bc, q_ky)}],
                           temperature=0, think=False)
    except Exception as e:
        return (not bat_buoc), f"lỗi gọi model bước 3: {e!r}"
    d = doc_json(text)
    if not d or "khop" not in d:
        return (not bat_buoc), f"bước 3 không đọc được JSON: {str(text)[:200]!r}"
    v = d["khop"]
    if isinstance(v, str):
        v = v.strip().lower() not in ("false", "no", "khong", "không", "0")
    return bool(v), f"model: {json.dumps(d, ensure_ascii=False)[:400]}"


# --------------------------------------------------------------------------- điều phối cả câu
def giai(question, bc):
    """-> dict kết quả như agent._solve_once: answer / thinking / code / output / steps / seconds."""
    t0 = time.time()
    q = question["question"]
    log = []
    tools, chuan, ghi = buoc1(q, bc)
    log.append(f"[bước 1] công cụ chọn: {tools or '(không chọn)'}; câu chuẩn: {chuan or '(giữ câu gốc)'}; {ghi}")
    chon = set()
    for t in tools:
        chon |= stage_cua(t)
    if TU_THEM_NGANH:
        them = [n for n in bc.nganh if n not in chon]
        if them:
            log.append(f"[bước 2] dữ liệu shop thuộc ngành {them} -> thêm module ngành vào nhóm chạy chính")
        chon |= set(bc.nganh)
    chinh = [s for s in THU_TU if s in chon]
    du_phong = [s for s in THU_TU if s not in chon]

    def xong(stage, r, q_dung, duong):
        ans, mo_ta = r
        can = KIEM == "all" or (KIEM == "fallback" and duong == "dự phòng")
        if can:
            khop, gc = buoc3(q, stage, ans, mo_ta, bc, bat_buoc=(duong == "dự phòng"), q_ky=q_dung)
            log.append(f"[bước 3] kiểm duyệt ({duong}, {stage}): khop={khop}; {gc}")
            if not khop:
                return {"answer": {"type": "abstain"}, "thinking": None,
                        "code": f"dieu_phoi: {stage}: {mo_ta}", "steps": 0,
                        "output": log + [f"Model kiểm duyệt từ chối đáp án {ans.get('answer')} -> bỏ qua"],
                        "seconds": round(time.time() - t0, 1)}
        else:
            log.append(f"[bước 3] không kiểm duyệt ({duong}, DIEUPHOI_KIEM={KIEM})")
        return {"answer": ans, "thinking": None, "code": f"dieu_phoi/{duong}: {mo_ta}",
                "output": log + [f"Giải bằng công thức cố định: {mo_ta} -> {ans.get('answer')}"],
                "steps": 0, "seconds": round(time.time() - t0, 1)}

    s, r = chay_cac(chinh, q, bc)
    if r is not None:
        return xong(s, r, q, "chính")
    if chuan and chuan != q and chinh:
        s, r = chay_cac(chinh, chuan, bc)
        if r is not None:
            log.append(f"[bước 2] ra kết quả trên câu chuẩn: {chuan}")
            return xong(s, r, chuan, "chính, câu chuẩn")
    s, r = chay_cac(du_phong, q, bc)
    if r is not None:
        log.append("[bước 2] công cụ model chọn không ra kết quả -> dùng module dự phòng")
        return xong(s, r, q, "dự phòng")
    return {"answer": {"type": "abstain"}, "thinking": None, "code": "", "steps": 0,
            "output": log + ["Không công cụ nào ra kết quả -> bỏ qua (chế độ dieu_phoi)"],
            "seconds": round(time.time() - t0, 1)}
