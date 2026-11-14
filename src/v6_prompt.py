# -*- coding: utf-8 -*-
"""v6_prompt — dựng prompt cho agent v6 (model hiểu câu -> DSL; model đọc kết quả engine).

DÙNG CHUNG cho chạy thật (src/agent_v6.py) và sinh dữ liệu huấn luyện (training_code/dsl/sinh_du_lieu.py):
prompt lúc huấn luyện phải GIỐNG HỆT prompt lúc chạy, nên mọi chỗ dựng prompt nằm ở file này.

API
    prompt_hieu_cau(question, as_of, ngu_canh_shop, vi_du, max_token=1500) -> messages   # M1: câu -> {"hieu","dsl"}
    prompt_doc_ket_qua(question, hieu, dsl, ket_qua, as_of=None, ngu_canh="") -> messages # M2: đọc kết quả / lỗi engine
    ngu_canh_shop(biz) -> str          # gọn: ngành, khoảng ngày, bảng có/không, <= 12 giá trị mẫu mỗi cột
    nap_kho_vi_du(...) -> KhoViDu      # ví dụ (câu -> DSL) từ train, KHÔNG gồm eval300 / public
    vi_du_gan(question, kho, k=6, loai_tru_id=None) -> list[dict]
    hieu_tu_dsl(dsl) -> str            # câu "hieu" dựng tự động từ DSL (cho ví dụ viết tay, vì chúng không có "hieu")
    tra_loi_m1(hieu, dsl) / tra_loi_m2(...) -> chuỗi JSON gọn (định dạng chuẩn của câu trả lời, dùng làm nhãn SFT)
Không có logic theo id / business_id / tên shop.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOKEN = int(os.environ.get("V6_MAX_TOKEN", "1500"))       # ngân sách prompt M1 (ước lượng)
SO_VI_DU = int(os.environ.get("V6_SO_VI_DU", "6"))            # số ví dụ tối đa trong prompt M1 (tự giảm nếu vượt ngân sách)
KY_TU_MOI_TOKEN = 2.4                                         # ước lượng thô tiếng Việt + JSON (đo lại bằng tokenizer thật nếu cần)


def uoc_token(text: str) -> int:
    return int(len(text) / KY_TU_MOI_TOKEN) + 4


def messages_token(messages) -> int:
    return sum(uoc_token(m["content"]) + 4 for m in messages)


def _gon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def fold(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


# ============================================================================ đặc tả DSL (gọn) — khớp src/dsl_engine.py
SYSTEM_M1 = """Bạn là bộ não của trợ lý dữ liệu cửa hàng nhỏ ở Việt Nam. Đọc câu hỏi của chủ shop, viết ĐẶC TẢ TRUY VẤN (DSL, JSON) cho công cụ tính; bạn KHÔNG tự tính số. Chỉ trả về MỘT JSON: {"hieu":"<1 câu: hỏi gì, kỳ nào, điều kiện gì, kiểu đáp án>","dsl":{...}}
DSL (khoá không cần thì bỏ):
loai: chi_so (một số) | so_sanh (% đổi / chênh lệch 2 kỳ hoặc 2 nhóm) | ti_trong (% chiếm trong mẫu số) | xep_hang (top / nhiều nhất) | dem_nguong (đếm khách/đơn/ngày/mặt hàng đạt ngưỡng) | theo_ky (từng ngày/tuần/tháng/quý) | tu_giai | khong_du_du_lieu
chi_so: doanh_thu so_don so_luong gia_tri_tb_don so_khach lai_gop so_don_huy ti_le_huy so_don_tra_hang so_mat_hang so_mat_hang_e(ế) so_ngay_ban so_ngay_mua gia_ban | tien_nhap so_phieu_nhap gia_nhap_tb | con_no thu_no no_them so_khach_no | so_dot_giao
ky: chép NGUYÊN VĂN cụm thời gian trong câu ("tháng 6/2025", "tuần trước"); nhiều kỳ -> list. ky_so_sanh: cụm khác, hoặc ky_truoc | cung_ky_nam_truoc | cung_ky_thang_truoc
loc: {mat_hang,nhom_hang,kenh,khach,nhom_khach,nhan_vien,thanh_toan,ca(sang|chieu|toi),hoat_chat,ke_don(true|false),don_vi,size,thu(2..7|CN),trang_thai(huy|tra_hang|dang_giao|hoan_thanh),nha_cung_cap,tuoi_no_hon,tuoi_no_tu,tuoi_no_den}; tên chép từ câu hoặc danh sách shop. loc_so_sanh: điều kiện vế so sánh (cùng kỳ); loc_mau: mẫu số của ti_trong (mặc định cả shop).
nhom_theo: mat_hang khach nhan_vien ngay thang tuan quy nam kenh nhom_hang nhom_khach thanh_toan gio nha_cung_cap thu don
sap_xep: giam|tang; top_k: số; nguong: {chi_so,toi_thieu(>=),lon_hon(>),toi_da(<=),nho_hon(<)}; doi_tuong_dem: khach|don|ngay|mat_hang
kieu_dap_an: number|percent|string|list|table|date|boolean (xep_hang top_k=1: string, hoặc date khi nhom_theo=ngay/thang)
cac_phan: [{khoá ghi đè},...] khi cần nhiều số; trung_binh_ngay: true khi hỏi trung bình mỗi ngày.
tu_giai: tồn kho, lô/hạn dùng, chi phí/lãi ròng, ngành riêng (sửa xe, điện thoại IMEI/trả góp, mẹ và bé), hoặc DSL không diễn tả được.
khong_du_du_lieu (kèm ly_do): shop không ghi thông tin hỏi (giới tính/địa chỉ khách, đánh giá nhân viên...) hoặc kỳ hỏi ngoài khoảng dữ liệu.
Hỏi "bao nhiêu %" -> percent; hỏi tên -> string/list; không đổi cụm thời gian thành ngày; không bịa tên."""

# bản rút gọn hơn cho prompt M2 (model chỉ cần nhớ khoá để sửa)
DSL_TOM_TAT = """Khoá DSL: loai(chi_so|so_sanh|ti_trong|xep_hang|dem_nguong|theo_ky|tu_giai|khong_du_du_lieu), chi_so, ky (chép nguyên văn từ câu), ky_so_sanh, loc{mat_hang,nhom_hang,kenh,khach,nhom_khach,nhan_vien,thanh_toan,ca,hoat_chat,ke_don,don_vi,size,thu,trang_thai,nha_cung_cap,tuoi_no_*}, loc_so_sanh, loc_mau, nhom_theo, sap_xep, top_k, nguong{chi_so,toi_thieu|lon_hon|toi_da|nho_hon}, doi_tuong_dem, kieu_dap_an, cac_phan, trung_binh_ngay."""

SYSTEM_M2 = """Bạn kiểm tra kết quả của công cụ tính trong trợ lý dữ liệu cửa hàng. Đọc câu hỏi, đặc tả DSL đã dùng và kết quả (đáp án + cách tính + bảng trung gian) hoặc lỗi của công cụ. Chỉ trả về MỘT JSON: {"chap_nhan":true|false,"ly_do":"<ngắn>","dsl_sua":{...}|null,"chuyen_tu_giai":true|false}
- chap_nhan=true khi: kỳ trong "cách tính" đúng kỳ câu hỏi; điều kiện lọc đủ, không thừa; đúng đại lượng; kiểu đáp án đúng điều được hỏi (số / % / tên / danh sách / ngày); độ lớn hợp lý.
- Sai chỗ nào thì chap_nhan=false và dsl_sua = đặc tả ĐÃ SỬA ĐẦY ĐỦ (tên chép từ câu hoặc danh sách shop).
- Công cụ báo lỗi: đọc lỗi (gợi ý tên gần đúng, tên mơ hồ, khoá sai...) rồi sửa dsl_sua. Nếu công cụ không thể diễn tả câu hỏi: chuyen_tu_giai=true, dsl_sua=null.
""" + DSL_TOM_TAT

THU = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")


def _hom_nay(as_of) -> str:
    s = str(as_of or "")[:10]
    try:
        d = date.fromisoformat(s)
        return f"{s} ({THU[d.weekday()]})"
    except Exception:
        return s or "không rõ"


# ============================================================================ "hieu" tự động từ DSL
_TEN_CS = {"doanh_thu": "doanh thu", "so_don": "số đơn", "so_luong": "số lượng bán", "gia_tri_tb_don": "giá trị TB mỗi đơn",
           "so_khach": "số khách", "lai_gop": "lãi gộp", "so_don_huy": "số đơn huỷ", "ti_le_huy": "tỉ lệ huỷ",
           "so_don_tra_hang": "số đơn trả hàng", "so_mat_hang": "số mặt hàng", "so_mat_hang_e": "số mặt hàng ế",
           "so_ngay_ban": "số ngày bán", "so_ngay_mua": "số ngày có mua", "gia_ban": "giá bán", "tien_nhap": "tiền nhập",
           "so_phieu_nhap": "số phiếu nhập", "gia_nhap_tb": "giá nhập bình quân", "con_no": "còn nợ", "thu_no": "thu nợ",
           "no_them": "nợ thêm", "so_khach_no": "số khách còn nợ", "so_dot_giao": "số đợt giao"}
_TEN_NHOM = {"mat_hang": "mặt hàng", "khach": "khách", "nhan_vien": "nhân viên", "ngay": "ngày", "thang": "tháng", "kenh": "kênh",
             "nhom_hang": "nhóm hàng", "gio": "giờ", "nha_cung_cap": "nhà cung cấp", "tuan": "tuần", "quy": "quý", "nam": "năm",
             "nhom_khach": "nhóm khách", "thanh_toan": "hình thức thanh toán", "don": "đơn", "thu": "thứ"}


def _ky_txt(ky) -> str:
    if isinstance(ky, list):
        return " & ".join(_ky_txt(k) for k in ky)
    if isinstance(ky, dict):
        return f"{ky.get('tu', '?')}..{ky.get('den', '?')}"
    return str(ky) if ky else "(không nêu)"


def _loc_txt(loc) -> str:
    if not isinstance(loc, dict) or not loc:
        return ""
    return ", ".join(f"{k}={v if not isinstance(v, list) else '/'.join(map(str, v))}" for k, v in loc.items())


def hieu_tu_dsl(dsl) -> str:
    """Câu 'hieu' ngắn dựng tự động từ DSL (cho ví dụ viết tay không có 'hieu')."""
    if not isinstance(dsl, dict):
        return ""
    loai = dsl.get("loai")
    if loai == "tu_giai":
        return "Câu cần tự giải bằng code (DSL không diễn tả được)"
    if loai == "khong_du_du_lieu":
        return "Dữ liệu shop không có thông tin này: " + str(dsl.get("ly_do") or dsl.get("ghi_chu") or "không đủ dữ liệu")[:80]
    cs = _TEN_CS.get(dsl.get("chi_so"), dsl.get("chi_so") or "?")
    ky = _ky_txt(dsl.get("ky"))
    loc = _loc_txt(dsl.get("loc"))
    kieu = dsl.get("kieu_dap_an") or "?"
    tail = f"; lọc {loc}" if loc else ""
    if dsl.get("cac_phan"):
        tail += f"; {len(dsl['cac_phan'])} số"
    if loai == "so_sanh":
        ks = dsl.get("ky_so_sanh")
        vs = f" với {_ky_txt(ks)}" if ks else (f" với vế {_loc_txt(dsl.get('loc_so_sanh'))}" if dsl.get("loc_so_sanh") else "")
        return f"So sánh {cs} kỳ {ky}{vs}{tail}; {kieu}"
    if loai == "ti_trong":
        return f"Tỉ trọng {cs} kỳ {ky}{tail} trên mẫu số{(' ' + _loc_txt(dsl.get('loc_mau'))) if dsl.get('loc_mau') else ' cả shop'}; {kieu}"
    if loai == "xep_hang":
        k = dsl.get("top_k")
        return f"Xếp hạng {_TEN_NHOM.get(dsl.get('nhom_theo'), dsl.get('nhom_theo'))} theo {cs} ({dsl.get('sap_xep', 'giam')}{', top ' + str(k) if k else ''}) kỳ {ky}{tail}; {kieu}"
    if loai == "dem_nguong":
        ng = dsl.get("nguong") or {}
        return f"Đếm {dsl.get('doi_tuong_dem')} có {_TEN_CS.get(ng.get('chi_so'), ng.get('chi_so'))} đạt ngưỡng kỳ {ky}{tail}; {kieu}"
    if loai == "theo_ky":
        return f"{cs.capitalize()} theo {_TEN_NHOM.get(dsl.get('nhom_theo'), dsl.get('nhom_theo'))} trong {ky}{tail}; {kieu}"
    return f"{cs.capitalize()} kỳ {ky}{tail}; {kieu}"


def tra_loi_m1(hieu, dsl) -> str:
    """Câu trả lời chuẩn của M1 (nhãn SFT)."""
    return _gon({"hieu": str(hieu or hieu_tu_dsl(dsl))[:200], "dsl": dsl})


def tra_loi_m2(chap_nhan, ly_do="", dsl_sua=None, chuyen_tu_giai=False) -> str:
    return _gon({"chap_nhan": bool(chap_nhan), "ly_do": str(ly_do or "")[:160], "dsl_sua": dsl_sua,
                 "chuyen_tu_giai": bool(chuyen_tu_giai)})


# ============================================================================ ngữ cảnh shop
_CAC_COT = (("nhom_hang", ("nhom_hang",)), ("kenh", ("kenh",)), ("nhom_khach", ("nhom_khach",)),
            ("nhan_vien", ("ten_nhan_vien", "nhan_vien")), ("thanh_toan", ("pttt", "thanh_toan")))
_TEN_NGANH = {"suaxe": "sửa xe", "mevabe": "mẹ và bé", "dienthoai": "điện thoại"}


def _gia_tri_mau(ban, cols, toi_da=12, ngan_sach=170):
    for c in cols:
        if c in ban.columns:
            s = ban[c].dropna().astype(str).str.strip()
            s = s[(s != "") & (s.str.lower() != "nan")]
            if not len(s):
                continue
            out, tong = [], 0
            for v in s.value_counts().index[:toi_da]:
                v = v[:32]
                if tong + len(v) + 2 > ngan_sach and out:
                    break
                out.append(v)
                tong += len(v) + 2
            return out
    return []


def ngu_canh_shop(biz, ban=None) -> str:
    """Mô tả GỌN shop cho model: ngành, khoảng ngày, bảng có/không, <= 12 giá trị mẫu mỗi cột. Cache trên biz."""
    cache = getattr(biz, "__dict__", {}).get("_v6_ngu_canh")
    if cache is not None and ban is None:
        return cache
    dong = []
    nganh = []
    try:
        import cong_cu
        nganh = [_TEN_NGANH.get(n, n) for n in cong_cu.nganh_cua_shop(biz)]
    except Exception:
        pass
    if ban is None:
        ban = getattr(biz, "ban", None)
    if ban is None:
        try:
            from sales import build_sales
            ban = build_sales(biz)
        except Exception:
            ban = None
    co = []
    for ten, thuoc in (("nhập hàng", "nhap"), ("công nợ", "no_don"), ("giao hàng", "giao")):
        d = getattr(biz, thuoc, None)
        if d is not None and len(d):
            co.append(ten)
    dong_dau = "Shop: ngành " + (", ".join(nganh) if nganh else "bán lẻ / dịch vụ chung")
    if ban is not None and len(ban):
        try:
            lo, hi = ban["ngay"].min(), ban["ngay"].max()
            dong_dau += f"; có dữ liệu bán {lo:%Y-%m-%d}..{hi:%Y-%m-%d}"
        except Exception:
            pass
        try:
            if ban["thoi_diem"].dt.hour.nunique() > 1:
                co.append("giờ tạo đơn")
        except Exception:
            pass
    dong_dau += "; có: " + (", ".join(co) if co else "chỉ bán hàng")
    dong.append(dong_dau)
    if ban is not None:
        for ten, cols in _CAC_COT:
            gt = _gia_tri_mau(ban, cols)
            if gt:
                dong.append(f"{ten}: " + "; ".join(gt))
    out = "\n".join(dong)
    try:
        biz.__dict__["_v6_ngu_canh"] = out
    except Exception:
        pass
    return out


# ============================================================================ kho ví dụ
class KhoViDu(list):
    """Danh sách {"id","question","dsl","hieu"?} + chỉ mục từ (idf) để chọn ví dụ gần nhất."""

    def __init__(self, rows=()):
        super().__init__()
        self.df = Counter()
        self.ids = set()
        self.cau = set()
        for r in rows:
            self.them(r)

    def them(self, r):
        if not isinstance(r, dict) or not r.get("question") or not isinstance(r.get("dsl"), dict):
            return False
        fq = fold(r["question"])
        if r.get("id") in self.ids or fq in self.cau:
            return False
        r = dict(r)
        r["_tok"] = _toks(r["question"])
        r["_loai"] = r["dsl"].get("loai")
        self.append(r)
        self.ids.add(r.get("id"))
        self.cau.add(fq)
        self.df.update(set(r["_tok"]))
        return True

    def idf(self, t):
        return math.log((len(self) + 2) / (self.df.get(t, 0) + 0.7))


def _toks(text):
    w = re.findall(r"[a-z0-9]+", fold(text))
    out = [t if not t.isdigit() else "#" for t in w]
    out += [f"{a}_{b}" for a, b in zip(w, w[1:]) if not (a.isdigit() or b.isdigit())]
    return out


def _doc_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return


def _tim_file(*rels):
    cands = [os.path.join(os.getcwd(), *r.split("/")) for r in rels] + \
            [os.path.join(HERE, "..", *r.split("/")) for r in rels]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def cau_cam():
    """(tập id, tập câu đã fold) của eval300 + public: KHÔNG được làm ví dụ."""
    ids, cau = set(), set()
    paths = [os.environ.get("V6_EVAL300") or _tim_file("data/eval300.jsonl")]
    pub = _tim_file("data/public_test/questions.jsonl")
    paths.append(pub)
    for p in paths:
        if p:
            for r in _doc_jsonl(p):
                ids.add(r.get("id"))
                if r.get("question"):
                    cau.add(fold(r["question"]))
    return ids, cau


def nap_kho_vi_du(files=None, them=()) -> KhoViDu:
    """Nạp ví dụ từ vi_du_v6.jsonl (cạnh file này) + training_code/dsl/vi_du_tay.jsonl + vi_du_sinh.jsonl nếu có
    (hoặc biến V6_KHO = các đường dẫn cách nhau dấu ':'). Loại ví dụ thuộc eval300 / public."""
    if files is None:
        env = os.environ.get("V6_KHO")
        if env:
            files = [p for p in env.split(":") if p]
        else:
            files = [os.path.join(HERE, "vi_du_v6.jsonl"), os.path.join(HERE, "vi_du_sinh_v6.jsonl")]
            for rel in ("training_code/dsl/vi_du_tay.jsonl", "training_code/dsl/vi_du_sinh.jsonl"):
                p = _tim_file(rel)
                if p:
                    files.append(p)
    ids_cam, cau_cam_ = cau_cam()
    kho = KhoViDu()
    n_cam = 0
    for p in files:
        for r in _doc_jsonl(p):
            if r.get("id") in ids_cam or fold(r.get("question", "")) in cau_cam_:
                n_cam += 1
                continue
            kho.them(r)
    for r in them:
        kho.them(r)
    kho.so_bi_loai = n_cam
    return kho


def vi_du_gan(question, kho, k=SO_VI_DU, loai_tru_id=None):
    """k ví dụ gần câu hỏi nhất (trùng từ + cụm 2 từ, trọng số idf), đa dạng loai (mỗi loai tối đa ceil(k/2) ở vòng 1)."""
    if not isinstance(kho, KhoViDu):
        kho = KhoViDu(kho)
    if not len(kho) or k <= 0:
        return []
    q = Counter(_toks(question))
    fq = fold(question)
    nq = math.sqrt(sum(kho.idf(t) ** 2 for t in q)) or 1.0
    diem = []
    for r in kho:
        if loai_tru_id is not None and r.get("id") == loai_tru_id:
            continue
        if fold(r["question"]) == fq:
            continue
        te = set(r["_tok"])
        chung = sum(kho.idf(t) ** 2 for t in q if t in te)
        if chung <= 0:
            continue
        ne = math.sqrt(sum(kho.idf(t) ** 2 for t in te)) or 1.0
        diem.append((chung / (nq * ne), r["question"], r))
    diem.sort(key=lambda x: (-x[0], x[1]))
    tran = max(2, -(-k // 2))
    chon, dem_loai, du = [], Counter(), []
    for d, _, r in diem:
        if len(chon) >= k:
            break
        if dem_loai[r["_loai"]] < tran:
            chon.append(r)
            dem_loai[r["_loai"]] += 1
        else:
            du.append(r)
    for r in du:
        if len(chon) >= k:
            break
        chon.append(r)
    return [{x: v for x, v in r.items() if not x.startswith("_")} for r in chon]


# ============================================================================ prompt M1 / M2
def _dong_vi_du(e) -> str:
    return f"Hỏi: {e['question'].strip()}\nĐáp: {tra_loi_m1(e.get('hieu') or hieu_tu_dsl(e['dsl']), e['dsl'])}"


def prompt_hieu_cau(question, as_of, ngu_canh_shop, vi_du=(), max_token=None):
    """M1: messages [system, user]. Tự bớt ví dụ (từ xa nhất) cho tới khi vừa ngân sách token (>= 0 ví dụ)."""
    max_token = MAX_TOKEN if max_token is None else max_token
    vi_du = list(vi_du)[:SO_VI_DU] if vi_du is not None else []

    def dung(vds):
        parts = [f"Hôm nay: {_hom_nay(as_of)}", (ngu_canh_shop or "").strip()]
        if vds:
            parts.append("Ví dụ:\n" + "\n".join(_dong_vi_du(e) for e in vds))
        parts.append(f"Câu hỏi: {question.strip()}\nTrả về JSON.")
        return [{"role": "system", "content": SYSTEM_M1},
                {"role": "user", "content": "\n".join(p for p in parts if p)}]

    ms = dung(vi_du)
    while vi_du and messages_token(ms) > max_token:
        vi_du = vi_du[:-1]
        ms = dung(vi_du)
    return ms


def _ket_qua_txt(ket_qua) -> str:
    if not isinstance(ket_qua, dict):
        return str(ket_qua)[:600]
    if not ket_qua.get("ok"):
        return "Công cụ báo LỖI: " + str(ket_qua.get("loi") or "không rõ")[:700]
    ans = ket_qua.get("answer")
    s = "Đáp án: " + _gon(ans)[:500]
    if ket_qua.get("mo_ta"):
        s += "\nCách tính: " + str(ket_qua["mo_ta"])[:600]
    if ket_qua.get("trung_gian"):
        s += "\nBảng trung gian:\n" + str(ket_qua["trung_gian"])[:700]
    return s


def prompt_doc_ket_qua(question, hieu, dsl, ket_qua, as_of=None, ngu_canh=""):
    """M2: đọc kết quả engine (hoặc lỗi) -> {"chap_nhan","ly_do","dsl_sua","chuyen_tu_giai"}."""
    parts = []
    if as_of:
        parts.append(f"Hôm nay: {_hom_nay(as_of)}")
    if ngu_canh:
        parts.append(ngu_canh.strip())
    parts += [f"Câu hỏi: {question.strip()}", f"Hiểu: {str(hieu or '')[:200]}", f"DSL: {_gon(dsl)}", _ket_qua_txt(ket_qua),
              "Trả về JSON."]
    return [{"role": "system", "content": SYSTEM_M2}, {"role": "user", "content": "\n".join(parts)}]


if __name__ == "__main__":          # thử nhanh: python src/v6_prompt.py "câu hỏi"
    import sys
    q = " ".join(sys.argv[1:]) or "Doanh thu tháng 6/2025 của nhóm cà phê là bao nhiêu?"
    kho = nap_kho_vi_du()
    print(f"kho: {len(kho)} ví dụ (loại {getattr(kho, 'so_bi_loai', 0)})")
    ms = prompt_hieu_cau(q, "2026-11-14", "Shop: ngành bán lẻ", vi_du_gan(q, kho))
    for m in ms:
        print(f"--- {m['role']} (~{uoc_token(m['content'])} token)\n{m['content']}")
    print("TỔNG ~", messages_token(ms), "token")
