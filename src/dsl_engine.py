# -*- coding: utf-8 -*-
"""
dsl_engine — công cụ tính TỔNG QUÁT thực thi "đặc tả truy vấn" (DSL) do MODEL sinh ra (kiến trúc v6, docs/kien_truc_v6.md §2).

Engine KHÔNG đọc câu hỏi: chỉ nhận dict DSL, tính trên bảng chuẩn `biz.ban` (+ `biz.nhap`, `biz.no_don`, `biz.thu_tien`,
`biz.giao`) theo rulebook (§1 doanh thu, §3 khách / hoà nhau theo mã / phí không là mặt hàng, §4 công nợ, §5 nhập, §9 làm tròn).

API
    thuc_thi(dac_ta, biz, rulebook, cal, as_of) -> {"ok", "answer", "mo_ta", "trung_gian", "loi", ...}
    kiem_tra_dac_ta(dac_ta) -> list[str]        # lỗi schema (rỗng = hợp lệ)

Kết quả: ok=True & answer={"type","answer"} khi tính được (kể cả answer.type == "unanswerable" theo rulebook §7: kỳ nằm hoàn toàn
ngoài dữ liệu / shop không ghi loại thông tin cần; khi đó thêm khoá "khong_tra_loi_duoc": True).
ok=False & loi (tiếng Việt, để model sửa đặc tả) khi đặc tả sai / mơ hồ / không đủ dữ liệu để engine tính.

------------------------------------------------------------------------------------------------------------------------
DSL (đúng như docs/kien_truc_v6.md §2, KÈM các giá trị MỞ RỘNG — đánh dấu [+])
  loai        chi_so | so_sanh | ti_trong | xep_hang | dem_nguong | theo_ky | khong_du_du_lieu | tu_giai
  chi_so      doanh_thu | so_don | so_luong | gia_tri_tb_don | so_khach | lai_gop | so_don_huy | tien_nhap | so_phieu_nhap |
              con_no | thu_no | no_them | so_khach_no | so_ngay_ban
              [+] ti_le_huy (percent) | so_don_tra_hang | so_mat_hang (số mặt hàng khác nhau đã bán) |
                  so_mat_hang_e (mặt hàng đã từng bán trước kỳ nhưng kỳ này không bán) | so_ngay_mua (số ngày có mua; dùng khi
                  xếp hạng khách "mua nhiều ngày nhất") | gia_ban (đơn giá 1 đơn vị cơ bản của mat_hang vào MỘT ngày, trước chiết
                  khấu, tính cả đơn huỷ/trả) | gia_nhap_tb (giá nhập bình quân theo số lượng nhập của mat_hang trong kỳ) |
                  so_dot_giao (số đợt giao hàng; mỗi đợt của mỗi đơn tính một lần)
  ky          cụm thời gian chép nguyên văn từ câu -> periods.find_periods(ky, as_of, cal).
              [+] có thể là LIST các cụm (mỗi cụm một kỳ -> answer là list theo đúng thứ tự), hoặc dict {"tu":"YYYY-MM-DD","den":"YYYY-MM-DD"},
              hoặc chuỗi ISO "YYYY-MM" / "YYYY-MM-DD"; cụm "N ngày kể từ ngày D" / "N ngày ngay trước ngày D" cũng được hiểu.
  ky_so_sanh  cụm thời gian khác hoặc "ky_truoc" | "cung_ky_nam_truoc" | "cung_ky_thang_truoc"
  loc         {"mat_hang","nhom_hang","kenh","khach","nhom_khach","nhan_vien","thanh_toan","ca","hoat_chat","ke_don","don_vi"} + [+]
              "size":"M|L" (món tên kết thúc bằng (M)/(L)), "thu":"2..7|CN" (thứ trong tuần), "trang_thai":"huy|tra_hang|dang_giao|hoan_thanh"
              (chỉ cho chi_so so_don: đếm đơn theo trạng thái, không lọc tinh_dt), "nha_cung_cap" (chỉ dữ liệu nhập),
              "tuoi_no_hon":N (nợ của đơn đặt quá N ngày), "tuoi_no_tu":a,"tuoi_no_den":b (tuổi nợ a..b ngày, gồm cả hai đầu).
              Giá trị có thể là chuỗi hoặc LIST chuỗi (hợp các giá trị). Tên khớp bỏ dấu + gần đúng; nhiều kết quả -> lỗi liệt kê.
  loc_so_sanh [+] loc dùng cho vế SO SÁNH của so_sanh (hoà vào loc): vd ca chiều so với ca sáng cùng một kỳ.
  loc_mau     [+] loc dùng cho MẪU SỐ của ti_trong (mặc định {} = cả shop cùng kỳ).
  nhom_theo   mat_hang | khach | nhan_vien | ngay | thang | kenh | nhom_hang | gio | nha_cung_cap   [+] tuan | quy | nam | nhom_khach |
              thanh_toan | don | thu
  sap_xep     giam | tang      top_k  số nguyên
  nguong      {"chi_so", "toi_thieu"(>=), "nghiem_ngat"(true: toi_thieu thành >)}  [+] "lon_hon"(>), "toi_da"(<=), "nho_hon"(<)
              dem_nguong: điều kiện đếm. [+] xep_hang: "chỉ xét nhóm có ... " (vd khách có từ 3 đơn trở lên).
  doi_tuong_dem khach | don | ngay | mat_hang
  kieu_dap_an number | percent | string | list | table | date | boolean
  [+] cac_phan  list các dict ghi đè khoá của đặc tả cha (loc được hoà): nhiều số trong một câu -> answer type list đúng thứ tự.
  [+] trung_binh_ngay  true: chia kết quả cho SỐ NGÀY của kỳ (nếu loc.thu thì chỉ đếm ngày thứ đó, kể cả ngày không có đơn).
  [+] ly_do / ghi_chu  (văn bản tự do, bỏ qua) — với loai=khong_du_du_lieu, ly_do là lý do unanswerable.

loai:
  chi_so      một số. so_sanh: % thay đổi (kieu_dap_an=percent, mặc định) = (kỳ hỏi − kỳ so)/kỳ so × 100, hoặc chênh lệch (number) = kỳ hỏi −
              kỳ so, hoặc boolean (kỳ hỏi > kỳ so). ti_trong: % = chi_so(loc) / chi_so(loc_mau) × 100.
  xep_hang    xếp theo chi_so giảm/tăng, hoà theo mã tăng dần; top_k=1 -> string (hoặc date khi nhom_theo=ngay/thang), top_k>1 -> list tên,
              kieu_dap_an=table -> {"columns":[tên đối tượng, số đo], rows}.
  dem_nguong  đếm đối tượng (khach/don/ngay/mat_hang) có chỉ số thoả nguong; kieu_dap_an=percent -> % trên tổng số đối tượng.
  theo_ky     chi_so cho từng tháng/ngày/tuần/quý/năm trong ky -> list giá trị; kieu_dap_an=table -> bảng; boolean -> "tăng liên tục?"
              (sap_xep=tang mặc định, tức mỗi kỳ > kỳ liền trước; sap_xep=giam: mỗi kỳ < kỳ liền trước).
Định dạng: tiền = số nguyên (làm tròn nửa lên), % = 2 chữ số, ngày "YYYY-MM-DD", tháng "YYYY-MM" (theo rulebook §9).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

try:
    from periods import Period, find_periods, previous, same_period_last_year, month as _month_p, snapshot as _snapshot_p
except ImportError:                                  # pragma: no cover
    from .periods import Period, find_periods, previous, same_period_last_year, month as _month_p, snapshot as _snapshot_p

# ----------------------------------------------------------------------------- hằng
LOAI = ("chi_so", "so_sanh", "ti_trong", "xep_hang", "dem_nguong", "theo_ky", "khong_du_du_lieu", "tu_giai")
CHI_SO_BAN = ("doanh_thu", "so_don", "so_luong", "gia_tri_tb_don", "so_khach", "lai_gop", "so_don_huy", "so_ngay_ban",
              "ti_le_huy", "so_don_tra_hang", "so_mat_hang", "so_mat_hang_e", "so_ngay_mua", "gia_ban")
CHI_SO_NHAP = ("tien_nhap", "so_phieu_nhap", "gia_nhap_tb")
CHI_SO_NO = ("con_no", "thu_no", "no_them", "so_khach_no")
CHI_SO_GIAO = ("so_dot_giao",)
CHI_SO = CHI_SO_BAN + CHI_SO_NHAP + CHI_SO_NO + CHI_SO_GIAO
LOC_KEYS = ("mat_hang", "nhom_hang", "kenh", "khach", "nhom_khach", "nhan_vien", "thanh_toan", "ca", "hoat_chat", "ke_don",
            "don_vi", "size", "thu", "trang_thai", "nha_cung_cap", "tuoi_no_hon", "tuoi_no_tu", "tuoi_no_den")
NHOM_THEO = ("mat_hang", "khach", "nhan_vien", "ngay", "thang", "kenh", "nhom_hang", "gio", "nha_cung_cap",
             "tuan", "quy", "nam", "nhom_khach", "thanh_toan", "don", "thu")
THOI_GIAN = ("ngay", "thang", "tuan", "quy", "nam")
KIEU = ("number", "percent", "string", "list", "table", "date", "boolean")
KHOA = ("loai", "chi_so", "ky", "ky_so_sanh", "loc", "loc_so_sanh", "loc_mau", "nhom_theo", "sap_xep", "top_k", "nguong",
        "doi_tuong_dem", "kieu_dap_an", "cac_phan", "trung_binh_ngay", "ly_do", "ghi_chu", "giai_thich", "ke_hoach")
KHOA_BO_QUA = ("ly_do", "ghi_chu", "giai_thich", "ke_hoach", "ghi_chu_model", "suy_luan")
NGUONG_KEYS = ("chi_so", "toi_thieu", "nghiem_ngat", "lon_hon", "toi_da", "nho_hon")
DOI_TUONG = ("khach", "don", "ngay", "mat_hang")

MO_TA_CHI_SO = {
    "doanh_thu": "Doanh thu = tổng thành tiền các dòng của đơn không huỷ và không trả hàng (theo ngày đặt)",
    "so_don": "Số đơn = số mã đơn khác nhau của đơn không huỷ, không trả",
    "so_luong": "Số lượng = tổng số lượng quy về đơn vị cơ bản của dòng thuộc đơn tính doanh thu (bỏ dòng phí)",
    "gia_tri_tb_don": "Giá trị TB một đơn = doanh thu / số đơn, làm tròn nửa lên",
    "so_khach": "Số khách = số khách (mã khách; không có mã thì tên) có đơn tính doanh thu, bỏ khách lẻ",
    "lai_gop": "Lãi gộp = tổng (thành tiền − số lượng × giá vốn ghi trên dòng bán) của đơn tính doanh thu",
    "so_don_huy": "Số đơn huỷ = số mã đơn có trạng thái huỷ đặt trong kỳ",
    "so_ngay_ban": "Số ngày bán = số ngày có đơn tính doanh thu",
    "ti_le_huy": "Tỉ lệ huỷ = đơn huỷ / mọi đơn đặt trong kỳ (mọi trạng thái) × 100",
    "so_don_tra_hang": "Số đơn trả hàng = số mã đơn có trả hàng (kể cả trả một phần) đặt trong kỳ",
    "so_mat_hang": "Số mặt hàng = số mặt hàng khác nhau (bỏ dòng phí) có trong đơn tính doanh thu",
    "so_mat_hang_e": "Mặt hàng ế = đã bán (đơn tính doanh thu) trước kỳ nhưng kỳ này không có đơn nào mua",
    "so_ngay_mua": "Số ngày có mua = số ngày khác nhau có đơn tính doanh thu",
    "gia_ban": "Giá bán = đơn giá ghi trên dòng bán / đơn vị cơ bản, trước chiết khấu dòng, kể cả đơn huỷ/trả",
    "tien_nhap": "Tiền nhập = tổng tiền các phiếu nhập không huỷ có ngày phiếu trong kỳ",
    "so_phieu_nhap": "Số phiếu nhập = số mã phiếu nhập không huỷ có ngày phiếu trong kỳ",
    "gia_nhap_tb": "Giá nhập bình quân = tổng tiền nhập / tổng số lượng nhập (theo đơn vị nhập) của mặt hàng trong kỳ",
    "con_no": "Còn nợ đến hết ngày cuối kỳ = Σ (tổng đơn − các khoản đã trả đến hết ngày đó) của đơn ghi nợ (không huỷ) còn nợ > 0",
    "thu_no": "Thu nợ = các khoản trả SAU ngày mua, ngày trả trong kỳ, của mọi đơn ghi nợ không huỷ",
    "no_them": "Nợ thêm = phần tổng đơn chưa trả ngay trong ngày mua của đơn ghi nợ đặt trong kỳ",
    "so_khach_no": "Số khách còn nợ đến hết ngày cuối kỳ (bỏ khách lẻ)",
    "so_dot_giao": "Số đợt giao = số (mã đơn, đợt) có ngày giao trong kỳ",
}

_HOUR_DEFAULT = {"sang": (6 * 60, 13 * 60 + 59), "chieu": (14 * 60, 21 * 60 + 59), "toi": (22 * 60, 23 * 60 + 59)}


class DacTaLoi(Exception):
    """Lỗi đặc tả / không đủ dữ liệu để engine tính: model đọc `loi` để sửa đặc tả."""


class KhongTraLoiDuoc(Exception):
    """Rulebook §7: dữ liệu phát hành không đủ để trả lời -> unanswerable."""


# ----------------------------------------------------------------------------- chữ
def fold(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def _toks(s) -> list:
    return re.findall(r"[a-z0-9]+", fold(s))


def _nz(s) -> str:
    """Như scorer.norm_str: NFC, thường, ký tự lạ -> khoảng trắng (GIỮ dấu)."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    return " ".join("".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s).split())


def _enum(x, allowed):
    if x is None or isinstance(x, (dict, list, bool)):
        return None
    k = re.sub(r"[\s\-]+", "_", fold(x))
    return k if k in allowed else None


def _so(x, nhan="giá trị"):
    """Số từ int/float/chuỗi ("3.300.000", "15 triệu", "2,5")."""
    if isinstance(x, bool):
        raise DacTaLoi(f"{nhan} phải là số, không phải true/false.")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        t = fold(x).replace(" ", "")
        mul = 1.0
        for suf, m in (("trieu", 1e6), ("tr", 1e6), ("nghin", 1e3), ("ngan", 1e3), ("k", 1e3), ("ty", 1e9)):
            if t.endswith(suf) and re.match(r"^[\d.,]+", t):
                t, mul = t[: -len(suf)], m
                break
        t = re.sub(r"(dong|vnd|d)$", "", t)
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):
            return float(re.sub(r"[.,]", "", t)) * mul
        if re.fullmatch(r"\d+([.,]\d+)?", t):
            return float(t.replace(",", ".")) * mul
    raise DacTaLoi(f"{nhan} phải là một số (nhận được {x!r}).")


def _tien(v) -> int:
    return int(Decimal(repr(round(float(v), 6))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _sl(v):
    v = round(float(v), 6)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


def _pct2(num, den) -> float:
    n, d = Decimal(repr(round(float(num), 6))), Decimal(repr(round(float(den), 6)))
    return float((n * 100 / d).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _chia_tien(num, den) -> int:
    return int((Decimal(repr(round(float(num), 6))) / Decimal(repr(round(float(den), 6)))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _gon(v):
    """Giá trị đáp án JSON: số nguyên nếu tròn, ngược lại 2 chữ số."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return int(round(v)) if abs(v - round(v)) < 1e-6 else round(v, 2)


# ----------------------------------------------------------------------------- kiểm tra schema
def kiem_tra_dac_ta(dac_ta) -> list:
    """Lỗi schema của đặc tả (rỗng nếu hợp lệ). Không đụng tới dữ liệu."""
    loi = []
    if not isinstance(dac_ta, dict):
        return ["Đặc tả phải là một đối tượng JSON (dict), có khoá \"loai\"."]
    for k in dac_ta:
        if k not in KHOA and k not in KHOA_BO_QUA:
            loi.append(f"Khoá lạ \"{k}\". Các khoá hợp lệ: {', '.join(k2 for k2 in KHOA if k2 not in KHOA_BO_QUA)}.")
    loai = _enum(dac_ta.get("loai"), LOAI)
    if loai is None:
        loi.append(f"Thiếu hoặc sai \"loai\" (nhận được {dac_ta.get('loai')!r}). Giá trị hợp lệ: {' | '.join(LOAI)}.")
        return loi
    if loai in ("khong_du_du_lieu", "tu_giai"):
        return loi
    phan = dac_ta.get("cac_phan")
    if phan is not None:
        if not isinstance(phan, list) or not phan or not all(isinstance(x, dict) for x in phan):
            loi.append("\"cac_phan\" phải là list các dict (mỗi dict ghi đè khoá của đặc tả cha).")
        else:
            for i, ph in enumerate(phan):
                m = {k: v for k, v in dac_ta.items() if k not in ("cac_phan", "kieu_dap_an")}
                m.update({k: v for k, v in ph.items() if k != "loc"})
                m["loc"] = {**(dac_ta.get("loc") or {}), **(ph.get("loc") or {})}
                loi += [f"cac_phan[{i}]: {e}" for e in _kiem_tra_don(m)]
            return loi
    return loi + _kiem_tra_don(dac_ta)


def _kiem_tra_loc(loc, nhan="loc") -> list:
    loi = []
    if loc is None:
        return loi
    if not isinstance(loc, dict):
        return [f"\"{nhan}\" phải là dict, vd {{\"nhom_hang\": \"cà phê\"}}."]
    for k, v in loc.items():
        if k not in LOC_KEYS:
            loi.append(f"{nhan}: khoá lạ \"{k}\". Khoá hợp lệ: {', '.join(LOC_KEYS)}.")
        elif isinstance(v, (dict,)):
            loi.append(f"{nhan}.{k} phải là chuỗi (hoặc list chuỗi), không phải dict.")
        elif k in ("tuoi_no_hon", "tuoi_no_tu", "tuoi_no_den"):
            try:
                _so(v, f"{nhan}.{k}")
            except DacTaLoi as e:
                loi.append(str(e))
    return loi


def _kiem_tra_don(d) -> list:
    loi = []
    loai = _enum(d.get("loai"), LOAI)
    cs = _enum(d.get("chi_so"), CHI_SO)
    if d.get("chi_so") is not None and cs is None:
        loi.append(f"\"chi_so\"={d.get('chi_so')!r} không hợp lệ. Giá trị hợp lệ: {' | '.join(CHI_SO)}.")
    loi += _kiem_tra_loc(d.get("loc"), "loc") + _kiem_tra_loc(d.get("loc_so_sanh"), "loc_so_sanh") + _kiem_tra_loc(d.get("loc_mau"), "loc_mau")
    if d.get("kieu_dap_an") is not None and _enum(d.get("kieu_dap_an"), KIEU) is None:
        loi.append(f"\"kieu_dap_an\"={d.get('kieu_dap_an')!r} không hợp lệ. Giá trị hợp lệ: {' | '.join(KIEU)}.")
    if d.get("nhom_theo") is not None and _enum(d.get("nhom_theo"), NHOM_THEO) is None:
        loi.append(f"\"nhom_theo\"={d.get('nhom_theo')!r} không hợp lệ. Giá trị hợp lệ: {' | '.join(NHOM_THEO)}.")
    if d.get("doi_tuong_dem") is not None and _enum(d.get("doi_tuong_dem"), DOI_TUONG) is None:
        loi.append(f"\"doi_tuong_dem\"={d.get('doi_tuong_dem')!r} không hợp lệ. Giá trị hợp lệ: {' | '.join(DOI_TUONG)}.")
    if d.get("sap_xep") is not None and not re.match(r"^(giam|tang)", fold(d["sap_xep"])):
        loi.append("\"sap_xep\" phải là \"giam\" hoặc \"tang\".")
    if d.get("top_k") is not None:
        try:
            if int(_so(d["top_k"], "top_k")) < 1:
                loi.append("\"top_k\" phải >= 1.")
        except DacTaLoi as e:
            loi.append(str(e))
    ng = d.get("nguong")
    if ng is not None:
        if not isinstance(ng, dict):
            loi.append("\"nguong\" phải là dict, vd {\"chi_so\":\"so_don\",\"toi_thieu\":4}.")
        else:
            for k, v in ng.items():
                if k not in NGUONG_KEYS:
                    loi.append(f"nguong: khoá lạ \"{k}\". Hợp lệ: {', '.join(NGUONG_KEYS)}.")
                elif k == "chi_so":
                    if _enum(v, CHI_SO) is None:
                        loi.append(f"nguong.chi_so={v!r} không hợp lệ.")
                elif k != "nghiem_ngat":
                    try:
                        _so(v, f"nguong.{k}")
                    except DacTaLoi as e:
                        loi.append(str(e))
            if not any(k in ng for k in ("toi_thieu", "lon_hon", "toi_da", "nho_hon")):
                loi.append("nguong cần ít nhất một trong: toi_thieu | lon_hon | toi_da | nho_hon.")
    ky = d.get("ky")
    if ky is not None and not isinstance(ky, (str, list, dict)):
        loi.append("\"ky\" phải là chuỗi (cụm thời gian chép từ câu), list chuỗi, hoặc dict {tu, den}.")
    if isinstance(ky, list) and (not ky or not all(isinstance(x, str) for x in ky)):
        loi.append("\"ky\" dạng list phải gồm các chuỗi cụm thời gian.")
    # yêu cầu theo loai
    if loai in ("chi_so", "so_sanh", "ti_trong", "xep_hang", "theo_ky") and cs is None and d.get("chi_so") is None:
        loi.append(f"loai={loai} cần \"chi_so\".")
    if loai == "dem_nguong":
        if not isinstance(ng, dict) or "chi_so" not in ng:
            loi.append("loai=dem_nguong cần \"nguong\" có \"chi_so\" và \"toi_thieu\" (vd {\"chi_so\":\"so_don\",\"toi_thieu\":4}).")
        if d.get("doi_tuong_dem") is None:
            loi.append("loai=dem_nguong cần \"doi_tuong_dem\" (khach | don | ngay | mat_hang).")
    if loai == "xep_hang" and d.get("nhom_theo") is None:
        loi.append("loai=xep_hang cần \"nhom_theo\" (mat_hang | khach | nhan_vien | ngay | thang | kenh | nhom_hang | gio | nha_cung_cap ...).")
    if loai == "theo_ky" and _enum(d.get("nhom_theo"), THOI_GIAN) is None:
        loi.append("loai=theo_ky cần \"nhom_theo\" là một trong: ngay | thang | tuan | quy | nam.")
    if loai in ("so_sanh", "ti_trong", "theo_ky") and d.get("ky") is None:
        loi.append(f"loai={loai} cần \"ky\".")
    if loai in ("chi_so", "xep_hang", "dem_nguong") and d.get("ky") is None and cs not in ("con_no", "so_khach_no") and not (loai == "dem_nguong" and _enum((ng or {}).get("chi_so"), ("con_no",))):
        loi.append("Thiếu \"ky\" (chép nguyên văn cụm thời gian trong câu; chỉ con_no / so_khach_no được bỏ trống = đến hết hôm nay).")
    return loi


# ----------------------------------------------------------------------------- ngữ cảnh
_PTTT_VI = {"cash": "tiền mặt", "transfer": "chuyển khoản", "bank transfer": "chuyển khoản", "bank": "chuyển khoản",
            "ewallet": "ví điện tử", "e wallet": "ví điện tử", "wallet": "ví điện tử", "card": "thẻ",
            "credit card": "thẻ", "debt": "ghi nợ", "credit": "ghi nợ", "cod": "COD"}
_PAYC = [("tm", r"tien mat|\btm\b|cash"), ("ck", r"chuyen khoan|\bck\b|transfer|bank"),
         ("vi", r"vi dien tu|e ?wallet|momo|zalopay|vnpay"), ("cod", r"\bcod\b|thu ho"),
         ("no", r"ghi no|debt|cong no|mua chiu|\bno\b|tra sau|credit"), ("the", r"\bthe\b|card|pos|visa")]


def _payc(v):
    t = fold(v)
    for k, pat in _PAYC:
        if re.search(pat, t):
            return k
    return None


_BAN_COLS = ["ma_don", "ngay", "thoi_diem", "trang_thai", "tra_hang", "tinh_dt", "ma_hang", "ten_hang", "la_phi", "nhom_hang",
             "ke_don", "hoat_chat", "don_vi", "don_vi_co_ban", "so_luong", "so_luong_co_ban", "don_gia", "thanh_tien", "gia_von",
             "ma_khach", "ten_khach", "khach", "nhom_khach", "kenh", "nhan_vien", "ten_nhan_vien", "pttt"]


class _Ctx:
    def __init__(self, biz, rulebook, cal, as_of):
        self.biz, self.rb, self.cal = biz, rulebook or "", cal
        self.as_of = pd.Timestamp(str(as_of)[:10]) if as_of else pd.Timestamp.today().normalize()
        self.as_of_s = self.as_of.strftime("%Y-%m-%d")
        self.lines, self.mota = [], []
        self._c = {}
        self.ban = self._chuan_bi_ban()
        ng = pd.to_datetime(self.ban["ngay"], errors="coerce")
        self.dmin, self.dmax = ng.min().normalize(), ng.max().normalize()

    def _chuan_bi_ban(self):
        ban = getattr(self.biz, "ban", None)
        if ban is None or len(ban) == 0:
            raise DacTaLoi("Shop không có bảng bán chuẩn (biz.ban rỗng hoặc không dựng được): engine không tính được, hãy chuyển sang tự giải (tu_giai).")
        miss = [c for c in _BAN_COLS if c not in ban.columns]
        extra = {}
        for c in miss:
            extra[c] = False if c in ("tra_hang", "la_phi") else (True if c == "tinh_dt" else np.nan)
        if "pttt" in miss:
            od = getattr(self.biz, "no_don", None)
            if od is not None and "pttt" in od and od["pttt"].notna().any() and "ma_don" in ban:
                pt = od.drop_duplicates("ma_don").set_index("ma_don")["pttt"]
                pt = pt.map(lambda v: _PTTT_VI.get(re.sub(r"[_\-]+", " ", fold(v)).strip(), v) if pd.notna(v) else v)
                extra["pttt"] = ban["ma_don"].map(pt).values
        if extra:
            ban = ban.assign(**extra)
        if "ngay" not in ban or ban["ngay"].isna().all():
            raise DacTaLoi("Bảng bán chuẩn không có cột ngày: engine không tính được, hãy chuyển sang tự giải (tu_giai).")
        return ban

    def uniq(self, col, tab=None):
        key = ("u", col, id(tab))
        if key not in self._c:
            df = self.ban if tab is None else tab
            self._c[key] = [str(x) for x in pd.unique(df[col].dropna().astype(str))] if col in df else []
        return self._c[key]

    def ghi(self, s):
        if len(self.lines) < 15:
            self.lines.append(s)


# ----------------------------------------------------------------------------- khớp tên
def _sub(a, b):
    """a là dãy con LIÊN TIẾP của b."""
    n = len(a)
    return n > 0 and any(b[i:i + n] == a for i in range(len(b) - n + 1))


def _tim(text, cands, chinh_xac=False):
    """(danh sách ứng viên khớp, bậc). Bậc 0-1 = khớp đúng; 2 = chứa cụm; 3 = đủ từ; 4 = tên nằm trong chuỗi; 5 = gần giống."""
    q = str(text).strip()
    qn, qt = _nz(q), _toks(q)
    cs = [(c, _nz(c), _toks(c)) for c in cands]
    h = [c for c, n, t in cs if n == qn]
    if h:
        return h, 0
    h = [c for c, n, t in cs if t == qt and qt]
    if h:
        return h, 1
    if chinh_xac or not qt:
        return [], 9
    h = [c for c, n, t in cs if _sub(qt, t)]
    if h:
        return h, 2
    h = [c for c, n, t in cs if set(qt) <= set(t)]
    if h:
        return h, 3
    h = [c for c, n, t in cs if t and len("".join(t)) >= 4 and _sub(t, qt)]
    if h:
        return h, 4
    fq = {" ".join(t): c for c, n, t in cs if t}
    cm = difflib.get_close_matches(" ".join(qt), list(fq), n=3, cutoff=0.86)
    if cm:
        return [fq[x] for x in cm], 5
    return [], 9


def _goi_y(ctx, text, bo_qua=()):
    """Gợi ý: giá trị gần giống ở các cột khác / cùng cột."""
    out = []
    for col in ("ten_hang", "nhom_hang", "kenh", "nhom_khach", "ten_khach", "nhan_vien", "ten_nhan_vien", "pttt", "hoat_chat"):
        if col in bo_qua or col not in ctx.ban:
            continue
        h, b = _tim(text, ctx.uniq(col))
        if h and b <= 3:
            out.append(f"cột {col}: " + "; ".join(h[:3]))
    return out


def _chon(ctx, text, cands, nhan, col=None, chinh_xac=False, tab=None):
    """Chọn ĐÚNG MỘT giá trị (hoặc nhiều chỉ khác cách viết). Lỗi liệt kê ứng viên nếu không có / nhập nhằng."""
    cands = list(cands)
    h, b = _tim(text, cands, chinh_xac)
    if not h:
        cm = difflib.get_close_matches(fold(text), [fold(c) for c in cands], n=5, cutoff=0.4)
        gan = [c for c in cands if fold(c) in cm][:5]
        if not gan:
            toks = set(_toks(text))
            gan = [c for c in cands if toks & set(_toks(c))][:5]
        ms = f"Không tìm thấy {nhan} nào khớp '{text}' trong dữ liệu."
        if gan:
            ms += " Gần giống nhất: " + "; ".join(gan) + "."
        gy = _goi_y(ctx, text, bo_qua=(col,) if col else ())
        if gy:
            ms += " Giá trị này xuất hiện ở: " + " | ".join(gy[:3]) + " (có thể bạn đặt nhầm khoá loc)."
        ms += " Nếu thông tin này không có trong dữ liệu của shop thì câu có thể không trả lời được (loai=khong_du_du_lieu)."
        raise DacTaLoi(ms)
    if b >= 2 and len(h) > 1:
        raise DacTaLoi(f"Có {len(h)} {nhan} cùng khớp '{text}': " + "; ".join(h[:10]) + (f" ... (+{len(h) - 10})" if len(h) > 10 else "")
                       + ". Hãy chọn đúng tên đầy đủ (hoặc truyền list các tên nếu câu hỏi gộp nhiều mặt).")
    return h


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _co_du_lieu(ctx, col, nhan):
    if col not in ctx.ban or not ctx.ban[col].notna().any():
        raise KhongTraLoiDuoc(f"Dữ liệu của shop không ghi {nhan}.")


def _tim_trong_goc(biz, text):
    """Thuật ngữ có xuất hiện ở bảng gốc (ngoài bảng chuẩn) không -> (bảng, cột) | None."""
    try:
        for t, df in list(getattr(biz, "tables", {}).items())[:12]:
            for c in df.columns:
                if c == "_nguon" or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
                    continue
                u = df[c].dropna().astype(str).unique()
                if len(u) > 3000:
                    continue
                h, b = _tim(text, list(u))
                if h and b <= 3:
                    return t, c
    except Exception:
        return None
    return None


def _khong_cot(ctx, col, nhan, text):
    """Cột trống trong bảng chuẩn: unanswerable nếu bảng gốc cũng không có thuật ngữ; còn lại báo lỗi cho model."""
    r = _tim_trong_goc(ctx.biz, text)
    if r:
        raise DacTaLoi(f"Bảng chuẩn không có dữ liệu cột '{col}' ({nhan}), nhưng '{text}' xuất hiện ở bảng gốc '{r[0]}' cột '{r[1]}'. "
                       f"Nếu cột đó thật sự là {nhan} thì engine chưa đọc được -> chuyển sang tự giải (tu_giai); "
                       f"nếu '{text}' chỉ là giá trị của cột khác (vd tên khách) thì shop không ghi {nhan} -> dùng loai khong_du_du_lieu.")
    raise KhongTraLoiDuoc(f"Dữ liệu của shop không ghi {nhan} (cột {col} trống).")


# ----------------------------------------------------------------------------- bộ lọc loc -> mask trên bảng bán
def _key(d):
    return repr(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items()))


def _ca_gio(ctx, ca):
    k = fold(ca)
    k = {"sang": "sang", "chieu": "chieu", "toi": "toi", "dem": "toi"}.get(k.replace("ca ", "").strip())
    if k is None:
        raise DacTaLoi("loc.ca phải là sang | chieu | toi.")
    lo, hi = _HOUR_DEFAULT[k]
    rb = ctx.rb or ""
    m = re.search(r"ca\s+" + {"sang": "s[aá]ng", "chieu": "chi[eề]u", "toi": "t[oố]i"}[k] + r"[^0-9\n]{0,12}(\d{1,2})[:hH](\d{2})\s*[–\-—~]+\s*(\d{1,2})[:hH](\d{2})",
                  rb, re.I)
    if m:
        lo, hi = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
    return k, lo, hi


def _thu_so(v):
    t = fold(v).replace("thu ", "").strip()
    if t in ("cn", "chu nhat", "8"):
        return 6
    if t in ("t2", "2", "hai"):
        return 0
    m = re.fullmatch(r"t?([2-7])", t)
    if m:
        return int(m[1]) - 2
    names = {"hai": 0, "ba": 1, "tu": 2, "nam": 3, "sau": 4, "bay": 5}
    if t in names:
        return names[t]
    raise DacTaLoi("loc.thu phải là 2..7 (thứ hai..thứ bảy) hoặc CN.")


def _mask_ban(ctx, loc):
    """Series bool trên ctx.ban theo loc (None nếu loc không lọc gì)."""
    ban = ctx.ban
    mask = None

    def an(m):
        nonlocal mask
        mask = m if mask is None else (mask & m)

    for k, v in (loc or {}).items():
        if v is None or v == "" or v == []:
            continue
        if k in ("trang_thai", "tuoi_no_hon", "tuoi_no_tu", "tuoi_no_den", "nha_cung_cap"):
            continue
        vals = _as_list(v)
        if k == "mat_hang":
            names, codes = set(), set()
            for t in vals:
                done = False
                if "ma_hang" in ban and ban["ma_hang"].notna().any():
                    h, b = _tim(t, ctx.uniq("ma_hang"), chinh_xac=True)
                    if h:
                        codes |= set(h)
                        done = True
                if not done:
                    names |= set(_chon(ctx, t, ctx.uniq("ten_hang"), "mặt hàng", "ten_hang"))
            m = ban["ten_hang"].astype(str).isin(names) if names else pd.Series(False, index=ban.index)
            if codes:
                m = m | ban["ma_hang"].astype(str).isin(codes)
            an(m)
        elif k == "nhom_hang":
            _co_du_lieu_loc(ctx, "nhom_hang", "nhóm hàng", vals[0])
            hs = set()
            for t in vals:
                hs |= set(_chon(ctx, t, ctx.uniq("nhom_hang"), "nhóm hàng", "nhom_hang"))
            an(ban["nhom_hang"].astype(str).isin(hs))
        elif k == "kenh":
            _co_du_lieu_loc(ctx, "kenh", "kênh bán", vals[0])
            hs = set()
            for t in vals:
                hs |= set(_chon(ctx, t, ctx.uniq("kenh"), "kênh bán", "kenh"))
            an(ban["kenh"].astype(str).isin(hs))
        elif k == "nhom_khach":
            _co_du_lieu_loc(ctx, "nhom_khach", "nhóm khách", vals[0])
            cands = [c for c in ctx.uniq("nhom_khach") if not c.strip().startswith("(")]
            hs = set()
            for t in vals:
                hs |= set(_chon(ctx, t, cands, "nhóm khách", "nhom_khach"))
            an(ban["nhom_khach"].astype(str).isin(hs))
        elif k == "thanh_toan":
            _co_du_lieu_loc(ctx, "pttt", "hình thức thanh toán", vals[0])
            raw = ctx.uniq("pttt")
            hs = set()
            for t in vals:
                c = _payc(t)
                if c is not None:
                    sel = [r for r in raw if _payc(r) == c]
                    if sel:
                        hs |= set(sel)
                        continue
                hs |= set(_chon(ctx, t, raw, "hình thức thanh toán", "pttt"))
            an(ban["pttt"].astype(str).isin(hs))
        elif k == "nhan_vien":
            if not (ban["nhan_vien"].notna().any() or ban["ten_nhan_vien"].notna().any()):
                _khong_cot(ctx, "nhan_vien", "nhân viên bán", str(vals[0]))
            m = pd.Series(False, index=ban.index)
            for t in vals:
                got = False
                for col in ("nhan_vien", "ten_nhan_vien"):
                    if ban[col].notna().any():
                        h, b = _tim(t, ctx.uniq(col))
                        if h and (b <= 1 or len(h) == 1):
                            m |= ban[col].astype(str).isin(h)
                            got = True
                            break
                        if h and b >= 2:
                            raise DacTaLoi(f"Có nhiều nhân viên cùng khớp '{t}': " + "; ".join(h[:8]) + ". Hãy chọn đúng.")
                if not got:
                    raise DacTaLoi(f"Không tìm thấy nhân viên '{t}'. Có: " + "; ".join((ctx.uniq("nhan_vien") or ctx.uniq("ten_nhan_vien"))[:10]) + ".")
            an(m)
        elif k == "khach":
            if not ban["khach"].notna().any():
                raise KhongTraLoiDuoc("Dữ liệu của shop không ghi khách hàng.")
            m = pd.Series(False, index=ban.index)
            for t in vals:
                hits = set()
                for col in ("ten_khach", "ma_khach"):
                    if col in ban and ban[col].notna().any():
                        h, b = _tim(t, ctx.uniq(col))
                        if h and b <= 4:
                            hits |= {(col, x) for x in h}
                            if b <= 1:
                                break
                if not hits:
                    cm = difflib.get_close_matches(fold(t), [fold(c) for c in ctx.uniq("ten_khach")], n=5, cutoff=0.5)
                    gan = [c for c in ctx.uniq("ten_khach") if fold(c) in cm][:5]
                    raise DacTaLoi(f"Không tìm thấy khách '{t}'." + (" Gần giống: " + "; ".join(gan) + "." if gan else "")
                                   + " Nếu khách không có trong dữ liệu, câu có thể không trả lời được.")
                mm = pd.Series(False, index=ban.index)
                for col, x in hits:
                    mm |= ban[col].astype(str).eq(x)
                keys = ban.loc[mm & ban["khach"].notna(), "khach"].astype(str).unique()
                tens = ban.loc[mm, "ten_khach"].astype(str).unique() if "ten_khach" in ban else []
                if len(keys) > 1:
                    raise DacTaLoi(f"Có {len(keys)} khách khác nhau (mã {', '.join(map(str, keys[:6]))}) cùng khớp '{t}'. Hãy nêu đúng tên đầy đủ hoặc mã khách.")
                m |= mm
            an(m)
        elif k == "hoat_chat":
            _co_du_lieu_loc(ctx, "hoat_chat", "hoạt chất", vals[0])
            hs = set()
            for t in vals:
                hs |= set(_chon(ctx, t, ctx.uniq("hoat_chat"), "hoạt chất", "hoat_chat", chinh_xac=True))
            an(ban["hoat_chat"].astype(str).isin(hs))
        elif k == "ke_don":
            _co_du_lieu_loc(ctx, "ke_don", "thuốc kê đơn", "kê đơn")
            b = bool(v) if not isinstance(v, str) else fold(v) in ("true", "co", "1", "yes", "ke don")
            an(ban["ke_don"].fillna(False).astype(bool) == b)
        elif k == "don_vi":
            col = "don_vi_co_ban" if ban["don_vi_co_ban"].notna().any() else "don_vi"
            fu = fold(vals[0])
            m = ban[col].map(lambda x: fold(x) == fu if pd.notna(x) else False).astype(bool)
            if not m.any():
                raise DacTaLoi(f"Không có mặt hàng nào có đơn vị cơ bản là '{vals[0]}'. Đơn vị cơ bản có trong dữ liệu: " + "; ".join(ctx.uniq(col)[:15]) + ".")
            an(m)
        elif k == "size":
            s = fold(vals[0]).replace("size", "").strip().upper()
            if s not in ("M", "L", "S", "XL"):
                raise DacTaLoi("loc.size phải là M hoặc L.")
            an(ban["ten_hang"].astype(str).str.strip().str.upper().str.endswith(f"({s})"))
        elif k == "thu":
            wd = _thu_so(vals[0])
            an(pd.to_datetime(ban["ngay"]).dt.dayofweek == wd)
        elif k == "ca":
            if ban["thoi_diem"].isna().all():
                raise KhongTraLoiDuoc("Dữ liệu của shop không ghi giờ tạo đơn nên không chia được theo ca.")
            kk, lo, hi = _ca_gio(ctx, vals[0])
            td = pd.to_datetime(ban["thoi_diem"], errors="coerce")
            mins = td.dt.hour * 60 + td.dt.minute
            an((mins >= lo) & (mins <= hi))
    # mat_hang + don_vi: đơn vị cơ bản của mặt hàng phải khớp
    if loc and loc.get("mat_hang") and loc.get("don_vi") and mask is not None:
        sub = ban[mask]
        bu = {fold(x) for x in sub["don_vi_co_ban"].dropna().unique()} if sub["don_vi_co_ban"].notna().any() else set()
        if len(sub) == 0 or (bu and fold(loc["don_vi"]) not in bu):
            raise DacTaLoi(f"Mặt hàng {loc['mat_hang']!r} có đơn vị cơ bản là {', '.join(sorted(bu)) or '(không rõ)'}, không phải '{loc['don_vi']}'. "
                           "so_luong luôn tính theo đơn vị cơ bản: hãy bỏ khoá don_vi.")
    return mask


def _co_du_lieu_loc(ctx, col, nhan, text):
    if col not in ctx.ban or not ctx.ban[col].notna().any():
        _khong_cot(ctx, col, nhan, str(text))


def _ban_loc(ctx, loc):
    """Bảng bán đã lọc theo loc (cache)."""
    k = ("ban_loc", _key({a: b for a, b in (loc or {}).items() if a != "trang_thai"}))
    if k not in ctx._c:
        m = _mask_ban(ctx, loc)
        ctx._c[k] = ctx.ban if m is None else ctx.ban[m]
    return ctx._c[k]


# ----------------------------------------------------------------------------- kỳ
def _ky_iso(text):
    t = text.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        d = date(int(m[1]), int(m[2]), int(m[3]))
        return Period(d, d, "day", f"ngày {d.isoformat()}")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", t)
    if m:
        return _month_p(int(m[1]), int(m[2]))
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:->|→|\.\.|đến|den|to|/)\s*(\d{4}-\d{2}-\d{2})", t)
    if m:
        a, b = date.fromisoformat(m[1]), date.fromisoformat(m[2])
        return Period(a, b, "range", f"từ {a.isoformat()} đến {b.isoformat()}")
    return None


def _ky_n_ngay(ctx, text):
    s = fold(text)
    md = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", s)
    mn = re.search(r"(\d+) ngay", s)
    if not (md and mn) or re.search(r"gan nhat|gan day|vua qua|\bqua\b", s):
        return None
    a = ctx.as_of
    try:
        dd, mm = int(md[1]), int(md[2])
        yy = int(md[3]) if md[3] else (a.year if date(a.year, mm, dd) <= a.date() else a.year - 1)
        d0 = date(yy, mm, dd)
    except ValueError:
        return None
    n = int(mn[1])
    if re.search(r"(ngay )?truoc (ngay )?\d", s) and not re.search(r"(ke|tinh) tu", s):
        return Period(d0 - timedelta(days=n), d0 - timedelta(days=1), "range", f"{n} ngày ngay trước {d0:%d/%m/%Y}")
    if re.search(r"(ke|tinh) tu|tu ngay", s):
        return Period(d0, d0 + timedelta(days=n - 1), "range", f"{n} ngày kể từ {d0:%d/%m/%Y} (tính cả ngày đó)")
    return None


def _periods(ctx, ky):
    """ky (str | dict) -> list[Period]."""
    if isinstance(ky, dict):
        try:
            a, b = date.fromisoformat(str(ky.get("tu"))[:10]), date.fromisoformat(str(ky.get("den"))[:10])
        except Exception:
            raise DacTaLoi("ky dạng dict phải là {\"tu\":\"YYYY-MM-DD\",\"den\":\"YYYY-MM-DD\"}.")
        return [Period(a, b, "range", f"từ {a.isoformat()} đến {b.isoformat()}")]
    k = ("ky", ky)
    if k in ctx._c:
        return ctx._c[k]
    ps = None
    p = _ky_iso(ky) or _ky_n_ngay(ctx, ky)
    if p is not None:
        ps = [p]
    if ps is None:
        try:
            ps = find_periods(ky, ctx.as_of_s, ctx.cal)
        except Exception as e:
            raise DacTaLoi(f"Không hiểu cụm thời gian {ky!r} ({e}).")
    if not ps:
        raise DacTaLoi(f"Không nhận ra kỳ thời gian trong {ky!r}. Hãy chép NGUYÊN VĂN cụm thời gian trong câu hỏi, vd 'tháng 6/2025', "
                       "'tuần trước', 'từ 01/03/2026 đến 07/03/2026', 'quý 2/2026', 'hôm qua', 'Tết 2026'.")
    ctx._c[k] = ps
    return ps


def _ky_so(ctx, p, ks):
    if ks is None:
        c = p.meta.get("compare") if p.meta else None
        if c is None:
            c = previous(p, ctx.cal)
            ctx.mota.append("Kỳ so sánh mặc định = kỳ trước (periods.previous)")
        return c
    t = _enum(ks, ("ky_truoc", "cung_ky_nam_truoc", "cung_ky_thang_truoc")) if isinstance(ks, str) else None
    if t == "ky_truoc":
        return (p.meta.get("compare") if p.meta else None) or previous(p, ctx.cal)
    if t == "cung_ky_nam_truoc":
        return same_period_last_year(p, ctx.cal)
    if t == "cung_ky_thang_truoc":
        return previous(p, ctx.cal)
    ps = _periods(ctx, ks)
    return ps[0]


def _mota_ky(p):
    s = p.start.isoformat() if p.start else "đầu dữ liệu"
    return f"{p.label} ({s} → {p.end.isoformat()})"


def _ngoai(ctx, p, lo=None, hi=None):
    lo = ctx.dmin if lo is None else lo
    hi = ctx.dmax if hi is None else hi
    st = pd.Timestamp(p.start) if p.start is not None else lo
    en = pd.Timestamp(p.end)
    if en < lo or st > hi:
        raise KhongTraLoiDuoc(f"Kỳ {_mota_ky(p)} nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop ({lo:%Y-%m-%d} → {hi:%Y-%m-%d}).")


def _mk_ky(ngay, p):
    m = pd.Series(True, index=ngay.index)
    if p.start is not None:
        m &= ngay >= pd.Timestamp(p.start)
    return m & (ngay <= pd.Timestamp(p.end))


# ----------------------------------------------------------------------------- số đo: bảng bán
def _dem_don(x):
    return int(x["ma_don"].nunique()) if x["ma_don"].notna().any() else int(len(x))


def _qcol(d):
    return "so_luong_co_ban" if d["so_luong_co_ban"].notna().any() else "so_luong"


def _khong_phi(d):
    return d[~d["la_phi"].fillna(False).astype(bool)]


def _khung_ban(ctx, p, loc):
    """(d_all, d) = dòng bán trong kỳ & loc (mọi trạng thái; chỉ đơn tính doanh thu)."""
    _ngoai(ctx, p)
    sub = _ban_loc(ctx, loc)
    d_all = sub[_mk_ky(sub["ngay"], p)]
    d = d_all[d_all["tinh_dt"].fillna(False).astype(bool)]
    return d_all, d


def _trang_thai_x(d_all, st):
    t = fold(st).replace(" ", "_")
    if t in ("huy",):
        return d_all[d_all["trang_thai"] == "huỷ"]
    if t in ("tra_hang", "tra"):
        return d_all[d_all["tra_hang"].fillna(False).astype(bool)]
    if t in ("dang_giao", "giao"):
        return d_all[d_all["trang_thai"] == "đang giao"]
    if t in ("hoan_thanh", "xong"):
        return d_all[d_all["trang_thai"] == "hoàn thành"]
    raise DacTaLoi("loc.trang_thai phải là huy | tra_hang | dang_giao | hoan_thanh.")


def _v_ban(ctx, cs, p, loc):
    loc = dict(loc or {})
    st = loc.pop("trang_thai", None)
    for k in ("tuoi_no_hon", "tuoi_no_tu", "tuoi_no_den", "nha_cung_cap"):
        if loc.get(k) not in (None, ""):
            raise DacTaLoi(f"loc.{k} không áp dụng cho chi_so {cs} (dữ liệu bán).")
    if cs == "gia_ban":
        return _gia_ban(ctx, p, loc)
    if cs == "so_mat_hang_e":
        return _mat_hang_e(ctx, p, loc)
    d_all, d = _khung_ban(ctx, p, loc)
    if cs == "doanh_thu":
        return _tien(d["thanh_tien"].sum())
    if cs == "so_don":
        return _dem_don(_trang_thai_x(d_all, st) if st else d)
    if cs == "so_don_huy":
        return _dem_don(d_all[d_all["trang_thai"] == "huỷ"])
    if cs == "so_don_tra_hang":
        return _dem_don(d_all[d_all["tra_hang"].fillna(False).astype(bool)])
    if cs == "ti_le_huy":
        n = _dem_don(d_all)
        if n == 0:
            raise DacTaLoi(f"Không có đơn nào đặt trong kỳ {_mota_ky(p)} (với bộ lọc này) nên không tính được tỉ lệ huỷ.")
        return _pct2(_dem_don(d_all[d_all["trang_thai"] == "huỷ"]), n)
    if cs == "so_luong":
        x = _khong_phi(d)
        col = _qcol(x)
        return _sl(x[col].sum())
    if cs == "gia_tri_tb_don":
        n = _dem_don(d)
        if n == 0:
            raise DacTaLoi(f"Không có đơn tính doanh thu trong kỳ {_mota_ky(p)} (với bộ lọc này) nên không tính được giá trị trung bình.")
        return _chia_tien(d["thanh_tien"].sum(), n)
    if cs == "so_khach":
        _co_du_lieu(ctx, "khach", "khách hàng")
        return int(d["khach"].dropna().nunique())
    if cs == "lai_gop":
        _co_du_lieu(ctx, "gia_von", "giá vốn trên dòng bán")
        if d["gia_von"].isna().any():
            raise DacTaLoi(f"{int(d['gia_von'].isna().sum())} dòng bán trong kỳ không có giá vốn: engine không tính lãi gộp được, hãy dùng tu_giai.")
        return _tien((d["thanh_tien"] - d["so_luong"] * d["gia_von"]).sum())
    if cs in ("so_ngay_ban", "so_ngay_mua"):
        return int(d["ngay"].nunique())
    if cs == "so_mat_hang":
        x = _khong_phi(d)
        key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if x["ma_hang"].notna().any() else x["ten_hang"]
        return int(key.dropna().nunique())
    raise DacTaLoi(f"chi_so {cs} không áp dụng cho dữ liệu bán.")


def _gia_ban(ctx, p, loc):
    if not loc.get("mat_hang"):
        raise DacTaLoi("chi_so gia_ban cần loc.mat_hang (và ky là MỘT ngày).")
    if p.start is None or p.start != p.end:
        raise DacTaLoi("chi_so gia_ban cần ky là MỘT ngày cụ thể (vd 'ngày 12/06/2025').")
    _ngoai(ctx, p)
    sub = _ban_loc(ctx, loc)
    r = sub[_mk_ky(sub["ngay"], p)]
    if len(r) == 0:
        raise KhongTraLoiDuoc(f"Không có dòng bán nào của {loc['mat_hang']} vào ngày {p.end.isoformat()}: không có giá bán ghi trong dữ liệu.")
    qb = r["so_luong_co_ban"] if r["so_luong_co_ban"].notna().any() else r["so_luong"]
    g = (r["don_gia"] * r["so_luong"] / qb).round().dropna()
    if len(g) == 0:
        raise DacTaLoi("Dòng bán không có đơn giá.")
    m = g.mode()
    if len(m) > 1:
        ctx.ghi(f"(lưu ý: có nhiều mức giá khác nhau trong ngày: {sorted(g.unique())[:5]}; lấy mức xuất hiện nhiều nhất)")
    return _tien(m.iloc[0])


def _mat_hang_e(ctx, p, loc):
    if p.start is None:
        raise DacTaLoi("so_mat_hang_e cần kỳ có ngày bắt đầu.")
    _ngoai(ctx, p)
    sub = _ban_loc(ctx, loc)
    x = _khong_phi(sub[sub["tinh_dt"].fillna(False).astype(bool)])
    key = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if x["ma_hang"].notna().any() else x["ten_hang"]
    x = x.assign(_k=key)
    truoc = set(x.loc[x["ngay"] < pd.Timestamp(p.start), "_k"].dropna())
    trong = set(x.loc[_mk_ky(x["ngay"], p), "_k"].dropna())
    return len(truoc - trong)


# ----------------------------------------------------------------------------- số đo: nhập hàng
def _co_cot_goc(biz, regex):
    try:
        for t, df in getattr(biz, "tables", {}).items():
            if re.search(regex, fold(t)) or any(re.search(regex, fold(c).replace("_", " ")) for c in df.columns):
                return True
    except Exception:
        return False
    return False


def _khung_nhap(ctx, p, loc):
    N = getattr(ctx.biz, "nhap", None)
    if N is None or len(N) == 0:
        if _co_cot_goc(ctx.biz, r"nhap"):
            raise DacTaLoi("Shop có dữ liệu nhập hàng nhưng engine không dựng được bảng nhập chuẩn (biz.nhap): hãy chuyển sang tự giải (tu_giai).")
        raise KhongTraLoiDuoc("Dữ liệu của shop không ghi nhập hàng.")
    bad = [k for k, v in (loc or {}).items() if v not in (None, "", []) and k not in ("mat_hang", "nha_cung_cap")]
    if bad:
        raise DacTaLoi(f"Bộ lọc {', '.join(bad)} không áp dụng cho dữ liệu nhập hàng (chỉ có mat_hang, nha_cung_cap).")
    ng = pd.to_datetime(N["ngay"], errors="coerce")
    _ngoai(ctx, p, ng.min().normalize(), ng.max().normalize())
    x = N[N["tinh"].fillna(True).astype(bool) & _mk_ky(ng, p)]
    loc = loc or {}
    if loc.get("mat_hang"):
        names = set()
        for t in _as_list(loc["mat_hang"]):
            names |= set(_chon(ctx, t, ctx.uniq("ten_hang", N), "mặt hàng (trong phiếu nhập)", "ten_hang", tab=N))
        x = x[x["ten_hang"].astype(str).isin(names)]
    if loc.get("nha_cung_cap"):
        if N["ncc"].notna().sum() == 0:
            raise KhongTraLoiDuoc("Dữ liệu nhập của shop không ghi nhà cung cấp.")
        names = set()
        for t in _as_list(loc["nha_cung_cap"]):
            names |= set(_chon(ctx, t, ctx.uniq("ncc", N), "nhà cung cấp", "ncc", tab=N))
        x = x[x["ncc"].astype(str).isin(names)]
    return x


def _v_nhap(ctx, cs, p, loc):
    x = _khung_nhap(ctx, p, loc)
    loc = loc or {}
    if cs == "tien_nhap":
        if x["thanh_tien"].isna().all() and ("tong_phieu" not in x or x["tong_phieu"].isna().all()):
            raise KhongTraLoiDuoc("Dữ liệu nhập của shop không ghi tiền nhập.")
        if not loc.get("mat_hang") and "tong_phieu" in x and x["tong_phieu"].notna().any() and x.drop_duplicates("ma_phieu")["tong_phieu"].notna().all():
            return _tien(x.drop_duplicates("ma_phieu")["tong_phieu"].sum())
        return _tien(x["thanh_tien"].sum())
    if cs == "so_phieu_nhap":
        return int(x["ma_phieu"].nunique())
    if cs == "gia_nhap_tb":
        if not loc.get("mat_hang"):
            raise DacTaLoi("gia_nhap_tb cần loc.mat_hang.")
        if x["thanh_tien"].isna().all() or x["so_luong"].fillna(0).sum() == 0:
            raise KhongTraLoiDuoc("Không có dữ liệu giá/số lượng nhập của mặt hàng này trong kỳ.")
        dv = x["don_vi"].dropna().unique()
        if len(dv) > 1:
            ctx.ghi(f"(lưu ý: mặt hàng nhập bằng nhiều đơn vị: {', '.join(map(str, dv[:4]))}; giá bình quân theo đơn vị nhập)")
        return _chia_tien(x["thanh_tien"].sum(), x["so_luong"].sum())
    raise DacTaLoi(f"chi_so {cs} không áp dụng cho dữ liệu nhập.")


# ----------------------------------------------------------------------------- số đo: giao hàng
def _v_giao(ctx, cs, p, loc):
    G = getattr(ctx.biz, "giao", None)
    if G is None or len(G) == 0:
        raise KhongTraLoiDuoc("Dữ liệu của shop không ghi giao hàng nhiều đợt.")
    if any(v not in (None, "", []) for v in (loc or {}).values()):
        raise DacTaLoi("so_dot_giao không hỗ trợ bộ lọc loc.")
    ng = pd.to_datetime(G["ngay_giao"], errors="coerce")
    _ngoai(ctx, p, ng.min().normalize(), ng.max().normalize())
    g = G[_mk_ky(ng.dt.normalize(), p)]
    return int(len(g.drop_duplicates(["ma_don", "dot"])))


# ----------------------------------------------------------------------------- số đo: công nợ
def _khung_no(ctx):
    k = ("no",)
    if k in ctx._c:
        return ctx._c[k]
    O, T = getattr(ctx.biz, "no_don", None), getattr(ctx.biz, "thu_tien", None)
    if O is None:
        if _co_cot_goc(ctx.biz, r"\bno\b|cong no|ghi no|con thieu|da tra|tien thu|thanh toan"):
            raise DacTaLoi("Shop có cột công nợ / thanh toán nhưng engine không dựng được bảng công nợ chuẩn (biz.no_don): hãy chuyển sang tự giải (tu_giai).")
        raise KhongTraLoiDuoc("Dữ liệu của shop không ghi công nợ.")
    if T is None:
        raise DacTaLoi("Shop có bảng công nợ theo đơn nhưng không có bảng các lần trả tiền (biz.thu_tien): hãy chuyển sang tự giải (tu_giai).")
    O2 = O[O["ghi_no"].fillna(False).astype(bool) & ~O["huy"].fillna(False).astype(bool)].copy()
    ban = ctx.ban
    kd = ban.dropna(subset=["ma_don"]).drop_duplicates("ma_don").set_index("ma_don")
    O2["_khach"] = O2["ma_don"].map(kd["khach"]) if "khach" in kd else np.nan
    O2["_nhom_khach"] = O2["ma_don"].map(kd["nhom_khach"]) if "nhom_khach" in kd else np.nan
    if O2["_khach"].isna().all():
        mk = O2["ma_khach"] if "ma_khach" in O2 else pd.Series(np.nan, index=O2.index)
        O2["_khach"] = mk
    O2["_ten"] = O2["ten_khach"] if "ten_khach" in O2 else np.nan
    O2["ngay"] = pd.to_datetime(O2["ngay"], errors="coerce").dt.normalize()
    T2 = T.copy()
    T2["ngay_tra"] = pd.to_datetime(T2["ngay_tra"], errors="coerce").dt.normalize()
    T2["ngay_don"] = pd.to_datetime(T2["ngay_don"], errors="coerce").dt.normalize()
    ctx._c[k] = (O2, T2)
    return ctx._c[k]


def _loc_no(ctx, O2, loc):
    for k, v in (loc or {}).items():
        if v in (None, "", []):
            continue
        if k in ("tuoi_no_hon", "tuoi_no_tu", "tuoi_no_den"):
            continue
        if k == "khach":
            m = pd.Series(False, index=O2.index)
            for t in _as_list(v):
                hits = set()
                for col in ("ten_khach", "ma_khach"):
                    if col in O2 and O2[col].notna().any():
                        h, b = _tim(t, [str(x) for x in O2[col].dropna().unique()])
                        if h and b <= 4:
                            hits |= {(col, x) for x in h}
                            if b <= 1:
                                break
                if not hits:
                    raise DacTaLoi(f"Không tìm thấy khách '{t}' trong sổ công nợ.")
                mm = pd.Series(False, index=O2.index)
                for col, x in hits:
                    mm |= O2[col].astype(str).eq(x)
                if O2.loc[mm, "_khach"].dropna().astype(str).nunique() > 1:
                    raise DacTaLoi(f"Nhiều khách cùng khớp '{t}': mã " + ", ".join(O2.loc[mm, "_khach"].dropna().astype(str).unique()[:6]) + ". Hãy nêu tên đầy đủ hoặc mã.")
                m |= mm
            O2 = O2[m]
        elif k == "kenh":
            if "kenh" not in O2 or O2["kenh"].notna().sum() == 0:
                raise KhongTraLoiDuoc("Dữ liệu của shop không ghi kênh bán.")
            hs = set()
            for t in _as_list(v):
                hs |= set(_chon(ctx, t, [str(x) for x in O2["kenh"].dropna().unique()], "kênh bán", "kenh"))
            O2 = O2[O2["kenh"].astype(str).isin(hs)]
        elif k == "nhom_khach":
            cands = [str(x) for x in O2["_nhom_khach"].dropna().unique() if not str(x).strip().startswith("(")]
            if not cands:
                raise KhongTraLoiDuoc("Dữ liệu của shop không ghi nhóm khách.")
            hs = set()
            for t in _as_list(v):
                hs |= set(_chon(ctx, t, cands, "nhóm khách", "nhom_khach"))
            O2 = O2[O2["_nhom_khach"].astype(str).isin(hs)]
        elif k == "thanh_toan":
            O2 = O2   # đơn ghi nợ: hình thức đã là ghi nợ
        else:
            raise DacTaLoi(f"Bộ lọc {k} không áp dụng cho công nợ (chỉ khach, kenh, nhom_khach, tuoi_no_*).")
    return O2


def _no_con(ctx, p, loc):
    """DataFrame các đơn còn nợ > 0 đến hết ngày p.end (cột 'no', 'tuoi')."""
    O2, T2 = _khung_no(ctx)
    X = pd.Timestamp(p.end)
    _ngoai(ctx, p)
    o = _loc_no(ctx, O2, loc)
    o = o[o["ngay"] <= X].copy()
    t = T2[T2["ma_don"].isin(set(o["ma_don"])) & (T2["ngay_tra"] <= X)].groupby("ma_don")["so_tien"].sum()
    o["no"] = o["tong_don"] - o["ma_don"].map(t).fillna(0)
    o["tuoi"] = (X - o["ngay"]).dt.days
    loc = loc or {}
    if loc.get("tuoi_no_hon") not in (None, ""):
        o = o[o["tuoi"] > _so(loc["tuoi_no_hon"], "tuoi_no_hon")]
    if loc.get("tuoi_no_tu") not in (None, ""):
        o = o[o["tuoi"] >= _so(loc["tuoi_no_tu"], "tuoi_no_tu")]
    if loc.get("tuoi_no_den") not in (None, ""):
        o = o[o["tuoi"] <= _so(loc["tuoi_no_den"], "tuoi_no_den")]
    return o[o["no"] > 0]


def _v_no(ctx, cs, p, loc):
    if cs in ("con_no", "so_khach_no"):
        o = _no_con(ctx, p, loc)
        if cs == "con_no":
            return _tien(o["no"].sum())
        return int(o["_khach"].dropna().astype(str).nunique())
    if p.start is None:
        raise DacTaLoi(f"{cs} cần kỳ có ngày bắt đầu (vd 'tháng 6/2025'), không phải mốc 'đến hết ngày X'.")
    _ngoai(ctx, p)
    O2, T2 = _khung_no(ctx)
    o = _loc_no(ctx, O2, loc)
    ids = set(o["ma_don"])
    t = T2[T2["ma_don"].isin(ids)]
    st, X = pd.Timestamp(p.start), pd.Timestamp(p.end)
    if cs == "thu_no":
        m = t["sau_ngay_mua"].fillna(False).astype(bool) & (t["ngay_tra"] >= st) & (t["ngay_tra"] <= X)
        return _tien(t.loc[m, "so_tien"].sum())
    if cs == "no_them":
        oo = o[(o["ngay"] >= st) & (o["ngay"] <= X)]
        same = t[t["ngay_tra"] == t["ngay_don"]].groupby("ma_don")["so_tien"].sum()
        return _tien((oo["tong_don"] - oo["ma_don"].map(same).fillna(0)).clip(lower=0).sum())
    raise DacTaLoi(f"chi_so {cs} không áp dụng cho công nợ.")


# ----------------------------------------------------------------------------- điều phối số đo
def _cs_nhom(cs):
    return ("ban" if cs in CHI_SO_BAN else "nhap" if cs in CHI_SO_NHAP else "no" if cs in CHI_SO_NO else "giao")


def _gia_tri(ctx, cs, p, loc):
    """Giá trị của chi_so trên kỳ p, bộ lọc loc (số)."""
    g = _cs_nhom(cs)
    v = (_v_ban if g == "ban" else _v_nhap if g == "nhap" else _v_no if g == "no" else _v_giao)(ctx, cs, p, loc)
    return v


# ----------------------------------------------------------------------------- định dạng số đo
_TIEN = ("doanh_thu", "tien_nhap", "lai_gop", "con_no", "thu_no", "no_them", "gia_tri_tb_don", "gia_ban", "gia_nhap_tb")


def _fmt(cs, v):
    if cs in _TIEN:
        return _tien(v)
    if cs == "so_luong":
        return _sl(v)
    if cs == "ti_le_huy":
        return round(float(v), 2)
    return int(round(float(v)))


# ----------------------------------------------------------------------------- nhóm
_TEN_COT = {"khach": "khach_hang", "mat_hang": "mat_hang", "nhan_vien": "nhan_vien", "kenh": "kenh", "nhom_hang": "nhom_hang",
            "nha_cung_cap": "nha_cung_cap", "ngay": "ngay", "thang": "thang", "gio": "gio", "nhom_khach": "nhom_khach",
            "thanh_toan": "thanh_toan", "don": "ma_don", "tuan": "tuan", "quy": "quy", "nam": "nam", "thu": "thu"}
_TEN_SO = {"con_no": "so_tien_no"}


def _gk(ctx, d_all, nhom):
    """DataFrame (cùng index d_all): _k khoá nhóm, _t tên hiển thị, _c mã để phá hoà, _ok dòng hợp lệ."""
    g = pd.DataFrame(index=d_all.index)
    if nhom == "mat_hang":
        la = d_all["la_phi"].fillna(False).astype(bool)
        k = d_all["ma_hang"].where(d_all["ma_hang"].notna(), d_all["ten_hang"]) if d_all["ma_hang"].notna().any() else d_all["ten_hang"]
        g["_k"], g["_t"], g["_ok"] = k.astype(str), d_all["ten_hang"].astype(str), (~la) & k.notna()
        g["_c"] = g["_k"]
    elif nhom == "khach":
        _co_du_lieu(ctx, "khach", "khách hàng")
        k = d_all["khach"]
        g["_k"] = k.astype(str)
        g["_t"] = d_all["ten_khach"].where(d_all["ten_khach"].notna(), k).astype(str)
        g["_ok"], g["_c"] = k.notna(), g["_k"]
    elif nhom == "nhan_vien":
        if not (d_all["nhan_vien"].notna().any() or d_all["ten_nhan_vien"].notna().any()):
            raise KhongTraLoiDuoc("Dữ liệu của shop không ghi nhân viên bán.")
        k = d_all["nhan_vien"] if d_all["nhan_vien"].notna().any() else d_all["ten_nhan_vien"]
        g["_k"], g["_t"], g["_ok"], g["_c"] = k.astype(str), k.astype(str), k.notna(), k.astype(str)
    elif nhom in ("kenh", "nhom_hang", "nhom_khach", "thanh_toan"):
        col = {"thanh_toan": "pttt"}.get(nhom, nhom)
        _co_du_lieu(ctx, col, {"kenh": "kênh bán", "nhom_hang": "nhóm hàng", "nhom_khach": "nhóm khách", "thanh_toan": "hình thức thanh toán"}[nhom])
        k = d_all[col]
        ok = k.notna()
        if nhom == "nhom_khach":
            ok &= ~k.astype(str).str.strip().str.startswith("(")
        g["_k"], g["_t"], g["_ok"], g["_c"] = k.astype(str), k.astype(str), ok, k.astype(str)
    elif nhom in THOI_GIAN or nhom in ("gio", "thu"):
        ng = pd.to_datetime(d_all["ngay"])
        if nhom == "ngay":
            s = ng.dt.strftime("%Y-%m-%d")
        elif nhom == "thang":
            s = ng.dt.strftime("%Y-%m")
        elif nhom == "tuan":
            s = (ng - pd.to_timedelta(ng.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
        elif nhom == "quy":
            s = ng.dt.year.astype(str) + "-Q" + ((ng.dt.month - 1) // 3 + 1).astype(str)
        elif nhom == "nam":
            s = ng.dt.year.astype(str)
        elif nhom == "thu":
            s = ng.dt.dayofweek.astype(str)
        else:
            td = pd.to_datetime(d_all["thoi_diem"], errors="coerce")
            if td.isna().all():
                raise KhongTraLoiDuoc("Dữ liệu của shop không ghi giờ tạo đơn nên không nhóm theo giờ.")
            s = td.dt.hour.astype("Int64").astype(str).str.zfill(2)
            g["_ok"] = td.notna()
        g["_k"], g["_t"], g["_c"] = s, s, s
        if "_ok" not in g:
            g["_ok"] = ng.notna()
    elif nhom == "don":
        k = d_all["ma_don"]
        g["_k"], g["_t"], g["_c"], g["_ok"] = k.astype(str), k.astype(str), k.astype(str), k.notna()
    else:
        raise DacTaLoi(f"nhom_theo={nhom} không áp dụng cho dữ liệu bán.")
    return g


def _agg_ban(cs, da, g):
    ga = g.loc[da.index, "_k"]
    tt = da["tinh_dt"].fillna(False).astype(bool)
    d, gd = da[tt], ga[tt]
    if cs == "doanh_thu":
        return d["thanh_tien"].groupby(gd).sum()
    if cs == "so_don":
        return d["ma_don"].groupby(gd).nunique() if d["ma_don"].notna().any() else d["thanh_tien"].groupby(gd).size()
    if cs == "so_luong":
        x = d[~d["la_phi"].fillna(False).astype(bool)]
        return x[_qcol(x)].groupby(gd[x.index]).sum()
    if cs == "so_khach":
        return d["khach"].groupby(gd).nunique()
    if cs == "lai_gop":
        if d["gia_von"].isna().any():
            raise DacTaLoi("Có dòng bán thiếu giá vốn: engine không tính lãi gộp được, hãy dùng tu_giai.")
        return (d["thanh_tien"] - d["so_luong"] * d["gia_von"]).groupby(gd).sum()
    if cs == "gia_tri_tb_don":
        n = d["ma_don"].groupby(gd).nunique()
        return d["thanh_tien"].groupby(gd).sum() / n
    if cs in ("so_ngay_mua", "so_ngay_ban"):
        return d["ngay"].groupby(gd).nunique()
    if cs == "so_don_huy":
        x = da[da["trang_thai"] == "huỷ"]
        return x["ma_don"].groupby(ga[x.index]).nunique()
    if cs == "so_don_tra_hang":
        x = da[da["tra_hang"].fillna(False).astype(bool)]
        return x["ma_don"].groupby(ga[x.index]).nunique()
    if cs == "so_mat_hang":
        x = d[~d["la_phi"].fillna(False).astype(bool)]
        k = x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"])
        return k.groupby(gd[x.index]).nunique()
    raise DacTaLoi(f"chi_so {cs} không dùng được khi nhóm theo (xếp hạng / đếm ngưỡng).")


def _nhom_serie(ctx, cs, p, loc, nhom):
    """DataFrame [k, ten, ma, val] — số đo `cs` theo từng nhóm `nhom` trong kỳ p."""
    grp = _cs_nhom(cs)
    if grp == "ban":
        loc2 = {a: b for a, b in (loc or {}).items() if a != "trang_thai"}
        _ngoai(ctx, p)
        sub = _ban_loc(ctx, loc2)
        da = sub[_mk_ky(sub["ngay"], p)]
        g = _gk(ctx, da, nhom)
        ok = g["_ok"]
        da, g = da[ok], g[ok]
        s = _agg_ban(cs, da, g)
        meta = g.drop_duplicates("_k").set_index("_k")
        out = pd.DataFrame({"k": s.index.astype(str), "val": s.values.astype(float)})
        out["ten"] = out["k"].map(meta["_t"]).values
        out["ma"] = out["k"].map(meta["_c"]).values
        return out
    if grp == "nhap":
        x = _khung_nhap(ctx, p, loc)
        if nhom == "nha_cung_cap":
            if x["ncc"].notna().sum() == 0:
                raise KhongTraLoiDuoc("Dữ liệu nhập của shop không ghi nhà cung cấp.")
            k = x["ncc"].astype(str)
        elif nhom == "mat_hang":
            k = (x["ma_hang"].where(x["ma_hang"].notna(), x["ten_hang"]) if x["ma_hang"].notna().any() else x["ten_hang"]).astype(str)
        elif nhom in THOI_GIAN:
            ng = pd.to_datetime(x["ngay"])
            k = {"ngay": ng.dt.strftime("%Y-%m-%d"), "thang": ng.dt.strftime("%Y-%m"),
                 "tuan": (ng - pd.to_timedelta(ng.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d"),
                 "quy": ng.dt.year.astype(str) + "-Q" + ((ng.dt.month - 1) // 3 + 1).astype(str), "nam": ng.dt.year.astype(str)}[nhom]
        else:
            raise DacTaLoi(f"nhom_theo={nhom} không áp dụng cho dữ liệu nhập (dùng nha_cung_cap | mat_hang | thang | ngay).")
        if cs == "tien_nhap":
            s = x["thanh_tien"].groupby(k).sum()
        elif cs == "so_phieu_nhap":
            s = x["ma_phieu"].groupby(k).nunique()
        else:
            raise DacTaLoi(f"chi_so {cs} không dùng được khi nhóm theo.")
        out = pd.DataFrame({"k": s.index.astype(str), "val": s.values.astype(float)})
        out["ten"], out["ma"] = out["k"], out["k"]
        if nhom == "mat_hang":
            nm = x.drop_duplicates("ten_hang").assign(_k=k).drop_duplicates("_k").set_index("_k")["ten_hang"]
            out["ten"] = out["k"].map(nm).values
        return out
    if grp == "no" and nhom == "khach" and cs in ("con_no", "so_don_no"):
        o = _no_con(ctx, p, loc)
        o = o[o["_khach"].notna()]
        s = o["no"].groupby(o["_khach"].astype(str)).sum()
        ten = o.drop_duplicates("_khach").assign(_k=lambda t: t["_khach"].astype(str)).set_index("_k")["_ten"]
        out = pd.DataFrame({"k": s.index.astype(str), "val": s.values.astype(float)})
        out["ten"] = out["k"].map(ten).fillna(out["k"]).astype(str).values
        out["ma"] = out["k"]
        return out
    raise DacTaLoi(f"Chưa hỗ trợ xếp hạng / đếm ngưỡng chi_so={cs} theo nhom_theo={nhom}. Nếu là câu hợp lệ, hãy chuyển sang tu_giai.")


def _so_nguong(ng):
    cond = []
    if "toi_thieu" in ng:
        v = _so(ng["toi_thieu"], "nguong.toi_thieu")
        cond.append(("gt", v) if ng.get("nghiem_ngat") else ("ge", v))
    if "lon_hon" in ng:
        cond.append(("gt", _so(ng["lon_hon"], "nguong.lon_hon")))
    if "toi_da" in ng:
        cond.append(("le", _so(ng["toi_da"], "nguong.toi_da")))
    if "nho_hon" in ng:
        cond.append(("lt", _so(ng["nho_hon"], "nguong.nho_hon")))
    return cond


def _thoa(s, cond):
    s = s.round(6)
    m = pd.Series(True, index=s.index)
    for op, v in cond:
        m &= {"ge": s >= v, "gt": s > v, "le": s <= v, "lt": s < v}[op]
    return m


def _mo_ta_nguong(ng):
    t = {"toi_thieu": ">=", "lon_hon": ">", "toi_da": "<=", "nho_hon": "<"}
    return " và ".join(f"{ng.get('chi_so')} {t[k]} {ng[k]}" for k in t if k in ng)


# ----------------------------------------------------------------------------- các loai
def _ky_list(ctx, d, bat_buoc=True):
    ky = d.get("ky")
    if ky is None:
        if bat_buoc:
            raise DacTaLoi("Thiếu \"ky\".")
        return None
    if isinstance(ky, list):
        out = []
        for t in ky:
            out += _periods(ctx, t)
        return out
    return _periods(ctx, ky)


def _ky_mac_dinh(ctx, cs):
    if cs in ("con_no", "so_khach_no"):
        return [_snapshot_p(ctx.as_of.date(), "đến hết hôm nay")]
    raise DacTaLoi("Thiếu \"ky\".")


def _so_ngay(p, loc):
    a = pd.Timestamp(p.start)
    days = pd.date_range(a, pd.Timestamp(p.end))
    if loc.get("thu") not in (None, ""):
        days = days[days.dayofweek == _thu_so(_as_list(loc["thu"])[0])]
    return len(days)


def _gia_tri_tb(ctx, cs, p, loc, d):
    v = _gia_tri(ctx, cs, p, loc)
    if d.get("trung_binh_ngay"):
        if p.start is None:
            raise DacTaLoi("trung_binh_ngay cần kỳ có ngày bắt đầu.")
        n = _so_ngay(p, loc)
        if n == 0:
            raise DacTaLoi("Kỳ không có ngày nào thoả bộ lọc thu.")
        v = _chia_tien(v, n) if cs in _TIEN else round(v / n, 2)
        ctx.mota.append(f"chia cho {n} ngày")
    return v


def _loai_chi_so(ctx, d):
    cs = _enum(d["chi_so"], CHI_SO)
    loc = d.get("loc") or {}
    ps = _ky_list(ctx, d, bat_buoc=False) or _ky_mac_dinh(ctx, cs)
    ctx.mota.append(MO_TA_CHI_SO[cs] + _mo_loc(loc))
    vals = []
    for p in ps:
        v = _gia_tri_tb(ctx, cs, p, loc, d)
        vals.append(v)
        ctx.ghi(f"{cs} | {_mota_ky(p)}{_mo_loc(loc)} = {v:,}" if isinstance(v, (int, float)) else f"{cs} {v}")
    kind = "percent" if cs == "ti_le_huy" else "number"
    kd = _enum(d.get("kieu_dap_an"), KIEU)
    if len(vals) > 1 or kd == "list":
        return "list", [_gon(v) for v in vals]
    if kd in ("percent",) and cs != "ti_le_huy":
        raise DacTaLoi(f"chi_so {cs} không phải phần trăm: nếu câu hỏi hỏi %, dùng loai=so_sanh (tăng/giảm %) hoặc loai=ti_trong (tỉ trọng).")
    if kd in ("string", "date", "table", "boolean"):
        raise DacTaLoi(f"loai=chi_so cho ra một số, không phải {kd}.")
    return kind, _gon(vals[0])


def _mo_loc(loc):
    xs = [f"{k}={v}" for k, v in (loc or {}).items() if v not in (None, "", [])]
    return f" [lọc: {', '.join(xs)}]" if xs else ""


def _loai_so_sanh(ctx, d):
    cs = _enum(d["chi_so"], CHI_SO)
    loc = d.get("loc") or {}
    ps = _ky_list(ctx, d)
    ks = d.get("ky_so_sanh")
    lcs = d.get("loc_so_sanh")
    p = ps[0]
    if ks is None and len(ps) >= 2:
        c = ps[1]
    elif ks is None and lcs:
        c = p
    else:
        c = _ky_so(ctx, p, ks)
    loc_b = {**loc, **(lcs or {})}
    a = _gia_tri(ctx, cs, p, loc)
    b = _gia_tri(ctx, cs, c, loc_b)
    ctx.mota.append(MO_TA_CHI_SO[cs])
    ctx.ghi(f"Kỳ hỏi: {_mota_ky(p)}{_mo_loc(loc)} → {a:,}")
    ctx.ghi(f"Kỳ so : {_mota_ky(c)}{_mo_loc(loc_b)} → {b:,}")
    kd = _enum(d.get("kieu_dap_an"), KIEU) or "percent"
    if kd == "percent":
        if b == 0:
            raise DacTaLoi(f"Kỳ so sánh ({_mota_ky(c)}) có {cs} = 0 nên % thay đổi không xác định (chia cho 0).")
        r = _pct2(a - b, b)
        ctx.mota.append("% thay đổi = (kỳ hỏi − kỳ so) / kỳ so × 100, kỳ so sánh = " + _mota_ky(c))
        ctx.ghi(f"% = ({a:,} − {b:,}) / {b:,} × 100 = {r}")
        return "percent", r
    if kd == "number":
        r = _fmt(cs, a - b) if cs != "ti_le_huy" else round(a - b, 2)
        ctx.mota.append("Chênh lệch = kỳ hỏi − kỳ so (âm nếu ít hơn)")
        return "number", _gon(r)
    if kd == "boolean":
        return "boolean", bool(a > b)
    raise DacTaLoi("so_sanh chỉ hỗ trợ kieu_dap_an: percent | number | boolean.")


def _loai_ti_trong(ctx, d):
    cs = _enum(d["chi_so"], CHI_SO)
    loc, lm = d.get("loc") or {}, d.get("loc_mau") or {}
    ps = _ky_list(ctx, d)
    out = []
    for p in ps:
        a = _gia_tri(ctx, cs, p, loc)
        b = _gia_tri(ctx, cs, p, lm)
        if b == 0:
            raise DacTaLoi(f"Mẫu số ({cs}{_mo_loc(lm)}) bằng 0 trong kỳ {_mota_ky(p)}: tỉ trọng không xác định.")
        r = _pct2(a, b)
        out.append(r)
        ctx.ghi(f"Tử: {a:,}{_mo_loc(loc)} | Mẫu: {b:,}{_mo_loc(lm)} | {_mota_ky(p)} → {r}%")
    ctx.mota.append(f"Tỉ trọng = {cs}(loc) / {cs}(cả shop cùng kỳ{', loc_mau' if lm else ''}) × 100")
    if len(out) > 1:
        return "list", out
    return "percent", out[0]


def _dinh_dang_gia_tri(cs, v):
    return _gon(_fmt(cs, v))


def _loai_xep_hang(ctx, d):
    cs = _enum(d["chi_so"], CHI_SO)
    nhom = _enum(d["nhom_theo"], NHOM_THEO)
    loc = d.get("loc") or {}
    ps = _ky_list(ctx, d, bat_buoc=False) or _ky_mac_dinh(ctx, cs)
    if len(ps) != 1:
        raise DacTaLoi("xep_hang chỉ nhận MỘT kỳ.")
    p = ps[0]
    df = _nhom_serie(ctx, cs, p, loc, nhom)
    if df.empty:
        raise KhongTraLoiDuoc(f"Không có dữ liệu nào để xếp hạng trong kỳ {_mota_ky(p)}.")
    ng = d.get("nguong")
    if ng:
        cs2 = _enum(ng.get("chi_so"), CHI_SO) or cs
        d2 = _nhom_serie(ctx, cs2, p, loc, nhom).set_index("k")["val"]
        keep = set(d2[_thoa(d2, _so_nguong(ng))].index)
        df = df[df["k"].isin(keep)]
        ctx.mota.append("Chỉ xét nhóm có " + _mo_ta_nguong(ng))
        if df.empty:
            raise DacTaLoi("Không nhóm nào thoả điều kiện nguong; kiểm tra lại ngưỡng.")
    giam = not str(d.get("sap_xep") or "giam").startswith("tang") and not fold(d.get("sap_xep") or "giam").startswith("tang")
    df = df.assign(_v=df["val"].round(6)).sort_values(by=["_v", "ma"], ascending=[not giam, True], kind="mergesort")
    k = int(_so(d["top_k"])) if d.get("top_k") is not None else None
    kd = _enum(d.get("kieu_dap_an"), KIEU)
    ctx.mota.append(f"Xếp theo {cs} {'giảm' if giam else 'tăng'} dần theo {nhom}; hoà: mã/khoá tăng dần" + _mo_loc(loc))
    for _, r in df.head(8).iterrows():
        ten = r["ten"] if str(r["ten"]) == str(r["ma"]) else f"{r['ten']} ({r['ma']})"
        ctx.ghi(f"  {ten}: {_dinh_dang_gia_tri(cs, r['val']):,}" if cs != "ti_le_huy" else f"  {ten}: {r['val']}")
    if kd is None:
        kd = ("date" if nhom in ("ngay", "thang") else "string") if k in (None, 1) else "list"
    top = df.head(k if k else (1 if kd in ("string", "date", "number") else len(df)))
    names = []
    for _, r in top.iterrows():
        t = r["ten"]
        if nhom == "gio" or nhom == "nam":
            t = int(str(t))
        elif nhom == "thu":
            t = int(str(t)) + 2
        names.append(t)
    if kd in ("string", "date", "number"):
        v = names[0]
        if kd == "number" and nhom not in ("gio", "nam", "thu"):
            v = _dinh_dang_gia_tri(cs, top.iloc[0]["val"])
        if kd == "string" and isinstance(v, (int, np.integer)):
            kd = "number"
        return kd, v
    if kd == "list":
        return "list", names
    if kd == "table":
        cols = [_TEN_COT.get(nhom, nhom), _TEN_SO.get(cs, cs)]
        rows = [[n, _dinh_dang_gia_tri(cs, v)] for n, v in zip(names, top["val"])]
        return "table", {"columns": cols, "rows": rows}
    raise DacTaLoi(f"xep_hang không hỗ trợ kieu_dap_an={kd}.")


def _loai_dem_nguong(ctx, d):
    ng = d["nguong"]
    cs = _enum(ng["chi_so"], CHI_SO)
    nhom = {"khach": "khach", "don": "don", "ngay": "ngay", "mat_hang": "mat_hang"}[_enum(d["doi_tuong_dem"], DOI_TUONG)]
    loc = d.get("loc") or {}
    ps = _ky_list(ctx, d, bat_buoc=False) or _ky_mac_dinh(ctx, cs)
    if len(ps) != 1:
        raise DacTaLoi("dem_nguong chỉ nhận MỘT kỳ.")
    p = ps[0]
    cond = _so_nguong(ng)
    s = _nhom_serie(ctx, cs, p, loc, nhom)
    n = int(_thoa(s["val"], cond).sum()) if len(s) else 0
    ctx.mota.append(f"Đếm {nhom} có {_mo_ta_nguong(ng)} trong {_mota_ky(p)}" + _mo_loc(loc))
    ctx.ghi(f"Số {nhom} xét: {len(s)}; thoả điều kiện: {n}")
    kd = _enum(d.get("kieu_dap_an"), KIEU) or "number"
    if kd == "percent":
        pop = len(s) if cs not in ("so_don_huy", "so_don_tra_hang") else len(_nhom_serie(ctx, "so_don", p, loc, nhom))
        if pop == 0:
            raise DacTaLoi("Không có đối tượng nào trong kỳ để tính %.")
        ctx.ghi(f"% = {n} / {pop} × 100")
        return "percent", _pct2(n, pop)
    return "number", n


def _chia_ky(ctx, p, nhom):
    if p.start is None:
        raise DacTaLoi("theo_ky cần kỳ có ngày bắt đầu.")
    a, b = pd.Timestamp(p.start), pd.Timestamp(p.end)
    out = []
    if nhom == "ngay":
        for x in pd.date_range(a, b):
            out.append((x.strftime("%Y-%m-%d"), x, x))
    else:
        cur = a
        while cur <= b:
            if nhom == "thang":
                nxt = (cur.replace(day=1) + pd.offsets.MonthBegin(1))
                lab = cur.strftime("%Y-%m")
            elif nhom == "tuan":
                nxt = cur - pd.Timedelta(days=cur.dayofweek) + pd.Timedelta(days=7)
                lab = (cur - pd.Timedelta(days=cur.dayofweek)).strftime("%Y-%m-%d")
            elif nhom == "quy":
                q = (cur.month - 1) // 3
                nxt = pd.Timestamp(cur.year + (1 if q == 3 else 0), 1 if q == 3 else 3 * (q + 1) + 1, 1)
                lab = f"{cur.year}-Q{q + 1}"
            else:
                nxt = pd.Timestamp(cur.year + 1, 1, 1)
                lab = str(cur.year)
            out.append((lab, cur, min(nxt - pd.Timedelta(days=1), b)))
            cur = nxt
    if len(out) > 400:
        raise DacTaLoi(f"theo_ky tạo {len(out)} kỳ con (quá nhiều, tối đa 400): hãy thu hẹp kỳ hoặc đổi nhom_theo.")
    return out


def _loai_theo_ky(ctx, d):
    cs = _enum(d["chi_so"], CHI_SO)
    nhom = _enum(d["nhom_theo"], THOI_GIAN)
    loc = d.get("loc") or {}
    ps = _ky_list(ctx, d)
    if len(ps) != 1:
        raise DacTaLoi("theo_ky chỉ nhận MỘT kỳ bao (vd 'từ tháng 2/2026 đến tháng 4/2026'); nhiều kỳ rời nhau thì dùng ky dạng list với loai=chi_so.")
    p0 = ps[0]
    subs = _chia_ky(ctx, p0, nhom)
    vals = []
    for lab, a, b in subs:
        pp = Period(a.date(), b.date(), "range", lab)
        try:
            vals.append(_gia_tri(ctx, cs, pp, loc))
        except KhongTraLoiDuoc:
            raise
    ctx.mota.append(MO_TA_CHI_SO[cs] + f"; tính cho từng {nhom} trong {_mota_ky(p0)}" + _mo_loc(loc))
    for (lab, a, b), v in list(zip(subs, vals))[:12]:
        ctx.ghi(f"  {lab}: {v:,}")
    kd = _enum(d.get("kieu_dap_an"), KIEU) or "list"
    if kd == "boolean":
        tang = not fold(d.get("sap_xep") or "tang").startswith("giam")
        ok = all((vals[i + 1] > vals[i]) if tang else (vals[i + 1] < vals[i]) for i in range(len(vals) - 1))
        ctx.mota.append(("Mỗi kỳ phải cao hơn kỳ liền trước (tăng liên tục)" if tang else "Mỗi kỳ phải thấp hơn kỳ liền trước (giảm liên tục)"))
        return "boolean", bool(ok)
    if kd == "table":
        return "table", {"columns": [nhom, _TEN_SO.get(cs, cs)], "rows": [[lab, _gon(v)] for (lab, a, b), v in zip(subs, vals)]}
    if kd == "list" and d.get("top_k") is None and d.get("sap_xep") is None:
        return "list", [_gon(v) for v in vals]
    # xếp hạng các kỳ con (vd tháng nào doanh thu cao nhất)
    giam = not fold(d.get("sap_xep") or "giam").startswith("tang")
    rows = sorted(zip([s[0] for s in subs], vals), key=lambda t: ((-t[1] if giam else t[1]), t[0]))
    k = int(_so(d["top_k"])) if d.get("top_k") is not None else 1
    if kd in ("date", "string"):
        return ("date" if nhom in ("ngay", "thang") else "string"), rows[0][0]
    return "list", [r[0] for r in rows[:k]]


def _chay(ctx, d):
    loai = _enum(d["loai"], LOAI)
    fn = {"chi_so": _loai_chi_so, "so_sanh": _loai_so_sanh, "ti_trong": _loai_ti_trong, "xep_hang": _loai_xep_hang,
          "dem_nguong": _loai_dem_nguong, "theo_ky": _loai_theo_ky}.get(loai)
    if fn is None:
        raise DacTaLoi(f"loai={loai} không thực thi bằng engine.")
    return fn(ctx, d)


def _gop_phan(dac_ta, ph):
    m = {k: v for k, v in dac_ta.items() if k not in ("cac_phan", "kieu_dap_an")}   # kieu_dap_an của cha (list) không áp cho từng phần
    m.update({k: v for k, v in ph.items() if k != "loc"})
    m["loc"] = {**(dac_ta.get("loc") or {}), **(ph.get("loc") or {})}
    return m


def thuc_thi(dac_ta, biz, rulebook="", cal=None, as_of=None):
    """Thực thi đặc tả DSL. Trả {"ok","answer","mo_ta","trung_gian","loi"} (+ "khong_tra_loi_duoc": True khi unanswerable theo §7)."""
    out = {"ok": False, "answer": None, "mo_ta": "", "trung_gian": "", "loi": ""}
    ctx = None
    try:
        errs = kiem_tra_dac_ta(dac_ta)
        if errs:
            out["loi"] = "Đặc tả sai schema: " + " | ".join(errs)
            return out
        loai = _enum(dac_ta["loai"], LOAI)
        if loai == "tu_giai":
            out["loi"] = "loai=tu_giai: engine không tính; chuyển sang vòng tự giải bằng code."
            return out
        if loai == "khong_du_du_lieu":
            ly = str(dac_ta.get("ly_do") or dac_ta.get("ghi_chu") or "dữ liệu phát hành không đủ để trả lời")
            out.update(ok=True, answer={"type": "unanswerable", "reason": ly}, mo_ta="Model xác định câu không trả lời được: " + ly,
                       khong_tra_loi_duoc=True)
            return out
        ctx = _Ctx(biz, rulebook, cal, as_of)
        if dac_ta.get("cac_phan"):
            vals = []
            for i, ph in enumerate(dac_ta["cac_phan"]):
                try:
                    kind, v = _chay(ctx, _gop_phan(dac_ta, ph))
                except DacTaLoi as e:
                    raise DacTaLoi(f"cac_phan[{i}]: {e}")
                if kind in ("list", "table"):
                    raise DacTaLoi(f"cac_phan[{i}] phải cho ra MỘT giá trị (số/ngày/tên), không phải {kind}.")
                vals.append(v)
            kind, v = "list", vals
        else:
            kind, v = _chay(ctx, dac_ta)
        out.update(ok=True, answer={"type": kind, "answer": v})
    except KhongTraLoiDuoc as e:
        out.update(ok=True, answer={"type": "unanswerable", "reason": str(e)}, khong_tra_loi_duoc=True, loi="")
        out["ghi_chu"] = "Theo rulebook §7: " + str(e)
    except DacTaLoi as e:
        out.update(ok=False, answer=None, loi=str(e))
    except Exception as e:            # lỗi bất ngờ: báo cho model, không làm sập pipeline
        out.update(ok=False, answer=None, loi=f"Lỗi nội bộ engine ({type(e).__name__}: {e}). Hãy kiểm tra lại đặc tả hoặc chuyển sang tu_giai.")
    if ctx is not None:
        out["mo_ta"] = "; ".join(dict.fromkeys(ctx.mota))
        out["trung_gian"] = "\n".join(ctx.lines[:15])
    return out
