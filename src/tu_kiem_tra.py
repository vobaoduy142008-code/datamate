"""
Tự kiểm tra toàn bộ code KHÔNG cần LLM (chạy vài chục giây). Chạy sau mỗi lần sửa code:

    uv run python src/tu_kiem_tra.py                 # kiểm tra trên data/train
    uv run python src/tu_kiem_tra.py --data data/train --nhanh   # bỏ phần đọc dữ liệu shop thật

In ra từng mục ĐẠT / LỖI; cuối cùng là tổng kết. Có mục LỖI thì đừng commit, báo cả nhóm.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

KET_QUA = []


def kiem(ten, dieu_kien, chi_tiet=""):
    KET_QUA.append(bool(dieu_kien))
    print(("  ĐẠT  " if dieu_kien else "  LỖI  ") + ten + ("" if dieu_kien else f"   <-- {chi_tiet}"))


def muc(ten):
    print(f"\n== {ten}")


def test_scorer(data):
    muc("scorer.py: chấm đáp án đúng phải ra PA = 1, nộp rỗng ra 0")
    import scorer
    ref = os.path.join(data, "answers.jsonl")
    if not os.path.exists(ref):
        kiem("có answers.jsonl", False, f"không thấy {ref}")
        return
    refs = scorer.load(ref)
    ok = sum(scorer.score_one({"type": r["type"], "answer": r.get("answer")}, r) for r in refs.values())
    kiem("nộp đúng đáp án -> mọi câu +1", ok == len(refs), f"{ok}/{len(refs)}")
    kiem("abstain -> 0 điểm", all(scorer.score_one({"type": "abstain"}, r) == 0 for r in refs.values()))
    r = next(r for r in refs.values() if r["type"] == "number")
    kiem("number sai -> -1", scorer.score_one({"type": "number", "answer": r["answer"] + 1}, r) == -1)
    kiem("sai type -> -1", scorer.score_one({"type": "string", "answer": str(r["answer"])}, r) == -1)
    al = next((r for r in refs.values() if r["type"] == "string" and r.get("alias")), None)
    if al:
        kiem("string chấp nhận mã (alias)", scorer.score_one({"type": "string", "answer": al["alias"][0]}, al) == 1)


def test_normalize():
    muc("agent.normalize_answer: chuẩn hoá đáp án đúng format §9")
    import numpy as np
    from agent import normalize_answer as N
    kiem("number numpy -> int", N({"type": "number", "answer": np.int64(4820000)})[0] == {"type": "number", "answer": 4820000})
    kiem("number 18.6 giữ nguyên", N({"type": "number", "answer": 18.6})[0]["answer"] == 18.6)
    kiem("number 4820000.0 -> 4820000", isinstance(N({"type": "number", "answer": 4820000.0})[0]["answer"], int))
    kiem("percent làm tròn 2 số", N({"type": "percent", "answer": -10.7654})[0]["answer"] == -10.77)
    kiem("date Timestamp -> YYYY-MM-DD", N({"type": "date", "answer": pd.Timestamp("2025-05-10")})[0]["answer"] == "2025-05-10")
    kiem("date tháng YYYY-MM", N({"type": "date", "answer": "2025-09"})[0]["answer"] == "2025-09")
    kiem("date sai format bị bắt", N({"type": "date", "answer": "10/05/2025"})[0] is None)
    kiem("list 1 phần tử vẫn là list", N({"type": "list", "answer": ["A"]})[0]["answer"] == ["A"])
    kiem("list chuỗi bị bắt", N({"type": "list", "answer": "A, B"})[0] is None)
    kiem("boolean numpy", N({"type": "boolean", "answer": np.bool_(True)})[0]["answer"] is True)
    kiem("boolean 'có' bị bắt", N({"type": "boolean", "answer": "có"})[0] is None)
    t = N({"type": "table", "answer": {"columns": ["k", "v"], "rows": [["A", np.float64(5.0)]]}})[0]
    kiem("table ô số -> int", t and t["answer"]["rows"] == [["A", 5]])
    kiem("unanswerable", N({"type": "unanswerable", "reason": "x"})[0] == {"type": "unanswerable", "reason": "x"})
    kiem("NaN bị bắt", N({"type": "number", "answer": float("nan")})[0] is None)
    kiem("thiếu type bị bắt", N({"answer": 1})[0] is None)


def test_helpers():
    muc("agent: các hàm trợ giúp cho LLM")
    from agent import _explain_error, _extract_code, _in_period, _match, _run, _status_one
    cases = {"xong": "hoàn thành", "ok": "hoàn thành", "Đã giao": "hoàn thành", "completed": "hoàn thành",
             "Đã hủy đơn": "huỷ", "khách hủy": "huỷ", "cancelled": "huỷ", "Không giao được": "huỷ",
             "trả lại": "trả hàng", "Hoàn trả": "trả hàng", "Đổi trả": "trả hàng", "Đã hoàn hàng": "trả hàng",
             "returned": "trả hàng", "Khách trả": "trả hàng", "Đang vận chuyển": "đang giao", "đang gửi": "đang giao",
             "Đang xử lý": "đang giao", "Đang giao dịch": "đang giao", "shipping": "đang giao", None: "hoàn thành"}
    bad = {k: _status_one(k) for k, v in cases.items() if _status_one(k) != v}
    kiem("status_of xếp đúng 4 nhóm (20 kiểu ghi)", not bad, bad)
    s = pd.Series(["Sting dâu 330ml", "Sting dầu 330ml", "PANADOL Extra hộp", "Cà phê sữa (M)", "Cà phê sữa (L)"])
    kiem("match giữ dấu: 'Sting dâu' không lẫn 'Sting dầu'", _match(s, "sting dâu 330ml").tolist() == [1, 0, 0, 0, 0])
    kiem("match không phân biệt hoa thường", _match(s, "panadol extra hộp").sum() == 1)
    kiem("match chứa: 'cà phê sữa' -> 2 dòng", _match(s, "cà phê sữa").sum() == 2)
    kiem("match chuỗi rỗng -> không dòng nào", _match(s, "").sum() == 0)
    p = {"start": pd.Timestamp("2025-05-01"), "end": pd.Timestamp("2025-05-31")}
    d = pd.Series(pd.to_datetime(["2025-04-30 23:00", "2025-05-01 00:00", "2025-05-31 22:15", "2025-06-01 00:00"]))
    kiem("in_period tính cả 2 đầu, bỏ giờ", _in_period(d, p).tolist() == [False, True, True, False])
    from agent import _valid_orders
    df = pd.DataFrame({"ma": ["A", "A", "B", "C", "D"], "st": ["xong", "trả lại", "hủy", "ok", "Đang giao"]})
    kiem("valid_orders: bỏ cả đơn có 1 dòng trả, bỏ đơn huỷ, giữ đơn đang giao",
         _valid_orders(df, "st", "ma").tolist() == [False, False, False, True, True])
    from agent import _self_check
    tb = {"SoBanHang": pd.DataFrame({"Trạng thái xử lý": ["Đã hoàn thành", "trả hàng", "Không giao được"] * 3})}
    kiem("tự kiểm: nhắc khi tính doanh thu mà quên bỏ đơn huỷ/trả",
         _self_check("Doanh thu tháng 7", "x = tables['SoBanHang']", tb) is not None)
    kiem("tự kiểm: không nhắc khi đã dùng valid_orders / câu hỏi về huỷ",
         _self_check("Doanh thu", "valid_orders(tables['SoBanHang'], 'Trạng thái xử lý')", tb) is None
         and _self_check("tỉ lệ huỷ đơn", "x = tables['SoBanHang']", tb) is None)
    from agent import _restore_names
    tb2 = {"ct": pd.DataFrame({"Tên hàng": ["Cà phê sữa (M)", "Cà phê đen (M)", "Bạc xỉu (M)"]})}
    kiem("trả lại đúng cách viết tên trong dữ liệu ('càphêsữa(m)' -> 'Cà phê sữa (M)')",
         _restore_names({"type": "list", "answer": ["càphêsữa(m)", "Cà phê đen (M)", "BẠC XỈU (M)"]}, tb2)["answer"]
         == ["Cà phê sữa (M)", "Cà phê đen (M)", "Bạc xỉu (M)"])
    kiem("extract_code lấy khối ```python```", _extract_code("abc\n```python\nx=1\n```") == "x=1")
    kiem("extract_code khối không xuống dòng", _extract_code("```python x=1```") == "x=1")
    env = {"tables": {"don_hang": pd.DataFrame({"Ngày bán": [1]})}}
    out, err = _run("print(1+1)", env)
    kiem("chạy code, lấy print", out.strip() == "2" and err is None)
    out, err = _run("x = tables['don_hang']['Ngay ban']", env)
    kiem("lỗi tên cột có gợi ý tên đúng", err and "Ngày bán" in err, err)
    import agent
    old = agent.CODE_TIMEOUT
    agent.CODE_TIMEOUT = 1
    out, err = _run("while True: pass", env)
    agent.CODE_TIMEOUT = old
    kiem("code chạy quá giờ bị dừng", err and "quá" in err, err)
    _ = _explain_error


def test_periods(data):
    muc("periods.py: cách nói thời gian theo rulebook §2")
    from periods import Calendar, find_periods
    cal_path = os.path.join(data, "calendar.csv")
    cal = Calendar(cal_path) if os.path.exists(cal_path) else None

    def one(q, today):
        ps = find_periods(q, today, cal)
        return ps[0] if ps else None

    def rng(p):
        return (str(p.start) if p and p.start else None, str(p.end) if p else None)

    p = find_periods("dthu tháng 8/2025 hơn hay kém tháng trước bao nhiêu %", "2026-01-10", cal)
    kiem("'tháng 8/2025 hơn hay kém tháng trước': so với 7/2025 (không phải tháng trước của hôm nay)",
         len(p) == 1 and str(p[0].meta.get("compare").start) == "2025-07-01", [str(x) for x in p])
    kiem("'từ ngày 1/1 đến hết hôm nay' = 1/1 → hôm nay",
         rng(one("Tính từ ngày 1/1 đến hết hôm nay, doanh thu năm nay?", "2026-06-18")) == ("2026-01-01", "2026-06-18"))
    p = find_periods("So số lon Red Bull bán ra trong 7 ngày kể từ ngày 05/01/2026, ngày tăng giá, với 7 ngày ngay trước ngày đó", "2026-06-01", cal)
    kiem("'7 ngày kể từ 05/01/2026' + so với 7 ngày ngay trước",
         len(p) == 1 and rng(p[0]) == ("2026-01-05", "2026-01-11") and str(p[0].meta["compare"].start) == "2025-12-29",
         [str(x) for x in p])
    kiem("'tuần trước' một mình vẫn là cả tuần liền trước",
         rng(one("doanh thu tuần trước", "2026-06-18")) == ("2026-06-08", "2026-06-14"))

    import agent as _ag
    _old_chat = _ag.llm.chat
    seen = {}

    def fake_chat(messages):
        seen["user"] = messages[1]["content"]
        return '```python\nANSWER = {"type": "number", "answer": 1}\n```', None
    _ag.llm.chat = fake_chat
    try:
        root = os.path.join(data, "data")
        dirs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))) if os.path.isdir(root) else []
        biz = dirs[0] if dirs else None
        if biz:
            _ag.solve({"id": "t", "business_id": biz, "question": "tháng 10/2025 so với tháng trước", "as_of_date": "2026-05-16"}, data)
            kiem("prompt ghi PERIODS[1] = kỳ so sánh", "PERIODS[1] = kỳ so sánh" in seen.get("user", ""), seen.get("user", "")[-300:])
    finally:
        _ag.llm.chat = _old_chat
    kiem("tháng này (đến hết hôm nay)", rng(one("doanh thu tháng này", "2026-09-07")) == ("2026-09-01", "2026-09-07"))
    p = one("tháng này so với tháng trước", "2026-03-31")
    kiem("so tháng trước cùng số ngày, tháng ngắn hơn", rng(p.meta["compare"]) == ("2026-02-01", "2026-02-28"))
    kiem("tuần trước = cả tuần T2-CN", rng(one("tuần trước", "2026-03-04")) == ("2026-02-23", "2026-03-01"))
    kiem("tuần này = T2 đến hôm nay", rng(one("tuần này", "2026-03-04")) == ("2026-03-02", "2026-03-04"))
    kiem("7 ngày gần nhất kể cả hôm nay", rng(one("7 ngày gần nhất", "2026-03-04")) == ("2026-02-26", "2026-03-04"))
    kiem("tháng 12 không ghi năm, sau hôm nay -> năm trước", rng(one("tháng 12", "2026-03-04")) == ("2025-12-01", "2025-12-31"))
    kiem("quý 2 không ghi năm -> năm trước", rng(one("quý 2", "2026-03-04")) == ("2025-04-01", "2025-06-30"))
    kiem("từ đầu năm", rng(one("từ đầu năm tới nay", "2026-03-04")) == ("2026-01-01", "2026-03-04"))
    kiem("đến hết ngày X = từ đầu dữ liệu", rng(one("còn nợ đến hết ngày 14/08/2026", "2026-08-20")) == (None, "2026-08-14"))
    kiem("viết tắt không dấu 'thang 10/2025'", rng(one("thang 10/2025 dthu", "2026-05-16")) == ("2025-10-01", "2025-10-31"))
    if cal:
        kiem("Tết 2026: 23 Chạp -> mùng 10", rng(one("dịp Tết 2026", "2026-06-28")) == ("2026-02-10", "2026-02-26"))


def test_loader_synthetic():
    muc("loader.py: dữ liệu giả có đủ các bẫy (dòng tổng, họ Tống, ngày mơ hồ, tiền viết tay, ô trống theo nhóm)")
    from loader import load_business
    d = tempfile.mkdtemp()
    rows = [["Mã HĐ", "Ngày", "Khách", "Tên hàng", "SL", "Thành tiền", "Giảm giá HĐ"],
            ["HD1", "01/05/2025", "Tống Thắng", "Sữa A", "2 hộp", "1tr440", "10 000"],
            ["", "", "", "Sữa B", "1", "396k", ""],
            ["HD2", "02/05/2025", "Công ty An", "Sữa A", "1", "105.000đ", "0"],
            ["HD2", "02/05/2025", "Công ty An", "Sữa A", "1", "105.000đ", "0"],
            ["", "", "", "", "", "2.046.000", "Tổng ngày 02/05"],
            ["HD3", "13/05/2025", "kl", "Sữa B", "1", "50000", "0"]]
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(os.path.join(d, "ban_hang.csv"), index=False)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write("Dữ liệu có đến hết 31/05/2025.\n\n## Cột trong `ban_hang.csv`\n- **Mã HĐ**: mã đơn\n")
    b = load_business(d, use_cache=False)
    t = b.tables["ban_hang"]
    kiem("bỏ dòng 'Tổng ngày'", not t["Giảm giá HĐ"].astype(str).str.contains("Tổng").any() and len(t) == 4, len(t))
    kiem("KHÔNG xoá khách họ 'Tống'", (t["Khách"] == "Tống Thắng").sum() == 2)
    kiem("điền xuống mã đơn ở dòng sau (§8)", t["Mã HĐ"].tolist() == ["HD1", "HD1", "HD2", "HD3"], t["Mã HĐ"].tolist())
    kiem("bỏ dòng trùng y hệt trong cùng đơn (§8)", (t["Mã HĐ"] == "HD2").sum() == 1)
    kiem("tiền viết tay 1tr440 / 396k / 105.000đ", t["Thành tiền"].tolist() == [1440000, 396000, 105000, 50000], t["Thành tiền"].tolist())
    kiem("'Giảm giá HĐ' là số (không phải mã)", pd.api.types.is_numeric_dtype(t["Giảm giá HĐ"]))
    kiem("ngày dd/mm/yyyy", t["Ngày"].dt.strftime("%Y-%m-%d").tolist() == ["2025-05-01", "2025-05-01", "2025-05-02", "2025-05-13"])
    kiem("số lượng '2 hộp' -> 2 + cột đơn vị", t["SL"].tolist() == [2, 1, 1, 1] and "SL (đơn vị ghi kèm)" in t.columns)
    kiem("ngày cuối dữ liệu từ README", str(b.data_end) == "2025-05-31")


def test_sales_synthetic():
    muc("sales.py: bảng bán hàng chuẩn (đơn + chi tiết, trả hàng ở bảng khác, dòng số âm, quy đổi đơn vị)")
    from loader import Business
    from sales import build_sales
    # shop kiểu "phần mềm": bảng đơn + bảng chi tiết + bảng lô có số lượng trả
    don = pd.DataFrame({"Mã hóa đơn": ["A1", "A2", "A3", "A4"],
                        "Thời gian": pd.to_datetime(["2025-05-01 08:10", "2025-05-01 15:00", "2025-05-02 09:00", "2025-05-03 10:00"]),
                        "Trạng thái": ["Hoàn thành", "Đã hủy", "Hoàn thành", "Hoàn thành"],
                        "Mã KH": ["KH1", None, "KH2", "KH1"]})
    ct = pd.DataFrame({"Mã hóa đơn": ["A1", "A1", "A2", "A3", "A4"], "Mã hàng": ["SP1", "SP2", "SP1", "SP1", "SP2"],
                       "Tên hàng": ["Sữa A", "Bánh B", "Sữa A", "Sữa A", "Bánh B"], "Số lượng": [2.0, 1, 1, 1, 3],
                       "Đơn giá": [10000.0, 5000, 10000, 10000, 5000], "Thành tiền": [20000.0, 5000, 10000, 10000, 15000]})
    lo = pd.DataFrame({"Mã hóa đơn": ["A1", "A3", "A4"], "Số lượng trả (ĐVT cơ bản)": [0.0, 1, 0]})
    kh = pd.DataFrame({"Mã khách hàng": ["KH1", "KH2"], "Tên khách hàng": ["Chị Lan", "Anh Ba"]})
    b = Business(path="x", tables={"DanhSachHoaDon": don, "DanhSachHoaDonChiTiet": ct, "DanhSachHoaDonChiTietLo": lo,
                                   "DanhSachKhachHang": kh})
    ban = build_sales(b)
    kiem("dựng được bảng", ban is not None and len(ban) == 5, None if ban is None else len(ban))
    if ban is None:
        return
    dt = ban.loc[ban["tinh_dt"], "thanh_tien"].sum()
    kiem("doanh thu bỏ đơn huỷ (A2) và đơn trả một phần ở bảng lô (A3)", dt == 40000, dt)
    kiem("ngày lấy từ bảng đơn, bỏ giờ; thoi_diem giữ giờ",
         ban["ngay"].dt.strftime("%Y-%m-%d").tolist()[:2] == ["2025-05-01"] * 2 and ban["thoi_diem"].dt.hour.iloc[0] == 8)
    kiem("trạng thái chuẩn", ban.loc[ban["ma_don"] == "A2", "trang_thai"].iloc[0] == "huỷ")
    kiem("tên khách lấy từ danh mục khách", ban.loc[ban["ma_don"] == "A4", "ten_khach"].iloc[0] == "Chị Lan")
    # sổ phẳng: trả hàng ghi số âm, ngày trả khác ngày bán; quy đổi theo bảng quy đổi
    so = pd.DataFrame({"Mã phiếu": ["B1", "B1", "B2", "B2"],
                       "Ngày lập": pd.to_datetime(["2025-05-01", "2025-05-01", "2025-05-31", "2025-06-02"]),
                       "Tên SP": ["Bia T", "Mì H", "BIA  T", "Bia T"], "Số lượng hàng": [1.0, 2, 2, -1],
                       "DonVi": ["thùng", "gói", "lon", "lon"], "Tiền": [300000.0, 8000, 20000, -10000],
                       "Ghi chú TT": ["xong", "xong", "xong", "trả lại"]})
    qd = pd.DataFrame({"Hàng": ["Bia T"], "ĐV lớn": ["thùng"], "Hệ số QĐ": [24.0]})
    b2 = Business(path="y", tables={"SoBanHang": so, "DanhMuc": qd},
                  col_meaning={"DanhMuc": {"Hàng": "tên hàng", "ĐV lớn": "đơn vị lớn", "Hệ số QĐ": "hệ số: số đơn vị cơ bản"}})
    ban2 = build_sales(b2)
    kiem("sổ phẳng: dựng được bảng", ban2 is not None and len(ban2) == 4)
    if ban2 is None:
        return
    kiem("đơn có dòng số âm là đơn trả hàng (bỏ cả đơn B2)", ban2.loc[ban2["tinh_dt"], "thanh_tien"].sum() == 308000)
    kiem("dòng trả mang ngày bán của đơn (31/05), không phải ngày trả",
         (ban2.loc[ban2["ma_don"] == "B2", "ngay"] == pd.Timestamp("2025-05-31")).all())
    kiem("tên hàng nhiều cách viết gom về một (theo bỏ hoa thường/khoảng trắng)",
         ban2.loc[ban2["ten_hang"].str.lower() == "bia t", "ten_hang"].nunique() == 1)
    kiem("quy đổi: 1 thùng Bia T = 24 lon", ban2["so_luong_co_ban"].tolist()[:2] == [24.0, 2.0], ban2["so_luong_co_ban"].tolist())


def test_real_data(data, n_biz):
    muc(f"Đọc dữ liệu thật ({n_biz} shop đầu của {data}) + chạy agent với LLM giả")
    from loader import load_business
    root = os.path.join(data, "data")
    shops = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))[:n_biz]
    for s in shops:
        try:
            b = load_business(os.path.join(root, s))
            kiem(f"{s}: {len(b.tables)} bảng, schema {len(b.schema_text())} ký tự", len(b.tables) > 0)
        except Exception as e:
            kiem(f"{s}", False, repr(e))
    out = tempfile.mkdtemp()
    env = {**os.environ, "LLM_BACKEND": "mock", "AGENT_VERBOSE": "0"}
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "main.py"), "--data", data, "--out", out,
                        "--sample", "3", "--seed", "7", "--quiet"], env=env, capture_output=True, text=True)
    subs = [json.loads(x) for x in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")] if r.returncode == 0 else []
    kiem("main.py chạy hết, ghi submission.jsonl", r.returncode == 0 and len(subs) == 3, r.stderr[-500:])
    logs = [json.loads(x) for x in open(os.path.join(out, "logs", "log.jsonl"), encoding="utf-8")] if subs else []
    kiem("log có đủ trường id/answer/elapsed_ms/thinking/code/output",
         logs and all(all(k in lg for k in ("id", "answer", "elapsed_ms", "thinking", "code", "output")) for lg in logs))
    kiem("không câu nào bị lỗi chương trình", logs and all(not lg.get("error") for lg in logs),
         [lg.get("error", "")[-200:] for lg in logs if lg.get("error")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--nhanh", action="store_true", help="bỏ qua phần đọc dữ liệu shop thật")
    ap.add_argument("--so-shop", type=int, default=3)
    a = ap.parse_args()
    tests = [test_normalize, test_helpers, lambda: test_periods(a.data), test_loader_synthetic, test_sales_synthetic,
             lambda: test_scorer(a.data)]
    if not a.nhanh:
        tests.append(lambda: test_real_data(a.data, a.so_shop))
    import traceback
    for t in tests:
        try:
            t()
        except Exception:
            kiem("chạy hết nhóm kiểm tra", False, traceback.format_exc()[-600:])
    n_ok, n = sum(KET_QUA), len(KET_QUA)
    print(f"\nTỔNG: {n_ok}/{n} mục ĐẠT" + ("" if n_ok == n else f"  —  {n - n_ok} mục LỖI, xem các dòng 'LỖI' ở trên"))
    sys.exit(0 if n_ok == n else 1)


if __name__ == "__main__":
    main()
