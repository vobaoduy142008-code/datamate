#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh dữ liệu SFT CÓ KIỂM CHỨNG cho v6 (chạy trên Mac với Ollama; model gốc = LLM_MODEL, mặc định qwen3:8b).

Với mỗi câu TRAIN không thuộc eval300:
  dựng prompt M1 bằng v6_prompt -> model gốc sinh {"hieu","dsl"} -> engine thực thi (lỗi -> M2 sửa tối đa 2 lần, như lúc chạy thật)
  -> so data/train/answers.jsonl bằng scorer.score_one. ĐÚNG -> ghi ví dụ SFT. SAI -> thử lại 1 lần (temperature 0.6, seed khác);
  vẫn sai -> that_bai.jsonl (KHÔNG dùng huấn luyện).

Các loại ví dụ ghi vào sft_raw.jsonl (trường loai_vd):
  m1_tay   câu trong vi_du_tay.jsonl, đáp án = DSL viết tay (hieu dựng từ DSL)
  m1       M1 của model gốc, DSL đã kiểm chứng đúng (nếu DSL phải qua M2 sửa thì nhãn = DSL cuối đã đúng)
  m2_sua   lượt 2: prompt có LỖI engine -> model sửa ra DSL đúng (chỉ lấy bước sửa CUỐI, đã kiểm chứng)
  m2_nhan  lượt 2: kết quả đúng được model chấp nhận (lấy mẫu --m2-ty-le số câu đúng; đo luôn tỉ lệ M2 từ chối nhầm câu đúng)
  tu_giai  câu engine không làm được (model sinh DSL thất bại / chọn tu_giai) mà log cũ (outputs/*/logs) có lời giải bằng
           code đúng theo answers.jsonl, đường LLM viết code, <= 2 lượt -> nhãn {"hieu":..,"dsl":{"loai":"tu_giai"}} (tối đa --ty-le-tu-giai số ví dụ DSL)
Định dạng dòng: {"id","biz","loai_vd","messages":[system,user,assistant]} — đúng thứ chuan_bi_du_lieu.py đọc.

Resume: sinh_tien_do.jsonl ghi từng câu đã xử lý (chạy lại bỏ qua). Tiến độ + ETA in từng câu.
Chạy (từ thư mục dự án):  .venv/bin/python training_code/dsl/sinh_du_lieu.py [--limit N] [--shops biz_101,biz_102]
Hoặc dùng chay_sinh_du_lieu.command (chép code vào thư mục riêng, caffeinate, ngủ máy khi xong).
Đầu ra cạnh sft_raw.jsonl: that_bai.jsonl, vi_du_sinh.jsonl (ví dụ đã kiểm chứng, dùng làm few-shot cho câu sau).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def doc_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def ghi_dong(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    f.flush()


def hhmm(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def ban_cua(biz):
    ban = getattr(biz, "ban", None)
    if ban is None and not getattr(biz, "_ban_tried", False):
        from sales import build_sales
        biz.ban = ban = build_sales(biz)
        biz._ban_tried = True
    return ban


def chi_muc_log_cu(root, gold, cam):
    """id -> {"steps","nguon"}: câu có log cũ do LLM viết code (không phải công thức) và ĐÚNG theo gold."""
    from scorer import score_one
    res = {}
    mau = [os.path.join(root, "outputs", "*", "logs", "log.jsonl"), os.path.join(root, "outputs", "*", "logs", "q_*.json")]
    for pat in mau:
        for p in sorted(glob.glob(pat)):
            try:
                lines = open(p, encoding="utf-8").read().splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = d.get("id")
                if qid not in gold or qid in cam or qid in res:
                    continue
                code = d.get("code") or ""
                if not isinstance(code, str) or code.startswith(("cong_thuc", "ct_", "dieu_phoi", "agent")) or "ANSWER" not in code:
                    continue
                steps = d.get("steps") or 0
                ans = d.get("answer")
                if not isinstance(ans, dict) or steps < 1 or steps > 2 or score_one(ans, gold[qid]) != 1:
                    continue
                res[qid] = {"steps": steps, "nguon": os.path.relpath(p, root)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..")), help="thư mục dự án (có data/, outputs/)")
    ap.add_argument("--src", default=None, help="thư mục code (agent.py, v6_prompt.py...); mặc định <root>/src")
    ap.add_argument("--out", default=None, help="sft_raw.jsonl (mặc định <root>/training_code/dsl/sft_raw.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="tổng số câu tối đa (chia đều cho các shop); 0 = hết")
    ap.add_argument("--shops", default="", help="danh sách shop, cách nhau dấu phẩy (vd biz_101,biz_102)")
    ap.add_argument("--so-vi-du", type=int, default=None, help="số ví dụ few-shot trong prompt M1 (mặc định V6_SO_VI_DU / 6)")
    ap.add_argument("--m2-ty-le", type=float, default=0.3, help="tỉ lệ câu đúng cho M2 đọc để lấy ví dụ chap_nhan (0 = tắt)")
    ap.add_argument("--nhiet-do-lai", type=float, default=0.6, help="temperature của lần thử lại")
    ap.add_argument("--ty-le-tu-giai", type=float, default=0.15, help="số ví dụ tu_giai tối đa = tỉ lệ này x số ví dụ DSL")
    ap.add_argument("--khong-tu-giai", action="store_true", help="không tạo ví dụ tu_giai từ log cũ")
    ap.add_argument("--chi-tay", action="store_true", help="chỉ ghi ví dụ viết tay (không gọi model)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    src = os.path.abspath(a.src) if a.src else os.path.join(root, "src")
    sys.path.insert(0, src)
    os.environ.setdefault("DATAMATE_CACHE", os.path.expanduser("~/dm_cache"))
    import agent                                   # noqa: E402  (nạp dữ liệu shop, rulebook)
    import agent_v6 as A                           # noqa: E402
    import v6_prompt as VP                         # noqa: E402
    from scorer import score_one                   # noqa: E402
    import llm                                     # noqa: E402
    if a.so_vi_du is not None:
        VP.SO_VI_DU = a.so_vi_du

    data = os.path.join(root, "data", "train")
    out_path = a.out or os.path.join(root, "training_code", "dsl", "sft_raw.jsonl")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    p_bai, p_sinh, p_tien = (os.path.join(out_dir, n) for n in ("that_bai.jsonl", "vi_du_sinh.jsonl", "sinh_tien_do.jsonl"))

    questions = {r["id"]: r for r in doc_jsonl(os.path.join(data, "questions.jsonl"))}
    gold = {r["id"]: r for r in doc_jsonl(os.path.join(data, "answers.jsonl"))}
    cam = {r["id"] for r in doc_jsonl(os.path.join(root, "data", "eval300.jsonl"))}
    if not cam:
        sys.exit("Không đọc được data/eval300.jsonl -> dừng để tránh dùng nhầm câu eval300.")
    giu = {r["id"] for r in doc_jsonl(os.path.join(root, "data", "giu_rieng.jsonl"))}      # tập giữ riêng (training_code/dsl/tao_giu_rieng.py)
    if giu:
        print(f"Loại thêm {len(giu)} câu của data/giu_rieng.jsonl (tập giữ riêng để hiệu chỉnh / đánh giá).")
        cam |= giu
    p_tay = os.path.join(root, "training_code", "dsl", "vi_du_tay.jsonl")
    tay = {r["id"]: r for r in doc_jsonl(p_tay if os.path.exists(p_tay) else os.path.join(HERE, "vi_du_tay.jsonl"))}
    assert not (set(tay) & cam), "vi_du_tay có câu thuộc eval300"
    rulebook, cal = agent._static(data)
    files_kho = [os.path.join(src, "vi_du_v6.jsonl"), os.path.join(src, "vi_du_sinh_v6.jsonl"),
                 os.path.join(root, "training_code", "dsl", "vi_du_tay.jsonl"), p_sinh]
    kho = VP.nap_kho_vi_du(files=[p for p in files_kho if os.path.exists(p)])
    print(f"Kho ví dụ few-shot: {len(kho)} (loại {kho.so_bi_loai} ví dụ thuộc eval300/public). Model: {llm.MODEL} ({llm.BACKEND})")

    tien = {}
    for r in doc_jsonl(p_tien):
        tien[r["id"]] = r["kq"]
    sft_co = {(r.get("id"), r.get("loai_vd")) for r in doc_jsonl(out_path)}
    n_dsl_vd = sum(1 for k in sft_co if k[1] in ("m1", "m1_tay"))
    n_tg_vd = sum(1 for k in sft_co if k[1] == "tu_giai")

    # ---- danh sách câu cần xử lý, theo shop
    theo_shop = defaultdict(list)
    for qid, q in sorted(questions.items()):
        if qid in cam or qid not in gold:
            continue
        theo_shop[q["business_id"]].append(q)
    shops = sorted(theo_shop)
    if a.shops:
        want = {s.strip() for s in a.shops.split(",") if s.strip()}
        shops = [s for s in shops if s in want]
    if a.limit:
        moi_shop = max(1, -(-a.limit // max(1, len(shops))))
        for s in shops:
            random.Random(a.seed).shuffle(theo_shop[s])
            theo_shop[s] = theo_shop[s][:moi_shop]
    can_lam = {s: [q for q in theo_shop[s] if q["id"] not in tien and q["id"] not in tay] for s in shops}
    tong = sum(len(v) for v in can_lam.values())
    print(f"{len(shops)} shop; còn {tong} câu cần xử lý (đã xong {sum(1 for s in shops for q in theo_shop[s] if q['id'] in tien)}); "
          f"ví dụ viết tay: {len(tay)}")

    log_cu = {}
    if not a.khong_tu_giai and not a.chi_tay:
        log_cu = chi_muc_log_cu(root, gold, cam)
        print(f"Log cũ có lời giải code đúng (đường LLM, <= 2 lượt, không eval300): {len(log_cu)} câu")

    f_sft, f_bai, f_sinh, f_tien = (open(p, "a", encoding="utf-8") for p in (out_path, p_bai, p_sinh, p_tien))
    st = Counter()
    m2_nhan_stat = Counter()
    t_start, da_lam, loi_lien_tiep = time.time(), 0, 0

    def sft(qid, bid, loai_vd, messages, tra_loi):
        nonlocal n_dsl_vd, n_tg_vd
        if (qid, loai_vd) in sft_co:
            return False
        ghi_dong(f_sft, {"id": qid, "biz": bid, "loai_vd": loai_vd,
                         "messages": messages + [{"role": "assistant", "content": tra_loi}]})
        sft_co.add((qid, loai_vd))
        st["ví dụ " + loai_vd] += 1
        if loai_vd in ("m1", "m1_tay"):
            n_dsl_vd += 1
        if loai_vd == "tu_giai":
            n_tg_vd += 1
        return True

    def thu(bc, nhiet=None, seed=None):
        """Một lượt M1 + sửa lỗi; -> (chuẩn kết quả, điểm)."""
        r = A.chay_dsl(bc, doc_ket_qua=False, temperature=nhiet, seed=seed)
        sc, pred = None, None
        if r["duong"] in ("dsl", "khong_du") and r["kq"] and r["kq"].get("ok"):
            pred = r["kq"]["answer"]
            sc = score_one(pred, gold[bc.qid])
        r["sc"], r["pred"] = sc, pred
        return r

    for bid in shops:
        qs = can_lam[bid]
        hand = [e for e in tay.values() if questions.get(e["id"], {}).get("business_id") == bid]
        cho_tg = [qid for qid, k in tien.items() if k in ("fail", "tu_giai_chon") and questions.get(qid, {}).get("business_id") == bid
                  and (qid, "tu_giai") not in sft_co]
        if not qs and not cho_tg and not [e for e in hand if (e["id"], "m1_tay") not in sft_co]:
            continue
        try:
            biz = agent._business(data, bid)
            ban = ban_cua(biz)
        except Exception as e:
            print(f"[{bid}] không nạp được dữ liệu shop: {e!r} -> bỏ qua shop")
            continue

        def mk_bc(q):
            return A.BoiCanhV6(q["id"], q["question"], biz, rulebook, cal, q["as_of_date"], kho=kho, ban=ban)

        # --- ví dụ viết tay của shop này
        for e in hand:
            if (e["id"], "m1_tay") in sft_co:
                continue
            q = questions[e["id"]]
            bc = mk_bc(q)
            ms = VP.prompt_hieu_cau(q["question"], q["as_of_date"], bc.ngu_canh(),
                                    VP.vi_du_gan(q["question"], kho, k=VP.SO_VI_DU, loai_tru_id=q["id"]))
            sft(q["id"], bid, "m1_tay", ms, VP.tra_loi_m1(e.get("hieu") or VP.hieu_tu_dsl(e["dsl"]), e["dsl"]))
        if a.chi_tay:
            continue
        # --- câu của shop
        for i, q in enumerate(qs, 1):
            qid = q["id"]
            t_q = time.time()
            bc = mk_bc(q)
            try:
                r = thu(bc)
                r_dau = None
                if r["sc"] != 1:                      # sai / lỗi / tu_giai -> thử lại 1 lần, lấy mẫu khác
                    r_dau, r = r, thu(bc, a.nhiet_do_lai, 7)
                loi_lien_tiep = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                loi_lien_tiep += 1
                print(f"   LỖI khi xử lý {qid}: {e!r}")
                if loi_lien_tiep >= 5:
                    sys.exit("5 lỗi liên tiếp (Ollama có chạy không? LLM_MODEL có tồn tại không?) -> dừng. Chạy lại để tiếp tục.")
                continue
            da_lam += 1
            if r["sc"] == 1:
                dsl, hieu = r["dsl"], r["hieu"] or VP.hieu_tu_dsl(r["dsl"])
                lan_m1 = next((x for x in r["lich_su"] if x["buoc"] == "m1" and x["dsl"] is not None), None)
                if lan_m1 is not None:
                    sft(qid, bid, "m1", lan_m1["messages"], VP.tra_loi_m1(hieu, dsl))
                m2s = [x for x in r["lich_su"] if x["buoc"] == "m2"]
                if m2s and m2s[-1].get("dsl_sua") == dsl:
                    sft(qid, bid, "m2_sua", m2s[-1]["messages"],
                        VP.tra_loi_m2(False, m2s[-1]["ly_do"] or "Sửa đặc tả theo lỗi của công cụ", dsl, False))
                if a.m2_ty_le > 0 and r["kq"] and r["kq"].get("ok") and not r["kq"].get("khong_tra_loi_duoc") \
                        and int(hashlib.md5(qid.encode()).hexdigest(), 16) % 1000 < a.m2_ty_le * 1000:
                    try:
                        m = A.buoc_m2(bc, hieu, dsl, r["kq"])
                        if m["hop_le"] and m["chap_nhan"] is True and not m["dsl_sua"] and not m["chuyen_tu_giai"]:
                            m2_nhan_stat["chấp nhận câu đúng"] += 1
                            sft(qid, bid, "m2_nhan", m["messages"],
                                VP.tra_loi_m2(True, m["ly_do"] or "Kết quả khớp điều câu hỏi", None, False))
                        else:
                            m2_nhan_stat["TỪ CHỐI/sửa nhầm câu đúng"] += 1
                    except Exception as e:
                        print(f"   (M2 đọc lỗi: {e!r})")
                kho.them({"id": qid, "question": q["question"], "dsl": dsl, "hieu": hieu}) and \
                    ghi_dong(f_sinh, {"id": qid, "question": q["question"], "dsl": dsl, "hieu": hieu})
                kq_txt = "ok"
                st["đúng"] += 1
                tag = f"ĐÚNG (sửa {r['so_sua']}{', lần 2' if r_dau is not None else ''})"
            else:
                chon_tg = r["duong"] == "tu_giai" or (r_dau is not None and r_dau["duong"] == "tu_giai")
                kq_txt = "tu_giai_chon" if chon_tg else "fail"
                ghi_dong(f_bai, {"id": qid, "biz": bid, "question": q["question"],
                                 "gold": {"type": gold[qid]["type"], "answer": gold[qid].get("answer")},
                                 "thu": [{"duong": x["duong"], "dsl": x["dsl"], "pred": x["pred"], "sc": x["sc"],
                                          "so_sua": x["so_sua"], "loi": (x["kq"] or {}).get("loi") or x["ly_do"]}
                                         for x in ([r_dau] if r_dau is not None else []) + [r]]})
                st["thất bại"] += 1
                tag = "thất bại" + (" (chọn tu_giai)" if chon_tg else "")
            ghi_dong(f_tien, {"id": qid, "kq": kq_txt})
            tien[qid] = kq_txt
            con = tong - da_lam
            eta = (time.time() - t_start) / max(1, da_lam) * con
            print(f"[{da_lam}/{tong}] {bid} {qid} {tag} {time.time() - t_q:.0f}s | đúng {st['đúng']}/{da_lam} | ETA {hhmm(eta)}", flush=True)

        # --- ví dụ tu_giai của shop (log cũ có code đúng cho câu model/engine không làm được)
        if log_cu:
            cho_tg = [qid for qid, k in tien.items() if k in ("fail", "tu_giai_chon") and questions.get(qid, {}).get("business_id") == bid
                      and (qid, "tu_giai") not in sft_co and qid in log_cu]
            for qid in cho_tg:
                if n_tg_vd >= a.ty_le_tu_giai * max(1, n_dsl_vd):
                    break
                q = questions[qid]
                bc = mk_bc(q)
                ms = VP.prompt_hieu_cau(q["question"], q["as_of_date"], bc.ngu_canh(),
                                        VP.vi_du_gan(q["question"], kho, k=VP.SO_VI_DU, loai_tru_id=qid))
                dsl = {"loai": "tu_giai"}
                sft(qid, bid, "tu_giai", ms, VP.tra_loi_m1(VP.hieu_tu_dsl(dsl), dsl))

    for f in (f_sft, f_bai, f_sinh, f_tien):
        f.close()
    print("\n=== XONG ===")
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    if m2_nhan_stat:
        print("  M2 đọc kết quả ĐÚNG ->", dict(m2_nhan_stat))
    print(f"Tổng ví dụ trong {out_path}: {sum(1 for _ in open(out_path, encoding='utf-8'))}. Thất bại: {p_bai}. Ví dụ đã kiểm chứng: {p_sinh}")
    print("Bước tiếp: .venv/bin/python training_code/finetune/chuan_bi_du_lieu.py")


if __name__ == "__main__":
    main()
