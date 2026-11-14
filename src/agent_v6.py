# -*- coding: utf-8 -*-
"""agent_v6 — agent do MODEL dẫn dắt (AGENT_MODE=v6). Logic nằm ở đây; agent.py chỉ gọi `giai(question, ctx)`.

Luồng một câu (docs/kien_truc_v6.md §1):
  M1  model hiểu câu -> {"hieu","dsl"}  (dsl.loai = tu_giai / khong_du_du_lieu cũng được)
  T   dsl_engine.thuc_thi. Lỗi -> M2 sửa DSL (tối đa V6_TOI_DA_SUA = 2 lần). OK -> M2 đọc kết quả: chấp nhận / sửa / chuyển tự giải
  tu_giai -> vòng LLM viết code hiện có (agent._solve_once mode "code": giữ sandbox + giới hạn bước)
  V   kiểm chứng ĐỘC LẬP (không quyết định thay model): pipeline công thức cũ (agent._solve_once mode "formula", không gọi LLM)
      -> doi_chieu in {khop, lech, khong_co} (hoặc "tat" khi V6_TAT_DOI_CHIEU=1)
  P   chính sách trả lời / bỏ qua: bảng src/chinh_sach_v6.json (do training_code/dsl/hieu_chinh.py hiệu chỉnh trên TRAIN),
      thiếu bảng -> quy tắc mặc định (quyet_dinh_mac_dinh)

Env ablation: V6_TAT_DOI_CHIEU=1 (không chạy công thức cũ) · V6_CHI_DSL=1 (tắt đường tu_giai) · V6_CHI_TU_GIAI=1 (bỏ M1/M2, chỉ code)
Khác: V6_DOC_KET_QUA=0 (bỏ bước M2 đọc kết quả khi engine OK) · V6_TOI_DA_SUA · V6_SO_VI_DU · V6_THINK=1 · V6_CHINH_SACH=<file>
Mock: LLM_BACKEND=mock -> model giả trả DSL rỗng (an toàn: bỏ qua); V6_MOCK_DSL=<jsonl {id,dsl}> -> trả DSL theo id (kiểm đường DSL).
Log: output[0] = "V6_JSON {...}" (một dòng JSON: hieu, dsl, kết quả engine, số lần sửa, doi_chieu, quyết định...) để đo đóng góp model.
Không có logic theo id / business_id / tên shop (id chỉ dùng cho mock + loại ví dụ trùng câu khỏi prompt).
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
import traceback

import llm
import dsl_engine
import v6_prompt as VP
from cong_cu import doc_json

HERE = os.path.dirname(os.path.abspath(__file__))
V6_TAG = "V6_JSON "


def _env(name, default=""):
    return os.environ.get(name, default)


# ============================================================================ JSON chịu lỗi
def _va_cut(t: str):
    """Thêm dấu đóng còn thiếu cho JSON bị cắt giữa chừng (model hết token)."""
    st, in_s, esc = [], False, False
    for c in t:
        if in_s:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_s = False
        elif c == '"':
            in_s = True
        elif c in "{[":
            st.append("}" if c == "{" else "]")
        elif c in "}]" and st:
            st.pop()
    return t + ('"' if in_s else "") + "".join(reversed(st))


def doc_json_chiu_loi(text):
    if not text:
        return None
    d = doc_json(text)
    if d is not None:
        return d
    t = re.sub(r"```(?:json)?", "", text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    d = doc_json(t)
    if d is not None:
        return d
    s, e = t.find("{"), t.rfind("}")
    if s >= 0:
        frag = t[s:e + 1] if e > s else t[s:]
        for f in (frag, _va_cut(re.sub(r",\s*$", "", frag))):
            for loader in (json.loads, ast.literal_eval):
                try:
                    v = loader(f)
                    if isinstance(v, dict):
                        return v
                except Exception:
                    pass
    return None


def _dict_hay_none(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    return v if isinstance(v, dict) and v else None


def tach_m1(text):
    """-> (hieu, dsl dict | None). Chịu cả trường hợp model trả thẳng DSL (có khoá 'loai')."""
    d = doc_json_chiu_loi(text)
    if not d:
        return "", None
    dsl = _dict_hay_none(d.get("dsl"))
    if dsl is not None:
        return str(d.get("hieu") or ""), dsl
    if "loai" in d:
        return "", d
    return str(d.get("hieu") or ""), None


def _bool(v, mac_dinh=None):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "có", "co", "1", "đúng", "dung"):
            return True
        if s in ("false", "no", "không", "khong", "0", "sai"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return mac_dinh


def tach_m2(text):
    """-> dict {hop_le, chap_nhan (True/False/None), ly_do, dsl_sua (dict|None), chuyen_tu_giai (bool)}."""
    d = doc_json_chiu_loi(text)
    if not d or not any(k in d for k in ("chap_nhan", "dsl_sua", "chuyen_tu_giai")):
        return {"hop_le": False, "chap_nhan": None, "ly_do": "", "dsl_sua": None, "chuyen_tu_giai": False}
    return {"hop_le": True, "chap_nhan": _bool(d.get("chap_nhan")), "ly_do": str(d.get("ly_do") or "")[:200],
            "dsl_sua": _dict_hay_none(d.get("dsl_sua")), "chuyen_tu_giai": bool(_bool(d.get("chuyen_tu_giai"), False))}


# ============================================================================ gọi model (thật / mock)
_MOCK = None


def _mock_dsl(qid):
    global _MOCK
    if _MOCK is None:
        _MOCK = {}
        f = _env("V6_MOCK_DSL")
        if f and os.path.exists(f):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    _MOCK[r["id"]] = r["dsl"]
    return _MOCK.get(qid, {})


def goi_model(messages, loai, bc, temperature=None, seed=None):
    """-> (text, thinking). loai: 'm1' | 'm2'. LLM_BACKEND=mock: model giả (xem docstring đầu file)."""
    if llm.BACKEND == "mock":
        if loai == "m1":
            return json.dumps({"hieu": "mock", "dsl": _mock_dsl(bc.qid)}, ensure_ascii=False), None
        return json.dumps({"chap_nhan": True, "ly_do": "mock", "dsl_sua": None, "chuyen_tu_giai": False}), None
    kw = {}
    if temperature is not None:
        kw["temperature"] = temperature
    if seed is not None:
        kw["seed"] = seed
    if _env("V6_THINK"):
        kw["think"] = _env("V6_THINK") == "1"
    return llm.chat(messages, **kw)


# ============================================================================ bối cảnh + các bước M1 / M2 / T
_KHO = None


def kho_mac_dinh():
    global _KHO
    if _KHO is None:
        _KHO = VP.nap_kho_vi_du()
    return _KHO


def khoi_dong():
    """Gọi ở tiến trình cha (trước khi fork từng câu) để nạp sẵn kho ví dụ."""
    kho_mac_dinh()


class BoiCanhV6:
    def __init__(self, qid, question, biz, rulebook, cal, as_of, kho=None, ban=None, het_gio=None):
        self.qid, self.question, self.biz, self.rulebook, self.cal, self.as_of = qid, question, biz, rulebook, cal, as_of
        self.kho, self.ban, self.het_gio = kho, ban, het_gio
        self._nc = None

    def ngu_canh(self):
        if self._nc is None:
            try:
                self._nc = VP.ngu_canh_shop(self.biz, self.ban)
            except Exception:
                self._nc = ""
        return self._nc


def buoc_m1(bc, temperature=None, seed=None, so_vi_du=None):
    k = VP.SO_VI_DU if so_vi_du is None else so_vi_du
    kho = bc.kho if bc.kho is not None else kho_mac_dinh()
    vd = VP.vi_du_gan(bc.question, kho, k=k, loai_tru_id=bc.qid)
    ms = VP.prompt_hieu_cau(bc.question, bc.as_of, bc.ngu_canh(), vd)
    text, th = goi_model(ms, "m1", bc, temperature, seed)
    hieu, dsl = tach_m1(text)
    return {"buoc": "m1", "messages": ms, "text": text, "thinking": th, "hieu": hieu, "dsl": dsl, "vi_du": [e.get("id") for e in vd]}


def buoc_m2(bc, hieu, dsl, kq, temperature=None, seed=None):
    ms = VP.prompt_doc_ket_qua(bc.question, hieu, dsl, kq, as_of=bc.as_of, ngu_canh=bc.ngu_canh())
    text, th = goi_model(ms, "m2", bc, temperature, seed)
    r = tach_m2(text)
    r.update(buoc="m2", messages=ms, text=text, thinking=th)
    return r


def chay_engine(dsl, bc):
    if not isinstance(dsl, dict):
        return {"ok": False, "answer": None, "mo_ta": "", "trung_gian": "", "loi": "Đặc tả phải là một đối tượng JSON có khoá loai."}
    try:
        return dsl_engine.thuc_thi(dsl, bc.biz, bc.rulebook, bc.cal, bc.as_of)
    except Exception as e:                    # engine không nên ném lỗi, nhưng đừng để sập cả câu
        return {"ok": False, "answer": None, "mo_ta": "", "trung_gian": "", "loi": f"Lỗi nội bộ engine: {type(e).__name__}: {e}"}


def chay_dsl(bc, doc_ket_qua=True, toi_da_sua=None, temperature=None, seed=None):
    """M1 -> engine -> (M2 sửa / đọc)*. -> dict:
       duong: dsl | khong_du | tu_giai | loi | tu_choi | het_gio ; hieu, dsl (cuối), dsl_dau, kq (kết quả engine cuối), so_sua,
       lich_su (list các bước M1/M2 kèm messages + text; sinh_du_lieu dùng), nhat_ky (list chuỗi), n_llm, thinking."""
    toi_da_sua = int(_env("V6_TOI_DA_SUA", "2")) if toi_da_sua is None else toi_da_sua
    out = {"duong": "loi", "hieu": "", "dsl": None, "dsl_dau": None, "kq": None, "so_sua": 0, "lich_su": [], "nhat_ky": [],
           "n_llm": 0, "thinking": [], "m2_cuoi": None, "ly_do": ""}
    nk, ls = out["nhat_ky"], out["lich_su"]

    def het():
        return bool(bc.het_gio and bc.het_gio())

    def ghi(r):
        ls.append(r)
        out["n_llm"] += 1
        if r.get("thinking"):
            out["thinking"].append(r["thinking"])

    if het():
        out["duong"], out["ly_do"] = "het_gio", "hết thời gian trước M1"
        return out
    r = buoc_m1(bc, temperature, seed)
    ghi(r)
    if r["dsl"] is None:                                  # JSON hỏng: thử lại 1 lần với lấy mẫu khác
        nk.append(f"M1 không ra JSON hợp lệ: {str(r['text'])[:150]!r} -> thử lại")
        if het():
            out["duong"], out["ly_do"] = "het_gio", "hết thời gian"
            return out
        r = buoc_m1(bc, 0.5, 11)
        ghi(r)
        if r["dsl"] is None:
            out["ly_do"] = "M1 không ra JSON hợp lệ (2 lần)"
            nk.append(out["ly_do"])
            return out
    hieu, dsl = r["hieu"], r["dsl"]
    out["hieu"], out["dsl_dau"] = hieu, dsl
    nk.append(f"M1: hieu={hieu!r}; dsl={json.dumps(dsl, ensure_ascii=False)}")
    so_sua = 0
    while True:
        out["dsl"], out["so_sua"] = dsl, so_sua
        loai = dsl.get("loai") if isinstance(dsl, dict) else None
        if loai == "tu_giai":
            out["duong"], out["ly_do"] = "tu_giai", "model chọn tu_giai"
            return out
        kq = chay_engine(dsl, bc)
        out["kq"] = kq
        if kq.get("ok"):
            nk.append(f"engine OK: {json.dumps(kq.get('answer'), ensure_ascii=False)[:160]} | {str(kq.get('mo_ta'))[:200]}")
            if kq.get("khong_tra_loi_duoc") or loai == "khong_du_du_lieu":
                out["duong"] = "khong_du"
                return out
            if not doc_ket_qua or so_sua >= toi_da_sua:
                out["duong"] = "dsl"
                if so_sua >= toi_da_sua:
                    nk.append("đã sửa tối đa -> nhận kết quả, không đọc lại")
                return out
            if het():
                out["duong"], out["ly_do"] = "het_gio", "hết thời gian trước M2"
                return out
            m = buoc_m2(bc, hieu, dsl, kq, temperature, seed)
            ghi(m)
            nk.append(f"M2 đọc: chap_nhan={m['chap_nhan']} chuyen_tu_giai={m['chuyen_tu_giai']} ly_do={m['ly_do']!r}")
            out["m2_cuoi"] = "chap_nhan" if m["chap_nhan"] else ("khong_hop_le" if not m["hop_le"] else "tu_choi")
            if not m["hop_le"] or m["chap_nhan"] is True or (m["chap_nhan"] is None and not m["dsl_sua"] and not m["chuyen_tu_giai"]):
                out["duong"] = "dsl"
                return out
            if m["chuyen_tu_giai"]:
                out["duong"], out["ly_do"] = "tu_giai", "M2 chuyển tự giải: " + m["ly_do"]
                return out
            if not m["dsl_sua"]:
                out["duong"], out["ly_do"] = "tu_choi", "M2 không chấp nhận nhưng không có bản sửa: " + m["ly_do"]
                return out
            so_sua += 1
            dsl = m["dsl_sua"]
            nk.append(f"M2 sửa (lần {so_sua}): {json.dumps(dsl, ensure_ascii=False)}")
            continue
        # engine báo lỗi
        nk.append(f"engine LỖI: {str(kq.get('loi'))[:250]}")
        if so_sua >= toi_da_sua:
            out["duong"], out["ly_do"] = "loi", "engine còn lỗi sau khi sửa tối đa: " + str(kq.get("loi"))[:120]
            return out
        if het():
            out["duong"], out["ly_do"] = "het_gio", "hết thời gian trước khi sửa"
            return out
        m = buoc_m2(bc, hieu, dsl, kq, temperature, seed)
        ghi(m)
        nk.append(f"M2 sửa lỗi: chuyen_tu_giai={m['chuyen_tu_giai']} ly_do={m['ly_do']!r} dsl_sua={json.dumps(m['dsl_sua'], ensure_ascii=False)}")
        so_sua += 1
        if m["chuyen_tu_giai"]:
            out["so_sua"] = so_sua
            out["duong"], out["ly_do"] = "tu_giai", "M2 chuyển tự giải sau lỗi engine: " + m["ly_do"]
            return out
        if not m["dsl_sua"]:
            out["so_sua"] = so_sua
            out["duong"], out["ly_do"] = "loi", "M2 không đưa được bản sửa: " + str(kq.get("loi"))[:120]
            return out
        dsl = m["dsl_sua"]


# ============================================================================ V: đối chiếu
def _fold_s(x):
    return VP.fold(x)


def so_khop(a, b):
    """Hai đáp án (dict type/answer) cho cùng một ý? Dùng bộ so khớp của scorer (không alias) + so chuỗi đã bỏ dấu."""
    if not isinstance(a, dict) or not isinstance(b, dict) or a.get("type") != b.get("type"):
        return False
    t = a["type"]
    if t == "unanswerable":
        return True
    try:
        import scorer
        if scorer.MATCHERS[t](a.get("answer"), b.get("answer")):
            return True
    except Exception:
        pass
    try:
        return json.dumps(a.get("answer"), ensure_ascii=False, sort_keys=True, default=str).lower() == \
            json.dumps(b.get("answer"), ensure_ascii=False, sort_keys=True, default=str).lower() or \
            _fold_s(json.dumps(a.get("answer"), ensure_ascii=False, default=str)) == _fold_s(json.dumps(b.get("answer"), ensure_ascii=False, default=str))
    except Exception:
        return False


# ============================================================================ P: chính sách
def dac_trung(duong, doi_chieu, so_sua, buoc, loai):
    """Đặc trưng rời rạc của một câu (dùng cho chính sách + hiệu chỉnh)."""
    return {"duong": duong, "doi_chieu": doi_chieu,
            "so_sua": str(min(int(so_sua or 0), 2)) if duong in ("dsl", "khong_du") else "-",
            "buoc": (str(buoc) if int(buoc or 0) < 3 else "3+") if duong == "tu_giai" else "-",
            "loai": (loai or duong)}


_MUC = (("duong", "doi_chieu", "so_sua", "buoc", "loai"), ("duong", "doi_chieu", "so_sua", "buoc"), ("duong", "doi_chieu"))


def khoa_o(dt, muc):
    """Khoá ô của đặc trưng ở mức muc (0 = chi tiết nhất ... 2 = thô nhất)."""
    return "|".join(f"{k}={dt[k]}" for k in _MUC[muc])


def dac_trung_tu_meta(meta):
    return dac_trung(meta.get("duong"), meta.get("doi_chieu"), meta.get("so_sua"), meta.get("buoc"), meta.get("loai"))


def quyet_dinh_mac_dinh(dt):
    """Quy tắc mặc định khi chưa có bảng hiệu chỉnh. -> (tra_loi | bo_qua, lý do)."""
    d, c = dt["duong"], dt["doi_chieu"]
    if d == "dsl":
        return ("bo_qua", "dsl & công thức cũ lệch nhau") if c == "lech" else ("tra_loi", f"dsl & doi_chieu={c}")
    if d == "khong_du":
        return ("bo_qua", "model nói không đủ dữ liệu nhưng công thức cũ có đáp án") if c == "lech" else ("tra_loi", f"khong_du & doi_chieu={c}")
    if d == "tu_giai":
        if c == "lech":
            return "bo_qua", "tu_giai & công thức cũ lệch nhau"
        return ("tra_loi", "tu_giai & code chạy ngay lượt đầu") if dt["buoc"] in ("0", "1") else ("bo_qua", f"tu_giai cần {dt['buoc']} lượt")
    return "bo_qua", f"đường {d}"


_BANG = {"path": None, "mtime": None, "o": {}, "co": False}


def _nap_bang():
    p = _env("V6_CHINH_SACH") or os.path.join(HERE, "chinh_sach_v6.json")
    try:
        mt = os.path.getmtime(p)
    except OSError:
        _BANG.update(path=p, mtime=None, o={}, co=False)
        return _BANG
    if _BANG["path"] != p or _BANG["mtime"] != mt:
        try:
            d = json.load(open(p, encoding="utf-8"))
            o = {c["khoa"]: c for c in d.get("o", []) if c.get("quyet_dinh") in ("tra_loi", "bo_qua")}
            _BANG.update(path=p, mtime=mt, o=o, co=True)
        except Exception:
            _BANG.update(path=p, mtime=mt, o={}, co=False)
    return _BANG


def quyet_dinh(dt):
    """-> (tra_loi | bo_qua, nguồn). Tra bảng từ ô chi tiết nhất; không có ô -> quy tắc mặc định."""
    if dt["duong"] not in ("dsl", "khong_du", "tu_giai"):
        return "bo_qua", f"đường {dt['duong']}"
    b = _nap_bang()
    if b["co"]:
        for muc in (0, 1, 2):
            c = b["o"].get(khoa_o(dt, muc))
            if c:
                return c["quyet_dinh"], f"bang:{c['khoa']} (n={c.get('n')}, dung={c.get('dung')}, sai={c.get('sai')})"
    h, ly = quyet_dinh_mac_dinh(dt)
    return h, "mac_dinh: " + ly


# ============================================================================ điều phối cả câu
def _gon_json(o, n=400):
    s = json.dumps(o, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "..."


def _rong_hay_khong(ans):
    return isinstance(ans, dict) and ((ans.get("type") in ("number", "percent") and ans.get("answer") == 0)
                                      or (ans.get("type") == "list" and not ans.get("answer")))


def giai(question, ctx):
    """question = {"id","business_id","question","as_of_date"}; ctx do agent.py dựng:
         biz, ban, rulebook, cal, as_of, het_gio(), abstain_zero, normalize(ans)->(ans|None, loi),
         giai_code(hint)->dict kết quả _solve_once (mode code), giai_cong_thuc()->(answer|None, mo_ta).
       -> dict như agent._solve_once: answer / thinking / code / output / steps / seconds (+ v6: meta)."""
    t0 = time.time()
    chi_dsl, chi_tg, tat_dc = _env("V6_CHI_DSL") == "1", _env("V6_CHI_TU_GIAI") == "1", _env("V6_TAT_DOI_CHIEU") == "1"
    bc = BoiCanhV6(question.get("id"), question["question"], ctx["biz"], ctx["rulebook"], ctx["cal"], ctx["as_of"],
                   ban=ctx.get("ban"), het_gio=ctx.get("het_gio"))
    meta = {"duong": "loi", "loai": None, "hieu": "", "dsl_dau": None, "dsl": None, "engine_ok": None, "mo_ta": "", "loi_engine": "",
            "so_sua": 0, "m2": None, "buoc": 0, "doi_chieu": "tat" if tat_dc else "khong_co", "ans_model": None, "ans_ct": None,
            "ct_mo_ta": "", "quyet_dinh": "bo_qua", "quyet_nguon": "", "n_llm": 0, "ly_do": "", "chi_dsl": chi_dsl, "chi_tu_giai": chi_tg,
            "tat_doi_chieu": tat_dc}
    thinking, nhat_ky, code_parts, outputs = [], [], [], []
    code_res, steps = None, 0
    try:
        if chi_tg:
            meta["duong"], meta["ly_do"] = "tu_giai", "V6_CHI_TU_GIAI=1"
            dsl_r = None
        else:
            dsl_r = chay_dsl(bc, doc_ket_qua=_env("V6_DOC_KET_QUA", "1") != "0")
            nhat_ky += dsl_r["nhat_ky"]
            thinking += dsl_r["thinking"]
            meta.update(duong=dsl_r["duong"], hieu=dsl_r["hieu"], dsl_dau=dsl_r["dsl_dau"], dsl=dsl_r["dsl"], so_sua=dsl_r["so_sua"],
                        m2=dsl_r["m2_cuoi"], n_llm=dsl_r["n_llm"], ly_do=dsl_r["ly_do"])
            kq = dsl_r["kq"]
            if kq is not None:
                meta.update(engine_ok=bool(kq.get("ok")), mo_ta=str(kq.get("mo_ta") or "")[:300],
                            loi_engine=str(kq.get("loi") or "")[:200])
            if isinstance(dsl_r["dsl"], dict):
                meta["loai"] = dsl_r["dsl"].get("loai")
        route = meta["duong"]
        ans_model = None
        if route in ("dsl", "khong_du"):
            a, problem = ctx["normalize"](dsl_r["kq"]["answer"])
            if a is None:
                meta["duong"], meta["ly_do"] = "loi", "đáp án engine sai định dạng: " + str(problem)
            else:
                ans_model = a
        elif route == "tu_giai":
            if chi_dsl:
                meta["ly_do"] = (meta["ly_do"] + "; " if meta["ly_do"] else "") + "V6_CHI_DSL=1: tắt đường tu_giai"
            else:
                hint = (f"(Gợi ý từ bước hiểu câu, có thể sai: {meta['hieu']})" if meta["hieu"] else "")
                code_res = ctx["giai_code"](hint)
                steps = int(code_res.get("steps") or 0)
                meta["buoc"] = steps
                thinking += list(code_res.get("thinking") or [])
                code_parts.append(str(code_res.get("code") or ""))
                outputs += [f"[code] {o}" for o in (code_res.get("output") or [])][-6:]
                a = code_res.get("answer") or {"type": "abstain"}
                if a.get("type") not in ("abstain", None):
                    ans_model = a
        meta["ans_model"] = ans_model
        # ---- V: đối chiếu độc lập với công thức cũ
        if ans_model is not None and not tat_dc:
            try:
                a_ct, ct_mo_ta = ctx["giai_cong_thuc"]()
            except Exception as e:
                a_ct, ct_mo_ta = None, f"lỗi công thức: {e!r}"
            meta["ans_ct"], meta["ct_mo_ta"] = a_ct, str(ct_mo_ta)[:200]
            meta["doi_chieu"] = "khong_co" if a_ct is None else ("khop" if so_khop(ans_model, a_ct) else "lech")
        # ---- P: chính sách
        final = {"type": "abstain"}
        if ans_model is not None:
            dt = dac_trung(meta["duong"], meta["doi_chieu"], meta["so_sua"], meta["buoc"], meta["loai"] or meta["duong"])
            hanh_dong, nguon = quyet_dinh(dt)
            if hanh_dong == "tra_loi" and ctx.get("abstain_zero") and _rong_hay_khong(ans_model):
                hanh_dong, nguon = "bo_qua", nguon + " | đáp án rỗng/bằng 0"
            meta["quyet_dinh"], meta["quyet_nguon"] = hanh_dong, nguon
            if hanh_dong == "tra_loi":
                final = ans_model
        else:
            meta["quyet_nguon"] = "không có đáp án của model: " + (meta["ly_do"] or meta["duong"])
    except Exception:
        meta["ly_do"] = "lỗi v6: " + traceback.format_exc()[-500:]
        final = {"type": "abstain"}
        meta["quyet_dinh"] = "bo_qua"
    meta["giay"] = round(time.time() - t0, 1)
    nhat_ky.append(f"V (đối chiếu công thức cũ): {meta['doi_chieu']} | model={_gon_json(meta['ans_model'], 160)} | công thức={_gon_json(meta['ans_ct'], 160)} {meta['ct_mo_ta']}")
    nhat_ky.append(f"P (chính sách): {meta['quyet_dinh']} <- {meta['quyet_nguon']}")
    code = "agent_v6 | đường=%s | hieu=%s\n" % (meta["duong"], meta["hieu"]) + "\n".join(nhat_ky)
    if code_parts:
        code += "\n\n# ----- code do model viết (tu_giai) -----\n" + "\n# ---\n".join(code_parts)
    return {"answer": final, "thinking": thinking or None, "code": code,
            "output": [V6_TAG + json.dumps(meta, ensure_ascii=False, default=str)] + outputs,
            "steps": steps, "seconds": meta["giay"], "v6": meta}
