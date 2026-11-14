#!/usr/bin/env python3
"""Kiểm nhanh một model Ollama: trả lời 1-3 câu (từ data/valid.jsonl) ở chế độ think=false, theo định dạng JSON.
Mã thoát 0 = đạt (JSON hợp lệ khi đáp án mẫu là JSON, không rò thẻ <think>), 1 = không đạt.
Dùng:  .venv/bin/python kiem_model_ollama.py <model> [--so-voi qwen3:8b] [--n 3] [--data data/valid.jsonl]"""
import argparse
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:11434"


def chat(model, messages, think=False, n_pred=400):
    body = {"model": model, "stream": False, "think": think, "messages": messages,
            "options": {"temperature": 0, "seed": 0, "num_predict": n_pred, "num_ctx": 4096}}
    req = urllib.request.Request(URL + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=600))["message"]
    return m.get("content", "") or "", m.get("thinking", "") or ""


def lay_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--so-voi", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--data", default="data/valid.jsonl")
    a = ap.parse_args()
    try:
        urllib.request.urlopen(URL + "/api/tags", timeout=5)
    except Exception as e:
        print("Ollama chưa chạy (mở app Ollama hoặc `ollama serve`):", e)
        return 1
    mau = []
    try:
        for i, l in enumerate(open(a.data, encoding="utf-8")):
            if i >= a.n:
                break
            ms = json.loads(l)["messages"]
            mau.append((ms[:-1], ms[-1]["content"]))
    except FileNotFoundError:
        pass
    if not mau:
        mau = [([{"role": "user", "content": 'Trả lời CHỈ bằng JSON: {"ket_qua": <tổng 17 + 25>}'}], '{"ket_qua": 42}')]
    ok_all, khac = True, 0
    for k, (ms, dap_an) in enumerate(mau, 1):
        t = time.time()
        try:
            c, th = chat(a.model, ms)
        except Exception as e:
            print(f"[{k}] LỖI gọi model: {e}")
            return 1
        cần_json = lay_json(dap_an) is not None
        j = lay_json(c)
        leak = ("<think>" in c) or ("</think>" in c) or bool(th)
        ok = (j is not None or not cần_json) and not leak and bool(c.strip())
        ok_all &= ok
        print(f"[{k}] {'ĐẠT' if ok else 'KHÔNG ĐẠT'} ({time.time()-t:.1f}s) json={'có' if j is not None else 'không'} rò_think={leak}\n     model: {c.strip()[:160]!r}\n     mẫu  : {dap_an.strip()[:160]!r}")
        if a.so_voi:
            cb, _ = chat(a.so_voi, ms)
            khac += cb.strip() != c.strip()
    if a.so_voi:
        print(f"Khác model gốc {a.so_voi}: {khac}/{len(mau)} câu (0/{len(mau)} nghĩa là adapter có thể KHÔNG được áp dụng - nhưng adapter smoke 10 bước có thể quá yếu để thấy khác biệt)")
    print("KẾT QUẢ:", "ĐẠT" if ok_all else "KHÔNG ĐẠT")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
