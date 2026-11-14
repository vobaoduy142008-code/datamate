# -*- coding: utf-8 -*-
"""Chạy src/dsl_engine.py trên các cặp (câu train -> DSL) viết tay (vi_du_tay.jsonl), so với data/train/answers.jsonl bằng scorer.score_one.

    DATAMATE_CACHE=$HOME/dm_cache python3 training_code/dsl/kiem_engine.py [--out FILE] [--budget 40] [--reset] [--show]

- Cache biz: $HOME/lead/pk/train_<biz>.pkl = pickle (biz, ban_ok) (xem lead/multi.py); không có thì load_business rồi lưu.
- Có ngân sách thời gian (--budget giây): hết giờ thì dừng, chạy lại lệnh để tiếp (kết quả cộng dồn trong --out).
- Quy ước chấm: engine trả unanswerable -> pred {"type":"unanswerable"}; ok=False -> abstain (0), tính là "engine báo lỗi" chứ không phải sai.
"""
import argparse, json, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT + "/src")
os.environ.setdefault("DATAMATE_CACHE", os.path.expanduser("~/dm_cache"))
from periods import Calendar            # noqa: E402
from scorer import score_one            # noqa: E402
import dsl_engine                       # noqa: E402


def load_biz(bid):
    fp = os.path.expanduser(f"~/lead/pk/train_{bid}.pkl")
    if os.path.exists(fp):
        return pickle.load(open(fp, "rb"))[0]
    from loader import load_business
    z = load_business(f"{ROOT}/data/train/data/{bid}")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    try:
        from sales import ban_health
        pickle.dump((z, ban_health(z)[0]), open(fp, "wb"))
    except Exception:
        pass
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi_du", default=HERE + "/vi_du_tay.jsonl")
    ap.add_argument("--out", default=os.path.expanduser("~/tmp_dsl/ket_qua_kiem.jsonl"))
    ap.add_argument("--budget", type=float, default=40)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.reset and os.path.exists(a.out):
        os.remove(a.out)
    ex = [json.loads(l) for l in open(a.vi_du, encoding="utf-8") if l.strip()]
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + "/data/train/questions.jsonl", encoding="utf-8")}
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + "/data/train/answers.jsonl", encoding="utf-8")}
    ev = {json.loads(l)["id"] for l in open(ROOT + "/data/eval300.jsonl", encoding="utf-8")}
    assert not any(e["id"] in ev for e in ex), "có câu thuộc eval300"
    rb = open(ROOT + "/data/train/rulebook.md", encoding="utf-8").read()
    cal = Calendar(ROOT + "/data/train/calendar.csv")
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    t0 = time.time()
    by = {}
    for e in ex:
        if e["id"] not in done:
            by.setdefault(qs[e["id"]]["business_id"], []).append(e)
    for bid, L in sorted(by.items()):
        if time.time() - t0 > a.budget:
            print("HET GIO — chạy lại để tiếp"); break
        biz = load_biz(bid)
        with open(a.out, "a", encoding="utf-8") as f:
            for e in L:
                q = qs[e["id"]]
                ts = time.time()
                r = dsl_engine.thuc_thi(e["dsl"], biz, rb, cal, q["as_of_date"])
                if r["ok"]:
                    pred = r["answer"]
                    sc = score_one(pred, gold[e["id"]])
                else:
                    pred, sc = None, None           # engine báo lỗi (không tính đúng/sai)
                f.write(json.dumps({"id": e["id"], "biz": bid, "q": q["question"], "dsl": e["dsl"], "ok": r["ok"], "pred": pred,
                                    "gold": {"type": gold[e["id"]]["type"], "answer": gold[e["id"]].get("answer")},
                                    "sc": sc, "loi": r.get("loi"), "mo_ta": r.get("mo_ta"), "sec": round(time.time() - ts, 2)},
                                   ensure_ascii=False, default=str) + "\n")
    res = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    n_ok = sum(1 for r in res if r["sc"] == 1)
    n_sai = sum(1 for r in res if r["sc"] == -1)
    n_loi = sum(1 for r in res if r["sc"] is None)
    print(f"Đã chạy {len(res)}/{len(ex)}: ĐÚNG {n_ok}  SAI {n_sai}  engine báo lỗi {n_loi}")
    if len(res) == len(ex):
        print(f"Tỉ lệ đúng = {n_ok}/{len(res)} = {n_ok / len(res):.1%}")
    for r in res:
        if r["sc"] != 1 or a.show:
            tag = "LOI " if r["sc"] is None else ("SAI " if r["sc"] == -1 else "ok  ")
            print(f"{tag}{r['id']} [{r['biz']}] {r['q'][:100]}")
            if r["sc"] is None:
                print("     loi:", (r["loi"] or "")[:300])
            else:
                print("     pred:", json.dumps(r["pred"], ensure_ascii=False)[:160], "| gold:", json.dumps(r["gold"], ensure_ascii=False)[:160])


if __name__ == "__main__":
    main()
