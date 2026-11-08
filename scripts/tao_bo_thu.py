"""Tạo một bộ /data NHỎ từ dữ liệu train để chạy thử image Docker giống hệt lúc chấm (mount vào /data).
Cách chạy:  python3 scripts/tao_bo_thu.py data/train ~/bo_thu50 --n 50 --seed 13
Kết quả:   ~/bo_thu50/data/<shop>/..., rulebook.md, calendar.csv, questions.jsonl
           ~/bo_thu50_dapan.jsonl  (đáp án, để NGOÀI thư mục /data, dùng chấm điểm sau khi chạy)
Chọn câu giống hệt  main.py --sample N --seed S  nên so được với kết quả chạy trên Mac cùng seed."""
import argparse, json, os, random, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    qs = [json.loads(l) for l in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8") if l.strip()]
    pick = random.Random(a.seed).sample(qs, min(a.n, len(qs)))
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for f in ("rulebook.md", "calendar.csv"):
        shutil.copy(os.path.join(a.train, f), out)
    for b in sorted({q["business_id"] for q in pick}):
        dst = os.path.join(out, "data", b)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(a.train, "data", b), dst)
    with open(os.path.join(out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in pick:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    ids = {q["id"] for q in pick}
    ans = os.path.join(a.train, "answers.jsonl")
    if os.path.exists(ans):
        with open(out.rstrip("/") + "_dapan.jsonl", "w", encoding="utf-8") as f:
            for l in open(ans, encoding="utf-8"):
                if l.strip() and json.loads(l)["id"] in ids:
                    f.write(l if l.endswith("\n") else l + "\n")
    print(f"{out}: {len(pick)} câu, {len({q['business_id'] for q in pick})} shop; đáp án: {out.rstrip('/')}_dapan.jsonl")


if __name__ == "__main__":
    main()
