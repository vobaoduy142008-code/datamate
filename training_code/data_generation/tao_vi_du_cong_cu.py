"""Tạo src/vi_du_cong_cu.jsonl: ví dụ "câu hỏi -> công cụ đúng" cho bước 1 của chế độ AGENT_MODE=dieu_phoi.

Cách làm: chạy các module công thức (đúng thứ tự ưu tiên của agent) trên câu TRAIN; chỉ giữ câu mà module trả lời ĐÚNG
theo data/train/answers.jsonl (src/scorer.py::score_one). Công cụ = nhãn của module đó (cong_cu.nhan_cong_cu).
KHÔNG dùng câu public test, KHÔNG dùng câu trong data/eval300.jsonl (tập kiểm của trưởng nhóm).

Chạy từng phần (mỗi lệnh <= ~50 giây, làm tiếp được):
    python3 training_code/data_generation/tao_vi_du_cong_cu.py --tien-do $HOME/tmp_dieuphoi/vd_tien_do.jsonl --ngan-sach 40
Khi in "XONG" thì gộp:
    python3 training_code/data_generation/tao_vi_du_cong_cu.py --tien-do $HOME/tmp_dieuphoi/vd_tien_do.jsonl --gop src/vi_du_cong_cu.jsonl
Biến môi trường: DATAMATE_CACHE (cache dữ liệu shop), PK_DIR (cache pickle (biz, ban_ok) từng shop, mặc định $HOME/lead/pk).
"""
import argparse
import json
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data", "train"))
ap.add_argument("--loai", default=os.path.join(ROOT, "data", "eval300.jsonl"), help="file id câu cần loại (tập kiểm)")
ap.add_argument("--cong-khai", default=os.path.join(ROOT, "data", "public_test", "questions.jsonl"), help="chỉ để LOẠI các câu trùng chữ với câu public")
ap.add_argument("--tien-do", required=True)
ap.add_argument("--ngan-sach", type=float, default=40)
ap.add_argument("--gop", default="", help="gộp file tiến độ ra file ví dụ này rồi thoát")
ap.add_argument("--pk", default=os.environ.get("PK_DIR", os.path.expanduser("~/lead/pk")))
args = ap.parse_args()

if args.gop:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import cong_cu
    cam = set()                                  # chữ câu của tập kiểm + public: không dùng làm ví dụ dù trùng câu train
    for f_ in (args.loai, args.cong_khai):
        if os.path.exists(f_):
            cam |= {json.loads(l).get("question") for l in open(f_, encoding="utf-8") if l.strip()}
    seen, rows, dem = set(), [], {}
    for line in open(args.tien_do, encoding="utf-8"):
        for r in json.loads(line)["rows"]:
            if r["q"] in seen or r["q"] in cam:
                continue
            seen.add(r["q"])
            cc = cong_cu.nhan_cong_cu(r["stage"], r["q"])       # gán nhãn lại theo danh mục công cụ hiện tại
            rows.append({"q": r["q"], "cong_cu": cc})
            dem[cc] = dem.get(cc, 0) + 1
    with open(args.gop, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã ghi {len(rows)} ví dụ -> {args.gop}\nTheo công cụ: {json.dumps(dem, ensure_ascii=False)}")
    sys.exit(0)

os.environ.setdefault("DATAMATE_CACHE", os.path.expanduser("~/dm_cache"))
from loader import load_business
from sales import build_sales, ban_health
import agent
import cong_cu
from scorer import score_one

rb, cal = agent._static(args.data)
loai = {json.loads(l)["id"] for l in open(args.loai, encoding="utf-8") if l.strip()}
gold = {}
for l in open(os.path.join(args.data, "answers.jsonl"), encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        gold[d["id"]] = d
qs = [json.loads(l) for l in open(os.path.join(args.data, "questions.jsonl"), encoding="utf-8") if l.strip()]
qs = [q for q in qs if q["id"] not in loai and q["id"] in gold]
xong = set()
if os.path.exists(args.tien_do):
    xong = {json.loads(l)["biz"] for l in open(args.tien_do, encoding="utf-8")}
os.makedirs(args.pk, exist_ok=True)
t0 = time.time()
con_lai = [s for s in sorted({q["business_id"] for q in qs}) if s not in xong]
for s in con_lai:
    if time.time() - t0 > args.ngan_sach:
        print(f"tiep (còn {len(con_lai)} shop)")
        sys.exit(0)
    fp = os.path.join(args.pk, f"{os.path.basename(args.data.rstrip('/'))}_{s}.pkl")
    if os.path.exists(fp):
        z, ok = pickle.load(open(fp, "rb"))
    else:
        z = load_business(os.path.join(args.data, "data", s))
        z.ban = build_sales(z)
        try:
            ok = ban_health(z)[0]
        except Exception:
            ok = False
        pickle.dump((z, ok), open(fp, "wb"))
    ban = z.ban if ok else None
    rows = []
    for q in (x for x in qs if x["business_id"] == s):
        bc = cong_cu.BoiCanh(z, ban, ok, rb, cal, q["as_of_date"])
        stage, r = cong_cu.chay_cac(cong_cu.THU_TU, q["question"], bc)
        if r is None or score_one(r[0], gold[q["id"]]) != 1:
            continue
        rows.append({"q": q["question"], "cong_cu": cong_cu.nhan_cong_cu(stage, q["question"]), "stage": stage})
    with open(args.tien_do, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"biz": s, "rows": rows}, ensure_ascii=False) + "\n")
print("XONG")
