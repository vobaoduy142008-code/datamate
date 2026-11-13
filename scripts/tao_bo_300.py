"""Tạo BỘ ĐÁNH GIÁ CỐ ĐỊNH 300 câu train, chia 3 mức: dễ / trung bình / khó (mặc định 100 câu mỗi mức).
- Chỉ lấy câu CHƯA dùng ở các lượt đo trước (danh sách id trong --seen), để điểm không "đẹp giả".
- Mức khó xếp theo NỘI DUNG câu hỏi + kiểu đáp án (luật cố định, ghi rõ lý do từng câu), không theo kết quả model.
- Rải đều các shop (lần lượt từng shop), seed cố định -> chạy lại ra đúng bộ câu cũ.
Chạy:  python3 scripts/tao_bo_300.py data/train data/eval300.jsonl --seen outputs/seen_ids.json
Kết quả: mỗi dòng {"id","business_id","muc","nhom","ly_do","question"}  (KHÔNG chứa đáp án)."""
import argparse, json, os, random, re, sys, unicodedata
from collections import defaultdict, Counter


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


# (tên nhóm, mẫu trong câu hỏi đã bỏ dấu)  — nhóm đầu tiên khớp được dùng
_KHO = [
    ("công nợ", r"\bno\b|cong no|mua chiu|ghi no|tra no|thu no|qua han|tuoi no|con thieu|thieu tien"),
    ("tồn kho / lô / hạn dùng", r"\bton\b|trong kho|lo hang|\bcac lo\b|\bmay lo\b|\blo (nao|con|sap|gan)\b|han dung|het han|\bhsd\b|hop chan|chan.*\ble\b"),
    ("giao hàng", r"\bdot\b.*giao|giao hang|\bdot hang\b|chua giao"),
    ("lãi / chi phí", r"lai rong|loi nhuan rong|chi phi|tien dien|tien thue|luong"),
    ("nhiều số / lần lượt", r"lan luot|tung thang|tung ngay|hai so|2 so|ba so|\bva doanh thu\b|tra duoc bao nhieu"),
    ("xu hướng nhiều kỳ", r"luon (cao|thap|tang|giam)|lien tuc|xu huong"),
]
_TB = [
    ("tăng/giảm %, tỉ lệ", r"%|phan tram|ty le|ti le|ty trong|ti trong|chiem|tang hay giam|thay doi|chenh"),
    ("hiệu ứng tăng giá", r"tang gia|len gia|doi gia"),
    ("khách hàng", r"\bkhach\b"),
    ("ca / giờ / kênh", r"\bca (sang|chieu|toi)\b|\bgio\b|kenh|zalo|dien thoai|tai quay|online"),
    ("nhập hàng", r"\bnhap\b|nha cung cap|\bncc\b"),
    ("trung bình", r"trung binh|binh quan|moi don"),
    ("nhóm / hoạt chất / kê đơn", r"\bnhom\b|hoat chat|ke don"),
    ("top / xếp hạng", r"\btop\b|nhat\b"),
]


def phan_loai(q, a):
    f = fold(q)
    t = a.get("type")
    if t in ("list", "table"):
        return "kho", f"đáp án dạng {t}", "nhiều giá trị"
    if t in ("boolean", "unanswerable"):
        return "kho", f"đáp án dạng {t}", "đúng/sai, không trả lời được"
    for nhom, pat in _KHO:
        if re.search(pat, f):
            return "kho", f"chủ đề {nhom}", nhom
    for nhom, pat in _TB:
        if re.search(pat, f):
            return "tb", f"chủ đề {nhom}", nhom
    # câu gõ không dấu / viết tắt nhiều: khó đọc hơn -> trung bình
    words = re.findall(r"[^\W\d_]+", q)
    if len(words) >= 4 and sum(fold(w) != w.lower() for w in words) / len(words) < 0.15:
        return "tb", "gõ không dấu", "gõ không dấu"
    return "de", "một số liệu, một kỳ, một bảng", "cơ bản (doanh thu, số đơn, số lượng, giá, ngày)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("out")
    ap.add_argument("--seen", default="")
    ap.add_argument("--moi_muc", type=int, default=100)
    ap.add_argument("--seed", type=int, default=300)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8") if l.strip()]
    an = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8") if l.strip()}
    seen = set(json.load(open(a.seen))) if a.seen else set()
    pool = defaultdict(lambda: defaultdict(list))          # muc -> shop -> [câu]
    for q in qs:
        if q["id"] in seen:
            continue
        muc, ly_do, nhom = phan_loai(q["question"], an[q["id"]])
        pool[muc][q["business_id"]].append(dict(id=q["id"], business_id=q["business_id"], muc=muc, nhom=nhom,
                                                ly_do=ly_do, question=q["question"]))
    rng = random.Random(a.seed)
    out = []
    for muc in ("de", "tb", "kho"):
        shops = sorted(pool[muc])
        for s in shops:
            rng.shuffle(pool[muc][s])
        rng.shuffle(shops)
        chosen, i = [], 0
        while len(chosen) < a.moi_muc and any(pool[muc][s] for s in shops):   # lần lượt từng shop -> rải đều
            s = shops[i % len(shops)]
            if pool[muc][s]:
                chosen.append(pool[muc][s].pop())
            i += 1
        out += chosen
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(a.out, len(out), "câu")
    for muc in ("de", "tb", "kho"):
        rs = [r for r in out if r["muc"] == muc]
        print(f"  {muc}: {len(rs)} câu, {len({r['business_id'] for r in rs})} shop | nhóm:",
              dict(Counter(r["nhom"] for r in rs).most_common()))


if __name__ == "__main__":
    main()
