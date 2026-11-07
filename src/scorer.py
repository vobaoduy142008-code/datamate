"""
Chấm điểm local mô phỏng luật DataMate (Penalized Accuracy: đúng +1, abstain 0, sai -1).

Cách dùng (chạy trong thư mục dự án):
    python src/scorer.py data/train/answers.jsonl outputs/run/submission.jsonl
    python src/scorer.py data/train/answers.jsonl outputs/run/submission.jsonl --subset --show
      --subset : chỉ chấm các câu CÓ trong file nộp (khi chạy thử một phần, vd --sample 30)
      --show   : in chi tiết từng câu sai (câu hỏi, đáp án của mình, đáp án đúng)

- File đáp án dùng ĐÚNG định dạng answers.jsonl của BTC:
    {"id": "...", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
    {"id": "...", "type": "list",   "answer": ["A", "B"], "alias": [["SP01"], ["SP02"]]}
    {"id": "...", "type": "table",  "answer": {"columns": [...], "rows": [...]}, "alias": [["KH01"], ...]}
    {"id": "...", "type": "unanswerable", "answer": null, "alias": []}
  "alias" là các cách viết khác / mã được chấp nhận (string: cho cả chuỗi; list: cho từng phần
  tử; table: cho ô tên ở cột đầu của từng dòng).
- File dự đoán dùng định dạng submission.jsonl; câu thiếu tính là abstain.
"""
import json
import sys
import unicodedata
from collections import Counter

# ---------------------------------------------------------------- chuẩn hoá

def norm_str(s) -> str:
    """NFC -> lowercase -> ký tự không phải chữ/số/khoảng trắng thành space -> gộp space.
    KHÔNG bỏ dấu tiếng Việt."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def _forms(ref):
    """Danh sách dạng được chấp nhận của một tham chiếu string."""
    return ref if isinstance(ref, list) else [ref]


def _is_num(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    try:
        float(str(x))
        return True
    except ValueError:
        return False

# ---------------------------------------------------------------- so khớp

def match_number(a, ref) -> bool:
    try:
        return float(a) == float(ref)          # exact match
    except (TypeError, ValueError):
        return False


def match_percent(a, ref) -> bool:
    try:
        return round(float(a), 2) == round(float(ref), 2)
    except (TypeError, ValueError):
        return False


def match_string(a, ref) -> bool:
    na = norm_str(a)
    return any(na == norm_str(f) for f in _forms(ref))


def match_cell(a, ref) -> bool:
    if isinstance(ref, list) or not _is_num(ref):
        return match_string(a, ref)
    return match_number(a, ref)


def match_list(a, ref) -> bool:
    if not isinstance(a, list) or len(a) != len(ref):
        return False
    return all(match_cell(x, r) for x, r in zip(a, ref))   # đúng thứ tự từng vị trí


def match_table(a, ref) -> bool:
    try:
        a_cols, a_rows = a["columns"], a["rows"]
        r_cols, r_rows = ref["columns"], ref["rows"]
    except (TypeError, KeyError):
        return False
    if len(a_cols) != len(r_cols) or len(a_rows) != len(r_rows):
        return False
    # cột: khớp theo tên đã chuẩn hoá; nếu tên không khớp thì theo vị trí
    na = [norm_str(c) for c in a_cols]
    nr = [norm_str(c) for c in r_cols]
    if sorted(na) == sorted(nr) and len(set(nr)) == len(nr):
        perm = [na.index(c) for c in nr]
    else:
        perm = list(range(len(r_cols)))
    a_rows = [[row[i] for i in perm] for row in a_rows]
    # hàng: khớp như tập hợp (không quan tâm thứ tự)
    unused = list(range(len(a_rows)))
    for rrow in r_rows:
        hit = next((j for j in unused
                    if all(match_cell(x, r) for x, r in zip(a_rows[j], rrow))), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


MATCHERS = {
    "number": match_number,
    "percent": match_percent,
    "string": match_string,
    "date": lambda a, r: str(a) == str(r),
    "boolean": lambda a, r: a is r or str(a).lower() == str(r).lower(),
    "list": match_list,
    "table": match_table,
}


def with_alias(ref):
    """Gộp 'alias' của answers.jsonl vào đáp án: mỗi tên thành danh sách các dạng được nhận."""
    t, ans, al = ref["type"], ref.get("answer"), ref.get("alias") or []
    if t == "string":
        return [ans] + list(al)
    if t == "list":
        return [([x] + list(al[i])) if (i < len(al) and al[i] and not _is_num(x)) else x
                for i, x in enumerate(ans)]
    if t == "table" and al:
        rows = []
        for i, row in enumerate(ans["rows"]):
            row = list(row)
            if i < len(al) and al[i] and row and not _is_num(row[0]):
                row[0] = [row[0]] + list(al[i])
            rows.append(row)
        return {"columns": ans["columns"], "rows": rows}
    return ans


def score_one(pred, ref) -> int:
    """+1 đúng, 0 abstain, -1 sai."""
    if pred is None or pred.get("type") == "abstain":
        return 0
    if pred.get("type") != ref["type"]:
        return -1
    if ref["type"] == "unanswerable":
        return 1
    return 1 if MATCHERS[ref["type"]](pred.get("answer"), with_alias(ref)) else -1

# ---------------------------------------------------------------- định dạng đáp án

def format_number(x):
    """Số nguyên -> int; còn lại làm tròn 2 chữ số. Tránh 48250000.000001."""
    x = float(x)
    return int(round(x)) if abs(x - round(x)) < 1e-6 else round(x, 2)


def format_percent(x):
    """x đã ở đơn vị phần trăm (-12.45, không phải -0.1245)."""
    return round(float(x), 2)

# ---------------------------------------------------------------- main

def load(path):
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}


def main(ref_path, sub_path, subset=False, show=False):
    refs, subs = load(ref_path), load(sub_path)
    if subset:
        refs = {k: v for k, v in refs.items() if k in subs}
    stats = Counter()
    wrong = []
    for qid, ref in refs.items():
        s = score_one(subs.get(qid), ref)
        stats[s] += 1
        if s == -1:
            wrong.append(qid)
    by_type = {}
    for qid, ref in refs.items():
        s = score_one(subs.get(qid), ref)
        d = by_type.setdefault(ref["type"], Counter()); d[s] += 1
    n = len(refs)
    pa = (stats[1] - stats[-1]) / n if n else 0.0
    print(f"N={n}  đúng={stats[1]}  abstain={stats[0]}  sai={stats[-1]}")
    print(f"Penalized Accuracy = {pa:.4f}")
    print("Theo loại đáp án (đúng / abstain / sai):")
    for t, d in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {t:13s} {d[1]:5d} / {d[0]:5d} / {d[-1]:5d}")
    if wrong:
        print("Một số câu sai:", ", ".join(wrong[:20]))
    if show and wrong:
        import os
        qpath = os.path.join(os.path.dirname(ref_path), "questions.jsonl")
        qs = load(qpath) if os.path.exists(qpath) else {}
        print("\n===== CHI TIẾT CÂU SAI =====")
        for qid in wrong:
            q = qs.get(qid, {})
            p, r = subs.get(qid, {}), refs[qid]
            print(f"- {qid} [{q.get('business_id', '')}] hôm nay {q.get('as_of_date', '')}: {q.get('question', '')}")
            print(f"    mình : {p.get('type')} {json.dumps(p.get('answer'), ensure_ascii=False)[:200]}")
            print(f"    đúng : {r['type']} {json.dumps(r.get('answer'), ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], subset="--subset" in sys.argv, show="--show" in sys.argv)
