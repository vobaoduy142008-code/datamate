"""Tạo src/am_tiet.txt: danh sách âm tiết tiếng Việt CÓ DẤU (chữ thường) gặp trong chữ của dữ liệu TRAIN
(câu hỏi train, rulebook, README các shop train) và từ điển viết tắt của đội (docs/tu_dien_viet_tat.md).
Dùng để sửa lỗi gõ Telex trong câu hỏi: chỉ nhận bản sửa nếu ra một âm tiết có trong danh sách.
Cách chạy:  python training_code/data_generation/tao_am_tiet.py data/train src/am_tiet.txt"""
import glob, json, os, re, sys, unicodedata

def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")

def main(train, out):
    texts = [open(os.path.join(train, "rulebook.md"), encoding="utf-8").read()]
    texts += [open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(train, "data", "*", "README.md"))]
    texts += [json.loads(l)["question"] for l in open(os.path.join(train, "questions.jsonl"), encoding="utf-8") if l.strip()]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "tu_dien_viet_tat.md")
    if os.path.exists(d):
        texts.append(open(d, encoding="utf-8").read())
    words = set()
    for t in texts:
        for w in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", t.lower())):
            if fold(w) != w and len(w) <= 7:          # âm tiết tiếng Việt dài nhất 7 chữ ("nghiêng")
                words.add(w)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))
    print(out, len(words), "âm tiết")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
