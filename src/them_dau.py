"""Thêm dấu cho câu hỏi gõ không dấu ("thang 1/2026 co may mat hang duoc mua o 246 don").
Mô hình bigram đơn giản: từ điển âm tiết có dấu + cặp từ liền nhau, học từ chữ có dấu (câu hỏi train, rulebook,
README) và tên hàng / nhóm / khách của chính shop đang hỏi. Chỉ đổi từ KHÔNG dấu; từ đã có dấu giữ nguyên."""
import json, math, os, re, unicodedata
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_FILE = os.path.join(_HERE, "ngram_dau.json")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


def _has_accent(w):
    return _fold(w) != w.lower()


class Model:
    def __init__(self, uni=None, bi=None):
        self.uni = defaultdict(Counter)     # từ không dấu -> {từ có dấu: số lần}
        self.bi = Counter()                 # "a b" (có dấu, chữ thường) -> số lần
        self.cnt = Counter()                # từ có dấu -> số lần
        for k, d in (uni or {}).items():
            self.uni[k].update(d)
            for w, n in d.items():
                self.cnt[w] += n
        self.bi.update(bi or {})

    def learn(self, text, weight=1):
        if it_dau(str(text)):          # câu không dấu không dạy được cách thêm dấu
            return
        words = [unicodedata.normalize("NFC", w.lower()) for w in _WORD.findall(str(text))]
        for w in words:
            self.uni[_fold(w)][w] += weight
            self.cnt[w] += weight
        for a, b in zip(words, words[1:]):
            self.bi[f"{a} {b}"] += weight

    def copy(self):
        m = Model()
        m.uni = defaultdict(Counter, {k: Counter(v) for k, v in self.uni.items()})
        m.bi = Counter(self.bi)
        m.cnt = Counter(self.cnt)
        return m

    def restore_words(self, words):
        """words: list chữ thường. Trả list có dấu (Viterbi bigram)."""
        cands = []
        for w in words:
            if _has_accent(w):
                cands.append([w])
                continue
            u = self.uni.get(w)
            cands.append(list(u) if u else [w])
        best = [{c: (self._lp_uni(words[0], c), None) for c in cands[0]}]
        for i in range(1, len(words)):
            cur = {}
            for c in cands[i]:
                lu = self._lp_uni(words[i], c)
                sc, bp = max(((best[-1][p][0] + self._lp_bi(p, c) + lu, p) for p in best[-1]), key=lambda x: x[0])
                cur[c] = (sc, bp)
            best.append(cur)
        c = max(best[-1], key=lambda k: best[-1][k][0])
        out = [c]
        for i in range(len(words) - 1, 0, -1):
            c = best[i][c][1]
            out.append(c)
        return out[::-1]

    def _lp_uni(self, w, c):
        u = self.uni.get(w)
        if not u:
            return 0.0
        return math.log((u.get(c, 0) + 0.1) / (sum(u.values()) + 0.1 * len(u)))

    def _lp_bi(self, p, c):
        n = self.bi.get(f"{p} {c}", 0)
        return math.log((n + 0.05) / (self.cnt.get(p, 0) + 1)) * 0.6 if n or self.cnt.get(p) else 0.0

    def restore(self, text):
        toks = re.split(r"([^\W\d_]+)", text)
        idx = [i for i in range(1, len(toks), 2)]
        if not idx:
            return text
        low = [toks[i].lower() for i in idx]
        acc = self.restore_words(low)
        for i, a in zip(idx, acc):
            w = toks[i]
            if _has_accent(w) or a == w.lower():
                continue
            if w.isupper() and len(w) > 1:
                a = a.upper()
            elif w[:1].isupper():
                a = a[:1].upper() + a[1:]
            toks[i] = a
        return "".join(toks)

    def save(self, path=_FILE):
        uni = {k: dict(v) for k, v in self.uni.items() if any(_has_accent(w) for w in v)}
        bi = {k: v for k, v in self.bi.items() if v >= 1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"uni": uni, "bi": bi}, f, ensure_ascii=False, separators=(",", ":"))


_BASE = None


def build_base(train_dir):
    """Dựng mô hình từ dữ liệu TRAIN (câu hỏi train, rulebook, README các shop train) + từ điển viết tắt."""
    import glob
    m = Model()
    for p in [os.path.join(train_dir, "rulebook.md"), *glob.glob(os.path.join(train_dir, "data", "*", "README.md")),
              os.path.join(_HERE, "..", "docs", "tu_dien_viet_tat.md")]:
        try:
            with open(p, encoding="utf-8") as f:
                m.learn(f.read())
        except OSError:
            pass
    try:
        with open(os.path.join(train_dir, "questions.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m.learn(json.loads(line)["question"])
    except OSError:
        pass
    return m


def base_model():
    """File ngram_dau.json (không đưa lên GitHub vì học từ dữ liệu train) – thiếu thì tự dựng từ data/train."""
    global _BASE
    if _BASE is None:
        try:
            with open(_FILE, encoding="utf-8") as f:
                d = json.load(f)
            _BASE = Model(d["uni"], d["bi"])
        except (OSError, ValueError):
            root = os.path.join(_HERE, "..")
            for td in (os.environ.get("DATAMATE_TRAIN", ""), os.path.join(root, "data", "train"), "/code/data/train"):
                if td and os.path.exists(os.path.join(td, "questions.jsonl")):
                    _BASE = build_base(td)
                    try:
                        _BASE.save()
                    except OSError:
                        pass
                    break
            else:
                _BASE = Model()
    return _BASE


def it_dau(text):
    """True nếu câu gần như không có dấu (dưới 15% số từ có dấu)."""
    words = _WORD.findall(text)
    words = [w for w in words if len(w) > 1]
    if len(words) < 3:
        return False
    return sum(_has_accent(w) for w in words) / len(words) < 0.15


def them_dau(text, extra_texts=()):
    """Thêm dấu nếu câu gần như không dấu; nếu không thì trả nguyên văn."""
    if not it_dau(text):
        return text
    m = base_model()
    if extra_texts:
        m = m.copy()
        for t in extra_texts:
            m.learn(t, weight=2)
    return m.restore(text)
