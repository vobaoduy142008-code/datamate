"""Ghi training_code/data_manifest.json: đường dẫn, kích thước, SHA-256 của dữ liệu train và các file sinh ra từ đó.
Cách chạy (Dockerfile tự chạy lúc build):  python scripts/tao_manifest.py /code /code/training_code/data_manifest.json"""
import hashlib, json, os, sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(root, out):
    files = []
    for rel in ("data/train", "src/ngram_dau.json", "src/am_tiet.txt", "docs/tu_dien_viet_tat.md"):
        p = os.path.join(root, rel)
        paths = [p] if os.path.isfile(p) else sorted(os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs)
        for q in paths:
            files.append({"path": os.path.relpath(q, root), "bytes": os.path.getsize(q), "sha256": sha(q)})
    qn = os.path.join(root, "data/train/questions.jsonl")
    n_q = sum(1 for l in open(qn, encoding="utf-8") if l.strip()) if os.path.exists(qn) else 0
    shops = len(os.listdir(os.path.join(root, "data/train/data"))) if os.path.isdir(os.path.join(root, "data/train/data")) else 0
    man = {
        "description": "DataMate – Zalo AI Challenge 2026. KHÔNG huấn luyện / fine-tune model. Dữ liệu duy nhất dùng là "
                       "bộ TRAIN do BTC cung cấp (đo điểm + dựng 2 bảng tra cứu chữ). Không dùng dữ liệu public/private test.",
        "datasets": [{"name": "DataMate train (BTC cung cấp)", "path": "data/train", "questions": n_q, "shops": shops,
                      "split": "toàn bộ dùng làm tập kiểm tra (validation); không có tập huấn luyện model"}],
        "generated": [
            {"path": "src/ngram_dau.json", "by": "training_code/data_generation/tao_ngram_dau.py",
             "from": ["data/train/questions.jsonl", "data/train/rulebook.md", "data/train/data/*/README.md", "docs/tu_dien_viet_tat.md"]},
            {"path": "src/am_tiet.txt", "by": "training_code/data_generation/tao_am_tiet.py",
             "from": ["data/train/questions.jsonl", "data/train/rulebook.md", "data/train/data/*/README.md", "docs/tu_dien_viet_tat.md"]},
            {"path": "docs/tu_dien_viet_tat.md", "by": "đội tự viết (từ viết tắt / từ địa phương thường gặp)"}],
        "base_model": {"name": "Qwen3-8B", "ollama_tag": "qwen3:8b", "license": "Apache-2.0",
                       "source": "https://ollama.com/library/qwen3:8b", "note": "trọng số nằm sẵn trong image (/models/ollama)"},
        "files": files,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(out, len(files), "file")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
