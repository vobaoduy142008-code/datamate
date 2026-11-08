"""Tạo src/ngram_dau.json: bảng thêm dấu cho câu hỏi gõ không dấu (đếm từ đơn + cặp từ liền nhau).
Học từ chữ CÓ DẤU của dữ liệu TRAIN: câu hỏi train, rulebook, README các shop train, từ điển viết tắt của đội.
Không dùng dữ liệu public/private test. Lúc chạy, agent cộng thêm tên hàng / nhóm hàng / README của chính shop đang hỏi.
Cách chạy:  python training_code/data_generation/tao_ngram_dau.py data/train
(Dockerfile chạy bước này lúc build; thiếu file thì src/them_dau.py cũng tự dựng lại y như vậy.)"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import them_dau  # noqa: E402

if __name__ == "__main__":
    m = them_dau.build_base(sys.argv[1] if len(sys.argv) > 1 else "data/train")
    m.save()
    print(them_dau._FILE, len(m.uni), "từ,", len(m.bi), "cặp từ")
