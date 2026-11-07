"""
Chương trình chính: đọc câu hỏi -> gọi agent -> ghi submission.jsonl + log từng câu.

Chạy trên máy cá nhân (dữ liệu để trong thư mục data/ của dự án):
    python src/main.py --data data/train --out outputs/train_run
    python src/scorer.py data/train/answers.jsonl outputs/train_run/submission.jsonl

Trong Docker của cuộc thi, predict.sh gọi:  python src/main.py --data /data --out /result
(đúng như Docker guideline: đọc /data, ghi /result/submission.jsonl và /result/logs/).

Tuỳ chọn:
    --limit 20          chỉ chạy 20 câu đầu (để thử nhanh)
    --business biz_101  chỉ chạy câu của một shop (CHỈ dùng khi phát triển, không dùng khi nộp)
"""
import argparse
import json
import os
import time
import traceback

from agent import solve


def load_questions(data_dir):
    path = os.path.join(data_dir, "questions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/result")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--business", default="")
    args = ap.parse_args()

    questions = load_questions(args.data)
    if args.business:
        questions = [q for q in questions if q["business_id"] == args.business]
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    sub_path = os.path.join(args.out, "submission.jsonl")
    log_path = os.path.join(args.out, "logs", "log.jsonl")

    n_ok = 0
    # mở file ở chế độ ghi; mỗi câu ghi ngay một dòng (flush) để nếu dừng giữa chừng vẫn giữ được kết quả
    with open(sub_path, "w", encoding="utf-8") as fsub, open(log_path, "w", encoding="utf-8") as flog:
        for i, q in enumerate(questions, 1):
            t0 = time.time()
            try:
                result = solve(q, args.data)          # agent trả về dict: answer + thông tin log
            except Exception:
                result = {"answer": {"type": "abstain"}, "error": traceback.format_exc()}
            answer = result["answer"]
            if answer.get("type") not in ("abstain", None):
                n_ok += 1
            elapsed_ms = int((time.time() - t0) * 1000)

            fsub.write(json.dumps({"id": q["id"], **answer}, ensure_ascii=False) + "\n")
            fsub.flush()
            log = {
                "id": q["id"],
                "answer": answer,
                "elapsed_ms": elapsed_ms,
                "thinking": result.get("thinking"),   # nguyên văn suy nghĩ của model, null nếu không có
                "code": result.get("code"),           # code đã chạy
                "output": result.get("output"),       # kết quả chạy code
                "error": result.get("error"),
            }
            flog.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
            flog.flush()
            if i % 500 == 0 or i == len(questions):
                print(f"[{i}/{len(questions)}] đã trả lời {n_ok} câu")

    print(f"Xong. Kết quả: {sub_path}")


if __name__ == "__main__":
    main()
