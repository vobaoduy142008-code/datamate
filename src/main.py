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
    --sample 30         lấy ngẫu nhiên 30 câu (cố định theo --seed) để thử nhanh, đại diện cho nhiều shop
Câu hỏi được chạy theo nhóm shop (để đọc dữ liệu mỗi shop 1 lần); thứ tự dòng trong file nộp không quan trọng.
"""
import argparse
import json
import random
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
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="chạy tiếp: giữ các câu đã có trong submission.jsonl, chỉ chạy các câu còn thiếu")
    args = ap.parse_args()

    questions = load_questions(args.data)
    if args.business:
        questions = [q for q in questions if q["business_id"] == args.business]
    if args.sample:
        questions = random.Random(args.seed).sample(questions, min(args.sample, len(questions)))
    if args.limit:
        questions = questions[: args.limit]
    questions.sort(key=lambda q: q["business_id"])      # gom theo shop: mỗi shop chỉ đọc dữ liệu 1 lần

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    sub_path = os.path.join(args.out, "submission.jsonl")
    log_path = os.path.join(args.out, "logs", "log.jsonl")

    n_ok = 0
    mode = "w"
    if args.resume and os.path.exists(sub_path):
        done = set()
        with open(sub_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass                                    # dòng cuối bị cắt dở khi máy tắt: bỏ qua
        questions = [q for q in questions if q["id"] not in done]
        mode = "a"
        print(f"Chạy tiếp: đã có {len(done)} câu, còn {len(questions)} câu.")
    # mỗi câu ghi ngay một dòng (flush) để nếu dừng giữa chừng vẫn giữ được kết quả
    with open(sub_path, mode, encoding="utf-8") as fsub, open(log_path, mode, encoding="utf-8") as flog:
        for i, q in enumerate(questions, 1):
            t0 = time.time()
            if not args.quiet:
                print(f"[{i}/{len(questions)}] {q['business_id']}: {q['question'][:90]}", flush=True)
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
                "steps": result.get("steps"),
                "question": q["question"],
                "business_id": q["business_id"],
            }
            flog.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
            flog.flush()
            if not args.quiet:
                shown = json.dumps(answer.get("answer", answer.get("type")), ensure_ascii=False)[:80]
                print(f"    => {elapsed_ms / 1000:.0f}s  {answer.get('type')}: {shown}", flush=True)
            elif i % 500 == 0 or i == len(questions):
                print(f"[{i}/{len(questions)}] đã trả lời {n_ok} câu")

    print(f"Xong. Kết quả: {sub_path}")


if __name__ == "__main__":
    main()
