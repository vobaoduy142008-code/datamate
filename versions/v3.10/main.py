"""
Chương trình chính: đọc câu hỏi -> gọi agent -> ghi submission.jsonl + log từng câu.

Chạy trên máy cá nhân (dữ liệu để trong thư mục data/ của dự án):
    python src/main.py --data data/train --out outputs/train_run
    python src/scorer.py data/train/answers.jsonl outputs/train_run/submission.jsonl

Trong Docker của cuộc thi, predict.sh gọi:  python src/main.py --data /data --out /result --workers 4 --deadline <giờ>
(đúng như Docker guideline: đọc /data, ghi /result/submission.jsonl và /result/logs/).

Tuỳ chọn:
    --limit 20          chỉ chạy 20 câu đầu (để thử nhanh)
    --business biz_101  chỉ chạy câu của một shop (CHỈ dùng khi phát triển, không dùng khi nộp)
    --sample 30         lấy ngẫu nhiên 30 câu (cố định theo --seed) để thử nhanh, đại diện cho nhiều shop
    --resume            chạy tiếp: giữ các câu đã có trong submission.jsonl
    --workers 4         chạy song song 4 câu một lúc (vLLM gom các yêu cầu lại nên nhanh hơn nhiều).
                        Trên Mac với Ollama để mặc định 1.
    --deadline 1731000000  mốc giờ (epoch giây) PHẢI xong. Mỗi câu được chia thời gian theo số câu còn lại;
                        đến mốc mà còn câu chưa xong thì ghi "abstain" cho các câu đó rồi thoát.
    --prewarm           chỉ đọc trước dữ liệu mọi shop vào cache (chạy lúc vLLM đang nạp model) rồi thoát.
Câu hỏi được chạy theo nhóm shop (để đọc dữ liệu mỗi shop 1 lần); thứ tự dòng trong file nộp không quan trọng.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import time
import traceback

import sys
if sys.platform == "darwin":          # macOS: tiến trình con (fork) không được tra proxy hệ thống (gây crash)
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
import agent
import llm
from agent import solve

BASE_BUDGET = agent.TIME_BUDGET          # giây tối đa cho 1 câu (AGENT_TIME_BUDGET)
BASE_LLM_TIMEOUT = llm.TIMEOUT

# biến dùng chung cho các tiến trình con (gán trước khi fork)
_G = {"data": None, "deadline": 0.0, "workers": 1, "total": 0, "done": None}


def load_questions(data_dir):
    path = os.path.join(data_dir, "questions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _budget_for_next():
    """Thời gian cho câu tiếp theo khi có hạn chót: chia đều thời gian còn lại cho các câu còn lại."""
    if not _G["deadline"]:
        return BASE_BUDGET
    remaining = _G["deadline"] - time.time()
    left = max(1, _G["total"] - (_G["done"].value if _G["done"] is not None else 0))
    fair = remaining * _G["workers"] / left
    # được dùng tối đa gấp đôi phần chia đều (nhiều câu xong sớm hơn), nhưng không quá BASE_BUDGET
    return int(max(0, min(BASE_BUDGET, 2 * fair, remaining - 15)))


def _solve_one(q):
    """Giải 1 câu, KHÔNG BAO GIỜ ném lỗi. Trả về (câu hỏi, kết quả, ms)."""
    t0 = time.time()
    budget = _budget_for_next()
    if budget < 15:
        result = {"answer": {"type": "abstain"}, "error": "hết thời gian tổng, bỏ qua câu này"}
    else:
        agent.TIME_BUDGET = budget
        llm.TIMEOUT = max(15, min(BASE_LLM_TIMEOUT, budget))
        try:
            result = solve(q, _G["data"])
        except KeyboardInterrupt:
            raise
        except BaseException:            # kể cả SystemExit do code của model: không để dừng cả lượt chạy
            result = {"answer": {"type": "abstain"}, "error": traceback.format_exc()}
    if _G["done"] is not None:
        with _G["done"].get_lock():
            _G["done"].value += 1
    return q, result, int((time.time() - t0) * 1000)


def _child(q, conn):
    try:
        conn.send(_solve_one(q)[1])
    except BaseException:
        try:
            conn.send({"answer": {"type": "abstain"}, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        conn.close()
        os._exit(0)


def _solve_isolated(q):
    """Giải 1 câu trong TIẾN TRÌNH CON (fork). Code của model làm sập tiến trình (hết bộ nhớ, crash thư viện...)
    thì chỉ mất câu đó (abstain), lượt chạy vẫn tiếp tục. Dữ liệu shop nạp sẵn ở tiến trình cha, con dùng chung."""
    t0 = time.time()
    try:
        agent._business(_G["data"], q["business_id"])     # nạp trước ở cha -> con không phải đọc lại
    except Exception:
        pass
    budget = _budget_for_next()
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(q, child), daemon=True)
    p.start()
    child.close()
    result = None
    limit = max(30, budget) + 90                          # quá hạn hẳn thì dừng con
    try:
        if parent.poll(limit):
            result = parent.recv()
    except (EOFError, OSError):
        result = None
    if p.is_alive():
        p.kill()
    p.join(5)
    if result is None:
        result = {"answer": {"type": "abstain"},
                  "error": f"tiến trình giải câu này bị dừng (mã {p.exitcode}; có thể code của model dùng quá nhiều bộ nhớ)"}
    return q, result, int((time.time() - t0) * 1000)


def _prewarm_one(business_id):
    t0 = time.time()
    try:
        agent._business(_G["data"], business_id)
        return business_id, round(time.time() - t0, 1), None
    except Exception as e:
        return business_id, round(time.time() - t0, 1), repr(e)


def _join(x, sep):
    """Gộp danh sách (mỗi lượt 1 phần) thành 1 chuỗi; không có gì thì null."""
    if isinstance(x, (list, tuple)):
        parts = [str(p) for p in x if p]
        return sep.join(parts) if parts else None
    return x or None


def _safe_name(s):
    return re.sub(r"[^\w.-]+", "_", str(s))[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/result")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--business", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default="", help="file danh sách câu cần chạy (jsonl có trường id, hoặc mỗi dòng một id)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="chạy tiếp: giữ các câu đã có trong submission.jsonl, chỉ chạy các câu còn thiếu")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AGENT_WORKERS", "1")))
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("DEADLINE_EPOCH", "0")))
    ap.add_argument("--prewarm", action="store_true")
    args = ap.parse_args()

    _G["data"] = args.data
    _G["deadline"] = args.deadline
    _G["workers"] = max(1, args.workers)

    questions = load_questions(args.data)
    if args.business:
        questions = [q for q in questions if q["business_id"] == args.business]
    if args.ids:
        want = []
        for line in open(args.ids, encoding="utf-8"):
            line = line.strip()
            if line:
                want.append(json.loads(line)["id"] if line.startswith("{") else line)
        pos = {i: k for k, i in enumerate(want)}
        questions = sorted([q for q in questions if q["id"] in pos], key=lambda q: pos[q["id"]])
    if args.sample:
        questions = random.Random(args.seed).sample(questions, min(args.sample, len(questions)))
    if args.limit:
        questions = questions[: args.limit]
    questions.sort(key=lambda q: q["business_id"])      # gom theo shop: mỗi shop chỉ đọc dữ liệu 1 lần

    if args.prewarm:                                     # đọc trước dữ liệu các shop (song song), ghi cache rồi thoát
        bids = sorted({q["business_id"] for q in questions})
        t0 = time.time()
        with mp.get_context("fork").Pool(max(1, min(len(bids), os.cpu_count() or 2, 8))) as pool:
            for bid, sec, err in pool.imap_unordered(_prewarm_one, bids):
                print(f"[prewarm] {bid}: {sec}s" + (f"  LỖI {err}" if err else ""), flush=True)
        print(f"[prewarm] xong {len(bids)} shop trong {time.time() - t0:.0f}s", flush=True)
        return

    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sub_path = os.path.join(args.out, "submission.jsonl")
    log_path = os.path.join(log_dir, "log.jsonl")

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

    _G["total"] = len(questions)
    n_ok, written = 0, set()
    t_start = time.time()
    # mỗi câu ghi ngay một dòng (flush) để nếu dừng giữa chừng vẫn giữ được kết quả
    with open(sub_path, mode, encoding="utf-8") as fsub, open(log_path, mode, encoding="utf-8") as flog:

        def write(i, q, result, elapsed_ms):
            nonlocal n_ok
            answer = result.get("answer") or {"type": "abstain"}
            if answer.get("type") not in ("abstain", None):
                n_ok += 1
            fsub.write(json.dumps({"id": q["id"], **answer}, ensure_ascii=False) + "\n")
            fsub.flush()
            written.add(q["id"])
            log = {
                "id": q["id"],
                "answer": answer,
                "elapsed_ms": elapsed_ms,
                "thinking": _join(result.get("thinking"), "\n\n----- lượt sau -----\n\n"),  # nguyên văn, null nếu không có
                "code": _join(result.get("code"), "\n\n# ----- lượt sau -----\n"),        # code đã chạy
                "output": _join(result.get("output"), "\n----- lượt sau -----\n"),         # kết quả chạy code
                "error": result.get("error"),
                "steps": result.get("steps"),
                "votes": result.get("votes"),           # khi AGENT_VOTE=1: đáp án 2 lần giải
                "question": q["question"],
                "business_id": q["business_id"],
            }
            line = json.dumps(log, ensure_ascii=False, default=str)
            flog.write(line + "\n")
            flog.flush()
            with open(os.path.join(log_dir, _safe_name(q["id"]) + ".json"), "w", encoding="utf-8") as f:
                f.write(line + "\n")                  # thêm 1 file log riêng cho từng câu
            if not args.quiet:
                shown = json.dumps(answer.get("answer", answer.get("type")), ensure_ascii=False)[:80]
                print(f"[{i}/{len(questions)}] {q['business_id']} {elapsed_ms / 1000:.0f}s "
                      f"{answer.get('type')}: {shown}   | {q['question'][:70]}", flush=True)
            elif i % 50 == 0 or i == len(questions):
                print(f"[{i}/{len(questions)}] đã trả lời {n_ok} câu, {time.time() - t_start:.0f}s", flush=True)

        if _G["workers"] == 1:
            for i, q in enumerate(questions, 1):
                if not args.quiet:
                    print(f"[{i}/{len(questions)}] {q['business_id']}: {q['question'][:90]}", flush=True)
                _, result, ms = (_solve_isolated(q) if os.environ.get("DATAMATE_ISOLATE", "1") != "0"
                                 else _solve_one(q))
                write(i, q, result, ms)
        else:
            ctx = mp.get_context("fork")
            _G["done"] = ctx.Value("i", 0)
            pool = ctx.Pool(_G["workers"], maxtasksperchild=None)
            it = pool.imap_unordered(_solve_one, questions, chunksize=1)
            i = 0
            try:
                while i < len(questions):
                    wait = None
                    if _G["deadline"]:
                        wait = _G["deadline"] - time.time()
                        if wait <= 0:
                            raise mp.TimeoutError
                    q, result, ms = it.next(timeout=wait)
                    i += 1
                    write(i, q, result, ms)
                pool.close()
            except (mp.TimeoutError, StopIteration):
                print("!! Hết hạn tổng: dừng các câu đang chạy.", flush=True)
                pool.terminate()
            finally:
                pool.join()

        # câu nào chưa có kết quả (hết giờ) -> abstain, để file nộp luôn đủ câu
        missing = [q for q in questions if q["id"] not in written]
        for q in missing:
            write(len(written) + 1, q, {"answer": {"type": "abstain"}, "error": "hết thời gian tổng"}, 0)
        if missing:
            print(f"!! {len(missing)} câu hết giờ -> abstain", flush=True)

    print(f"Xong. Trả lời {n_ok}/{len(questions)} câu trong {time.time() - t_start:.0f}s. Kết quả: {sub_path}")


if __name__ == "__main__":
    main()
