#!/usr/bin/env python3
"""Chuẩn bị dữ liệu SFT cho mlx_lm.lora.

Đọc  : training_code/dsl/sft_raw.jsonl
       mỗi dòng {"messages":[{"role":"system"|"user"|"assistant","content":...}, ...], "biz": "...", "id": "..."}
Ghi  : <out-dir>/train.jsonl, <out-dir>/valid.jsonl  (mỗi dòng {"messages":[...]} - định dạng "chat" của mlx_lm.lora)

- Chia theo SHOP (biz): cả shop vào train hoặc cả shop vào valid (~90/10 theo số ví dụ), tránh rò rỉ giữa train và valid.
- Loại mọi ví dụ thuộc eval300 (theo id, và theo nội dung câu hỏi) và mọi câu hỏi public (theo id và nội dung).
- Loại ví dụ trùng, ví dụ hỏng định dạng, ví dụ quá dài (> --max-tokens).

Chạy từ thư mục datamate:
    .venv/bin/python training_code/finetune/chuan_bi_du_lieu.py
    .venv/bin/python training_code/finetune/chuan_bi_du_lieu.py --tokenizer Qwen/Qwen3-8B   # đếm token chính xác + in mẫu đã áp chat template
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def chuan_hoa(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def doc_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [cảnh báo] {path} dòng {i}: JSON hỏng ({e}) -> bỏ qua", file=sys.stderr)


def nap_cau_cam(duong_dan_list):
    """Trả về (tập id, danh sách câu hỏi đã chuẩn hoá) của các file câu hỏi bị cấm dùng để huấn luyện."""
    ids, cau = set(), set()
    for p in duong_dan_list:
        if not os.path.exists(p):
            print(f"  [cảnh báo] không thấy {p}", file=sys.stderr)
            continue
        n = 0
        for r in doc_jsonl(p):
            if r.get("id"):
                ids.add(r["id"])
            q = chuan_hoa(r.get("question", ""))
            if len(q) >= 15:           # câu quá ngắn dễ trùng ngẫu nhiên -> chỉ loại theo id
                cau.add(q)
            n += 1
        print(f"  nạp {n:5d} câu cấm từ {os.path.relpath(p, ROOT)}")
    return ids, cau


def hop_le(row):
    """Kiểm tra định dạng; trả về (messages sạch, lý do lỗi)."""
    ms = row.get("messages")
    if not isinstance(ms, list) or len(ms) < 2:
        return None, "thiếu messages"
    sach = []
    for m in ms:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            return None, "message sai định dạng"
        sach.append({"role": m["role"], "content": m["content"].strip()})
    if sach[-1]["role"] != "assistant" or not sach[-1]["content"]:
        return None, "không kết thúc bằng assistant có nội dung"
    if not any(m["role"] == "user" and m["content"] for m in sach):
        return None, "thiếu user"
    return sach, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(ROOT, "training_code/dsl/sft_raw.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--eval", default=os.path.join(ROOT, "data/eval300.jsonl"))
    ap.add_argument("--public", nargs="*", default=None,
                    help="file câu hỏi public (mặc định: mọi data/public*/questions.jsonl)")
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--tokenizer", default=None,
                    help="tên/đường dẫn tokenizer HF (vd Qwen/Qwen3-8B) để đếm token chính xác; bỏ trống = ước lượng theo số ký tự")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        sys.exit(f"Không thấy {args.raw}. Chạy script sinh dữ liệu (training_code/dsl) trước.")
    pub = args.public if args.public is not None else sorted(glob.glob(os.path.join(ROOT, "data/public*/questions.jsonl")))

    print("Nạp danh sách câu CẤM (eval300 + public):")
    ids_cam, cau_cam = nap_cau_cam([args.eval] + list(pub))
    if not cau_cam and not ids_cam:
        sys.exit("Không nạp được câu cấm nào -> dừng để tránh rò rỉ eval/public vào dữ liệu huấn luyện.")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def dem_token(ms):
        if tok is not None:
            return len(tok.apply_chat_template(ms, tokenize=True))
        return int(sum(len(m["content"]) for m in ms) / 2.5) + 20 * len(ms)   # tiếng Việt/JSON: ~2.5 ký tự/token (ước lượng thô)

    loai = defaultdict(int)
    giu, da_thay = [], set()
    for row in doc_jsonl(args.raw):
        ms, loi = hop_le(row)
        if loi:
            loai[loi] += 1
            continue
        rid, biz = row.get("id"), row.get("biz")
        if not biz:
            loai["thiếu biz"] += 1
            continue
        if rid in ids_cam:
            loai["trùng id eval300/public"] += 1
            continue
        user_all = chuan_hoa(" ".join(m["content"] for m in ms if m["role"] == "user"))
        if any(q in user_all for q in cau_cam):
            loai["nội dung chứa câu eval300/public"] += 1
            continue
        khoa = json.dumps(ms, ensure_ascii=False, sort_keys=True)
        if khoa in da_thay:
            loai["trùng lặp"] += 1
            continue
        da_thay.add(khoa)
        n_tok = dem_token(ms)
        if n_tok > args.max_tokens:
            loai[f"dài hơn {args.max_tokens} token"] += 1
            continue
        if "<think>" in ms[-1]["content"]:
            loai["assistant chứa <think> (giữ, nhưng kiểm tra lại)"] += 0   # chỉ để in cảnh báo bên dưới
            print(f"  [cảnh báo] id={rid}: câu trả lời có thẻ <think>", file=sys.stderr)
        giu.append({"biz": biz, "id": rid, "messages": ms, "n_tok": n_tok})

    print("\nĐã loại:")
    for k, v in sorted(loai.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {v:6d}  {k}")
    print(f"Còn lại: {len(giu)} ví dụ, {len({g['biz'] for g in giu})} shop")
    if len(giu) < 50:
        sys.exit("Quá ít ví dụ (<50) -> kiểm tra lại sft_raw.jsonl.")

    # ---- chia theo shop --------------------------------------------------------
    theo_biz = defaultdict(list)
    for g in giu:
        theo_biz[g["biz"]].append(g)
    shops = sorted(theo_biz)
    if len(shops) < 2:
        sys.exit("Chỉ có 1 shop -> không chia theo shop được.")
    random.Random(args.seed).shuffle(shops)
    muc_tieu = max(1, round(len(giu) * args.valid_frac))
    valid_shops, dem = [], 0
    for s in shops:
        if dem >= muc_tieu:
            break
        # không cho valid vượt quá nhiều so với mục tiêu (shop quá lớn thì thử shop khác)
        if dem + len(theo_biz[s]) > muc_tieu * 2 and valid_shops:
            continue
        valid_shops.append(s)
        dem += len(theo_biz[s])
    tap_valid = set(valid_shops)
    train = [g for g in giu if g["biz"] not in tap_valid]
    valid = [g for g in giu if g["biz"] in tap_valid]
    rng = random.Random(args.seed + 1)
    rng.shuffle(train)
    rng.shuffle(valid)

    os.makedirs(args.out_dir, exist_ok=True)
    for ten, ds in (("train", train), ("valid", valid)):
        p = os.path.join(args.out_dir, f"{ten}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for g in ds:
                f.write(json.dumps({"messages": g["messages"]}, ensure_ascii=False) + "\n")
        tb = sum(g["n_tok"] for g in ds) / max(1, len(ds))
        print(f"Ghi {p}: {len(ds)} ví dụ, TB {tb:.0f} token/ví dụ ({'đếm bằng tokenizer' if tok else 'ước lượng'})")
    print(f"Shop valid ({len(valid_shops)}): {', '.join(sorted(valid_shops))}")
    print(f"Shop train ({len(shops) - len(valid_shops)})")

    # kiểm tra lần cuối: không rò rỉ
    assert not ({g["biz"] for g in train} & {g["biz"] for g in valid}), "shop lọt cả train và valid"
    if tok is not None:
        print("\n=== MẪU ĐÃ ÁP CHAT TEMPLATE (ví dụ train đầu tiên) - kiểm tra thẻ <think> ở lượt assistant cuối ===")
        print(tok.apply_chat_template(train[0]["messages"], tokenize=False)[-700:])


if __name__ == "__main__":
    main()
