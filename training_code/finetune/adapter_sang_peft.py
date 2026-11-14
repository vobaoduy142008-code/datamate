#!/usr/bin/env python3
"""Đổi adapter LoRA của MLX (adapters.safetensors + adapter_config.json) sang định dạng PEFT (HuggingFace)
để llama.cpp `convert_lora_to_gguf.py` đọc được.

MLX : key  model.layers.N.<module>.lora_a  (in x r)   và  ...lora_b  (r x out);  đầu ra = y + scale * (x @ a @ b)
PEFT: key  base_model.model.model.layers.N.<module>.lora_A.weight (r x in)  và  ...lora_B.weight (out x r);
      đầu ra = y + (lora_alpha / r) * B(A(x))   =>  lora_alpha = scale * r

Dùng:  .venv/bin/python adapter_sang_peft.py <thư_mục_adapter_mlx> <thư_mục_ra> [--base-id Qwen/Qwen3-8B] [--file checkpoint.safetensors]
"""
import argparse
import json
import os
import re
import sys

import numpy as np


def doc_tensor(path):
    """Trả về dict tên -> numpy float32 (MLX có thể lưu bf16, numpy không đọc được bf16 nên đi qua mlx nếu có)."""
    try:
        import mlx.core as mx
        return {k: np.array(v.astype(mx.float32)) for k, v in mx.load(path).items()}
    except ImportError:
        from safetensors.numpy import load_file
        return {k: v.astype(np.float32) for k, v in load_file(path).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--base-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--file", default=None, help="checkpoint cụ thể thay cho adapters.safetensors")
    a = ap.parse_args()

    cfg_path = os.path.join(a.adapter_dir, "adapter_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    lp = cfg.get("lora_parameters", {})
    scale = float(lp.get("scale", 20.0))
    src = a.file or os.path.join(a.adapter_dir, "adapters.safetensors")
    ten = doc_tensor(src)

    pat = re.compile(r"^(?:.*?\.)?(layers\.\d+\..+)\.lora_(a|b)$")
    out, mods, rank = {}, set(), None
    for k, v in ten.items():
        m = pat.match(k)
        if not m:
            print(f"  bỏ qua key lạ: {k}", file=sys.stderr)
            continue
        mod, ab = m.group(1), m.group(2)
        mods.add(mod.rsplit(".", 1)[-1])
        if ab == "a":      # (in, r) -> (r, in)
            out[f"base_model.model.model.{mod}.lora_A.weight"] = np.ascontiguousarray(v.T)
            rank = v.shape[1]
        else:              # (r, out) -> (out, r)
            out[f"base_model.model.model.{mod}.lora_B.weight"] = np.ascontiguousarray(v.T)
    if not out or rank is None:
        sys.exit("Không tìm thấy tensor LoRA nào trong " + src)

    os.makedirs(a.out_dir, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(out, os.path.join(a.out_dir, "adapter_model.safetensors"))
    peft_cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.base_id,
        "r": int(rank), "lora_alpha": scale * rank, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted(mods), "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
        "inference_mode": True, "init_lora_weights": True,
    }
    json.dump(peft_cfg, open(os.path.join(a.out_dir, "adapter_config.json"), "w"), indent=2)
    print(f"OK: {len(out)} tensor, rank={rank}, scale_mlx={scale} -> lora_alpha={scale*rank}, modules={sorted(mods)}")
    print("ví dụ key:", next(iter(out)))


if __name__ == "__main__":
    main()
