"""
Gọi LLM qua HTTP, không cần cài thêm thư viện. Dùng chung cho cả máy cá nhân và Docker nộp bài.

Chọn bằng biến môi trường:
    LLM_BACKEND   ollama (mặc định, chạy trên Mac/PC) | openai (vLLM trong Docker, hoặc server OpenAI-compatible)
    LLM_BASE_URL  ollama: http://localhost:11434 ; openai: http://localhost:8000/v1
    LLM_MODEL     ollama: qwen3:8b ; vLLM: tên model khi chạy `vllm serve` (vd Qwen/Qwen3-14B)
    LLM_THINK     1 = cho model "suy nghĩ" trước khi trả lời (chậm hơn, thường đúng hơn); 0 = tắt
    LLM_NUM_CTX   độ dài ngữ cảnh cho Ollama (mặc định 24576; Ollama mặc định chỉ 4096 -> cắt mất prompt)
    LLM_TIMEOUT   giây chờ mỗi lần gọi (mặc định 300)
    LLM_MAX_TOKENS số token tối đa mỗi lượt (mặc định 1500, hoặc 5000 khi LLM_THINK=1)

    from llm import chat
    text, thinking = chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
BASE_URL = os.environ.get("LLM_BASE_URL",
                          "http://localhost:11434" if BACKEND == "ollama" else "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b" if BACKEND == "ollama" else "Qwen/Qwen3-14B")
THINK = os.environ.get("LLM_THINK", "0") == "1"
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "24576"))
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))
# số token tối đa model được viết mỗi lượt. Không suy nghĩ: 1 khối code ngắn là đủ (tránh model lặp vô tận);
# có suy nghĩ: cần dài hơn cho phần suy nghĩ.
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "5000" if THINK else "1500"))

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _split_think(text: str, thinking: str | None):
    """Tách phần <think>...</think> (nếu model nhét vào nội dung) ra khỏi câu trả lời."""
    text = text or ""
    found = _THINK_RE.findall(text)
    if found:
        thinking = (thinking or "") + "\n".join(found)
        text = _THINK_RE.sub("", text)
    if "</think>" in text:                      # trường hợp chỉ có thẻ đóng
        head, text = text.split("</think>", 1)
        thinking = (thinking or "") + head
    return text.strip(), (thinking.strip() if thinking else None)


# Nhiệt độ 0 + seed cố định: chạy lại cho cùng kết quả (BTC yêu cầu chạy lại lệch không quá ±0.03)
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
SEED = int(os.environ.get("LLM_SEED", "0"))


def chat(messages: list[dict], temperature: float | None = None, think: bool | None = None, seed: int | None = None):
    """-> (nội dung trả lời, suy nghĩ của model hoặc None)."""
    think = THINK if think is None else think
    temperature = TEMPERATURE if temperature is None else temperature
    seed = SEED if seed is None else seed
    if BACKEND == "ollama":
        max_tok = MAX_TOKENS if think == THINK else (5000 if think else 1500)
        payload = {"model": MODEL, "messages": messages, "stream": False, "think": think,
                   "options": {"num_ctx": NUM_CTX, "temperature": temperature, "num_predict": max_tok,
                               "repeat_penalty": 1.05}}
        if seed is not None:
            payload["options"]["seed"] = seed
        out = _post(f"{BASE_URL}/api/chat", payload)
        msg = out.get("message", {})
        return _split_think(msg.get("content", ""), msg.get("thinking"))
    if BACKEND == "openai":
        payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": MAX_TOKENS,
                   # Qwen3 trên vLLM: bật/tắt suy nghĩ qua chat template
                   "chat_template_kwargs": {"enable_thinking": think}}
        if seed is not None:
            payload["seed"] = seed
        out = _post(f"{BASE_URL}/chat/completions", payload)
        msg = out["choices"][0]["message"]
        return _split_think(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning"))
    if BACKEND == "mock":                       # chỉ để thử đường ống khi chưa có LLM
        from mock_llm import mock_chat
        return mock_chat(messages)
    raise ValueError(f"LLM_BACKEND không hợp lệ: {BACKEND}")


if __name__ == "__main__":                      # thử nhanh: python src/llm.py
    ans, th = chat([{"role": "user", "content": "Tính 15% của 2.400.000, chỉ ghi số."}])
    print("Suy nghĩ:", (th or "")[:300])
    print("Trả lời:", ans)
