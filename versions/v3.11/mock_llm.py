"""LLM giả để thử đường ống (LLM_BACKEND=mock): lượt 1 in tên bảng, lượt 2 trả số dòng bảng đầu tiên."""


import os
import time


def mock_chat(messages):
    time.sleep(float(os.environ.get("MOCK_SLEEP", "0")))   # giả lập model chậm để thử cơ chế hạn giờ
    n_turn = sum(1 for m in messages if m["role"] == "assistant")
    if n_turn == 0:
        return "```python\nfor k, v in tables.items():\n    print(k, v.shape)\nprint(PERIODS)\n```", None
    return ('```python\nname = list(tables)[0]\nANSWER = {"type": "number", "answer": len(tables[name])}\n```',
            "mock thinking")
