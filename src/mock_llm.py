"""LLM giả để thử đường ống (LLM_BACKEND=mock): lượt 1 in tên bảng, lượt 2 trả số dòng bảng đầu tiên."""


def mock_chat(messages):
    n_turn = sum(1 for m in messages if m["role"] == "assistant")
    if n_turn == 0:
        return "```python\nfor k, v in tables.items():\n    print(k, v.shape)\nprint(PERIODS)\n```", None
    return ('```python\nname = list(tables)[0]\nANSWER = {"type": "number", "answer": len(tables[name])}\n```',
            "mock thinking")
