"""
Agent: nhận 1 câu hỏi, trả về đáp án.

PHIÊN BẢN 0 (baseline rỗng): luôn trả lời "abstain" (PA = 0).
Mục đích: kiểm tra cả quy trình (đọc câu hỏi -> ghi file nộp -> chấm điểm) chạy đúng
trước khi thêm LLM. Các bước sau sẽ thay phần thân hàm solve().

Hợp đồng (interface) cả đội dùng chung — KHÔNG đổi tên/kiểu trả về khi chưa báo cả nhóm:
    solve(question: dict, data_dir: str) -> dict
        question = {"id", "business_id", "question", "as_of_date"}
        trả về   = {
            "answer":   {"type": "...", "answer": ...}   # đúng định dạng submission
            "thinking": str | list[str] | None,          # suy nghĩ của model (ghi log)
            "code":     str | None,                       # code đã chạy (ghi log)
            "output":   object | None,                    # kết quả code (ghi log)
        }
Lưu ý luật: KHÔNG được viết logic theo id câu hỏi hay business_id.
"""


def solve(question: dict, data_dir: str) -> dict:
    return {"answer": {"type": "abstain"}, "thinking": None, "code": None, "output": None}
