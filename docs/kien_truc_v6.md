# Kiến trúc v6 — Agent do model dẫn dắt (model-driven)

Mục tiêu: model Qwen3 (≤ 15B) là bộ não; code Python chỉ là công cụ tính. Phải làm tốt cả private test (shop / cách hỏi mới), không chỉ public.
Bản dự phòng an toàn: v5.0 (public 0,865) = `AGENT_MODE=dieu_phoi` + `AGENT_ONLY_FORMULA=1`. v6 bật bằng `AGENT_MODE=v6`.

## 1. Luồng xử lý một câu

```
câu hỏi ─► [M1] model hiểu câu + lập kế hoạch → JSON "đặc tả truy vấn" (DSL, mục 2)  hoặc  {"loai": "tu_giai"} / {"loai": "khong_du_du_lieu"}
            │
            ├─ DSL ─► [T] dsl_engine.thuc_thi(dac_ta, biz, ...) — công cụ tính tổng quát trên bảng chuẩn
            │           kết quả + bảng trung gian ngắn ──► [M2] model đọc kết quả: chấp nhận / sửa đặc tả (tối đa 1 lần) / chuyển tự giải
            │
            ├─ tu_giai ─► [M3] model tự viết code pandas (vòng lặp code hiện có trong agent.py, sandbox hiện có)
            │
            └─► [V] kiểm chứng độc lập: so với công cụ công thức cũ (ct_*.py, cong_thuc.py) nếu chúng bắt được câu;
                     + kiểm định dạng / kiểu đáp án; + tín hiệu từ quá trình (số lượt sửa, lỗi code)
                 ──► [P] chính sách trả lời / bỏ qua: hiệu chỉnh trên dữ liệu train giữ riêng (KHÔNG dùng public)
```

Công cụ công thức cũ (ct_*.py) KHÔNG còn quyết định đáp án một mình; chúng là **công cụ kiểm chứng độc lập** và là công cụ model có thể gọi (đường dự phòng).

## 2. Đặc tả truy vấn (DSL) — model sinh, `src/dsl_engine.py` thực thi

JSON (mọi khoá tuỳ chọn trừ `loai`):

| khoá | giá trị | ý nghĩa |
|---|---|---|
| `loai` | `chi_so` \| `so_sanh` \| `ti_trong` \| `xep_hang` \| `dem_nguong` \| `theo_ky` \| `khong_du_du_lieu` \| `tu_giai` | dạng câu |
| `chi_so` | `doanh_thu` \| `so_don` \| `so_luong` \| `gia_tri_tb_don` \| `so_khach` \| `lai_gop` \| `so_don_huy` \| `tien_nhap` \| `so_phieu_nhap` \| `con_no` \| `thu_no` \| `no_them` \| `so_khach_no` \| `so_ngay_ban` | đại lượng |
| `ky` | cụm thời gian **chép nguyên văn từ câu** (vd `"tháng 6/2025"`, `"tuần trước"`, `"từ đầu năm đến hết hôm nay"`) | engine dùng `periods.find_periods(ky, as_of, cal)` |
| `ky_so_sanh` | cụm thời gian kỳ so sánh, hoặc `"ky_truoc"` / `"cung_ky_nam_truoc"` / `"cung_ky_thang_truoc"` | cho `so_sanh` |
| `loc` | `{"mat_hang": "...", "nhom_hang": "...", "kenh": "...", "khach": "...", "nhom_khach": "...", "nhan_vien": "...", "thanh_toan": "...", "ca": "sang\|chieu\|toi", "hoat_chat": "...", "ke_don": true\|false, "don_vi": "ly\|kg\|..."}` | điều kiện; tên chép từ câu, engine tự khớp với dữ liệu (bỏ dấu, gần đúng, nhiều kết quả → lỗi) |
| `nhom_theo` | `mat_hang` \| `khach` \| `nhan_vien` \| `ngay` \| `thang` \| `kenh` \| `nhom_hang` \| `gio` \| `nha_cung_cap` | cho `xep_hang` / `theo_ky` |
| `sap_xep` | `giam` \| `tang` | |
| `top_k` | số nguyên | 1 → trả string / date / number theo kiểu |
| `nguong` | `{"chi_so": "so_don\|doanh_thu\|...", "toi_thieu": số, "nghiem_ngat": false}` | cho `dem_nguong` (vd khách từ 3 đơn trở lên; ngày doanh thu ≥ 15 triệu) |
| `doi_tuong_dem` | `khach` \| `don` \| `ngay` \| `mat_hang` | cho `dem_nguong` |
| `kieu_dap_an` | `number` \| `percent` \| `string` \| `list` \| `table` \| `date` \| `boolean` | |

Ví dụ:
- "Tháng 6/2025 doanh thu nhóm cà phê bao nhiêu?" → `{"loai":"chi_so","chi_so":"doanh_thu","ky":"tháng 6/2025","loc":{"nhom_hang":"cà phê"},"kieu_dap_an":"number"}`
- "Doanh thu tháng này so với cùng kỳ tháng trước tăng mấy %?" → `{"loai":"so_sanh","chi_so":"doanh_thu","ky":"tháng này","ky_so_sanh":"cung_ky_thang_truoc","kieu_dap_an":"percent"}`
- "Top 3 khách mua nhiều tiền nhất tuần trước" → `{"loai":"xep_hang","chi_so":"doanh_thu","nhom_theo":"khach","sap_xep":"giam","top_k":3,"ky":"tuần trước","kieu_dap_an":"list"}`
- "Tháng 5/2025 có mấy khách mua từ 4 đơn trở lên?" → `{"loai":"dem_nguong","doi_tuong_dem":"khach","nguong":{"chi_so":"so_don","toi_thieu":4},"ky":"tháng 5/2025","kieu_dap_an":"number"}`

Định nghĩa đại lượng theo rulebook (§1 doanh thu, đơn huỷ/trả; §3 xếp hạng, xác định khách, hoà nhau theo mã; §4 công nợ; §5 nhập; §9 làm tròn & định dạng) — engine dùng lại bảng chuẩn `biz.ban`, `biz.nhap`, `biz.no_don`, `biz.thu_tien` và các hàm sẵn có.

## 3. Học (training)
1. **Sinh dữ liệu có kiểm chứng (rejection sampling / STaR):** model gốc (Qwen3-8B, few-shot) sinh DSL cho câu **train**; engine thực thi; chỉ giữ DSL cho ra đúng đáp án train (`scorer.score_one`). Câu không ra → ví dụ `tu_giai` có code đúng (nếu vòng code giải đúng) hoặc bỏ.
2. **SFT LoRA** (MLX trên Mac Apple Silicon) cho Qwen3 trên các cặp (câu + ngữ cảnh shop → kế hoạch + DSL). Chia train/validation **theo shop**; eval300 không dùng để huấn luyện.
3. Gộp adapter → GGUF → `ollama create` → model riêng trong Docker. Không tuyên bố đã fine-tune khi chưa chạy và chưa đánh giá checkpoint.

## 4. Đánh giá (bắt buộc có ablation)
A. chỉ công thức (v5, tắt model) · B. model tự giải (code) · C. model + DSL · D. hybrid đầy đủ · E. hybrid với model đã fine-tune.
Đo trên eval300 (train giữ riêng) + public: đúng / sai / bỏ, độ chính xác DSL, tỉ lệ câu model giải được mà công thức cũ KHÔNG bắt (đo khả năng tổng quát hoá), latency.
