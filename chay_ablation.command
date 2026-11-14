#!/bin/zsh -l
# ABLATION trên 300 câu train giữ riêng (docs/kien_truc_v6.md §4): đo model đóng góp bao nhiêu.
#   A  = AGENT_MODE=dieu_phoi + AGENT_ONLY_FORMULA=1 (v5.0: công thức dẫn dắt, model chỉ chọn công cụ / kiểm duyệt)
#   C  = v6, V6_TAT_DOI_CHIEU=1 (model + DSL, KHÔNG đối chiếu công thức cũ)
#   D  = v6 đầy đủ (model + DSL + tu_giai + đối chiếu công thức + chính sách)
#   (tuỳ chọn) A0 = không dùng model (AGENT_MODE rỗng, ONLY_FORMULA=1: chỉ công thức) · B = v6 chỉ tu_giai, không đối chiếu
# Cách dùng:  ./chay_ablation.command                       -> chạy A C D tuần tự (mỗi chế độ 300 câu; làm tiếp được nếu bị ngắt), rồi in bảng so sánh
#             MODES="A0 A C D" ./chay_ablation.command      (chọn chế độ)    ·   LLM_MODEL=qwen3:4b ./chay_ablation.command
#   Kết quả: outputs/eval300_abl_<chế độ>_<model>. Cần src/chinh_sach_v6.json (hieu_chinh.py) cho D nếu muốn chính sách đã hiệu chỉnh.
#   Lưu ý: bảng hiệu chỉnh trên chính eval300 thì số D trên eval300 lạc quan -> dùng tập giữ riêng (tao_giu_rieng.py, IDS=data/giu_rieng.jsonl).
#   Chạy XONG tất cả: máy ngủ sau 60 giây (NGU=0 để tắt).
cd "$(dirname "$0")"
PY=.venv/bin/python
export LLM_MODEL=${LLM_MODEL:-qwen3:8b}
IDS=${IDS:-data/eval300.jsonl}
TEN=$(echo "$LLM_MODEL" | tr '/:' '__')
MODES=${MODES:-"A C D"}
RUNS=()
for M in ${=MODES}; do
  case $M in
    A0) OUT=outputs/eval300_abl_A0_congthuc;     ENVS=(AGENT_MODE= AGENT_ONLY_FORMULA=1) ;;
    A)  OUT=outputs/eval300_abl_A_$TEN;          ENVS=(AGENT_MODE=dieu_phoi AGENT_ONLY_FORMULA=1) ;;
    B)  OUT=outputs/eval300_abl_B_$TEN;          ENVS=(AGENT_MODE=v6 V6_CHI_TU_GIAI=1 V6_TAT_DOI_CHIEU=1) ;;
    C)  OUT=outputs/eval300_abl_C_$TEN;          ENVS=(AGENT_MODE=v6 V6_TAT_DOI_CHIEU=1) ;;
    D)  OUT=outputs/eval300_abl_D_$TEN;          ENVS=(AGENT_MODE=v6) ;;
    *)  echo "Chế độ lạ: $M (dùng A0 A B C D)"; exit 1 ;;
  esac
  echo "################ chế độ $M -> $OUT"
  env "${ENVS[@]}" OUT=$OUT IDS=$IDS NGU=0 ./chay_v6_300.command
  RUNS+=($OUT)
done
$PY training_code/dsl/bang_v6.py "${RUNS[@]}" --eval $IDS 2>&1 | tee outputs/eval300_abl_so_sanh_$TEN.txt
if [ "${NGU:-1}" = "1" ]; then
  echo "Xong ablation. Máy sẽ ngủ sau 60 giây (bấm Ctrl+C để huỷ)..."
  sleep 60 && pmset sleepnow
fi
