# DataMate – Zalo AI Challenge 2026
# Base image theo Docker guideline của BTC (vLLM 0.10.1.1, CUDA 12.8, PyTorch 2.7.1).
FROM vllm/vllm-openai:v0.10.1.1

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# 1) Thư viện của agent: cài vào thư mục riêng để không đổi phiên bản numpy/pandas mà vLLM đang dùng
COPY requirements-docker.txt /code/requirements-docker.txt
RUN python3 -m pip install --no-cache-dir --target /opt/agent/lib -r /code/requirements-docker.txt

# 2) Trọng số model (tải trước bằng scripts/tai_model.sh vào models/Qwen3-14B). Đặt trước code để
#    khi chỉ sửa code thì Docker dùng lại lớp này, build lại rất nhanh.
COPY models/Qwen3-14B /models/Qwen3-14B

# 3) Dữ liệu train (dùng để đo điểm; không fine-tune) và code
COPY data/train /code/data/train
COPY README.md requirements.txt requirements-docker.txt Dockerfile /code/
COPY scripts /code/scripts
COPY src /code/src
COPY predict.sh /code/predict.sh
RUN sed -i 's/\r$//' /code/predict.sh /code/scripts/*.sh && chmod +x /code/predict.sh /code/scripts/*.sh \
 && PYTHONPATH=/opt/agent/lib python3 -c "import pandas, openpyxl, python_calamine; print('pandas', pandas.__version__)"

WORKDIR /code
# Ảnh gốc của vLLM tự bật server khi chạy; bỏ đi để BTC chạy đúng /code/predict.sh
ENTRYPOINT []
CMD ["/code/predict.sh"]
