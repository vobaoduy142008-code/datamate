# DataMate – Zalo AI Challenge 2026
# Model: Qwen3-8B (bản "qwen3:8b" của thư viện Ollama, lượng tử hoá 4-bit, Apache 2.0), chạy bằng Ollama trên GPU NVIDIA.
# Dùng ĐÚNG phiên bản Ollama và đúng model như máy của đội để kết quả chạy lại giống nhau.
#
# Build (trên PC, trong thư mục datamate):
#   docker build --platform linux/amd64 --build-arg OLLAMA_VERSION=<số phiên bản> -t zaic2026_datamate_<ten_doi>:final .
ARG OLLAMA_VERSION
FROM ollama/ollama:${OLLAMA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    OLLAMA_MODELS=/models/ollama OLLAMA_HOST=127.0.0.1:11434 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1

# 1) curl + uv (để cài Python 3.12 riêng cho agent, không phụ thuộc Python của image gốc)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 2) Python 3.12 + thư viện của agent (cùng phiên bản với máy phát triển)
COPY requirements-docker.txt /code/requirements-docker.txt
RUN uv venv --python 3.12 /opt/agent \
 && VIRTUAL_ENV=/opt/agent uv pip install -r /code/requirements-docker.txt \
 && /opt/agent/bin/python -c "import pandas, openpyxl, python_calamine; print('pandas', pandas.__version__)"

# 3) Trọng số model: tải về ngay trong lúc build, nằm sẵn trong image (khi chạy chấm thì mạng tắt)
ARG MODEL=qwen3:8b
RUN (ollama serve > /tmp/ollama_build.log 2>&1 &) && sleep 8 \
 && ollama pull ${MODEL} && ollama list && ollama show ${MODEL} --modelfile | head -n 5

# 4) Dữ liệu train (dùng để đo điểm và dựng bảng thêm dấu cho câu hỏi gõ không dấu; không huấn luyện LLM) và code
COPY data/train /code/data/train
COPY README.md requirements.txt requirements-docker.txt Dockerfile /code/
COPY scripts /code/scripts
COPY src /code/src
COPY docs /code/docs
COPY training_code /code/training_code
COPY predict.sh /code/predict.sh
RUN sed -i 's/\r$//' /code/predict.sh /code/scripts/*.sh && chmod +x /code/predict.sh /code/scripts/*.sh
# bảng thêm dấu (src/ngram_dau.json) dựng từ data/train ngay lúc build
RUN rm -f /code/src/ngram_dau.json && cd /code/src && /opt/agent/bin/python -c "import them_dau; m = them_dau.base_model(); print('them_dau', len(m.uni), len(m.bi))"
# danh sách dữ liệu + SHA-256 (BTC yêu cầu training_code/data_manifest.json)
RUN /opt/agent/bin/python /code/scripts/tao_manifest.py /code /code/training_code/data_manifest.json

WORKDIR /code
# Image gốc của Ollama tự chạy "ollama serve"; bỏ đi để BTC chạy đúng /code/predict.sh
ENTRYPOINT []
CMD ["/code/predict.sh"]
