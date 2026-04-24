FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY wheels/ /tmp/wheels/

RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels/ /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# 白名单 COPY —— 仅复制必要文件，防止 .venv 等大文件膨胀镜像
COPY main.py search.py model.py dataset.py train.py batch_extract.py config.py preprocessing.py ./
COPY index.html favicon.svg ./
COPY init_db.py make_noise.py ./
COPY requirements.txt ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
