FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统层面依赖（librosa 依赖 ffmpeg，soundfile 依赖 libsndfile1）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 将本地预下载好的依赖包复制进容器（彻底绕开服务器龟速网络）
COPY wheels/ /tmp/wheels/

# 直接从本地 .whl 文件安装所有依赖，零网络下载
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels/ /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# 将代码复制进容器
COPY . .

# 暴露给容器外的端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
