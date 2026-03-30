FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统层面依赖（librosa 依赖 ffmpeg，soundfile 依赖 libsndfile1）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件并安装（利用 Docker 缓存层加速构建）
COPY requirements.txt .
# 使用清华大学的 pip 镜像源加速下载
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 将代码复制进容器（由于我们在 .dockerignore 中排除了一些文件夹，这里只会打包纯项目代码文件）
COPY . .

# 暴露给容器外的端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
