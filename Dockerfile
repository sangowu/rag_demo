# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（faiss / chromadb 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY src/ ./src/
COPY config/ ./config/

# data/ 和 models/ 通过 volume 挂载，不打包进镜像

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
