# Django后端 Dockerfile - 多阶段构建优化版
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 使用阿里云镜像源加速apt包下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖和编译工具
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 生产阶段
FROM python:3.10-slim

WORKDIR /app

# 使用阿里云镜像源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 只安装运行时依赖
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从builder阶段复制已安装的Python包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/media /app/staticfiles /app/logs

# 设置适当的权限
RUN chmod +x /app/*.sh || true

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# 暴露端口
EXPOSE 8000

# 复制启动脚本
COPY entrypoint.sh /app/
COPY init_django.sh /app/
RUN chmod +x /app/entrypoint.sh /app/init_django.sh

# 设置entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# 默认运行Django初始化脚本 (可以被docker-compose command覆盖)
CMD ["/app/init_django.sh"]
