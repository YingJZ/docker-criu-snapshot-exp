FROM docker.xuanyuan.me/ubuntu:22.04

# 避免交互式提示卡住构建
ENV DEBIAN_FRONTEND=noninteractive

# 仅安装 Python3（CRIU 安装在宿主机上，容器内不需要）
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get clean

WORKDIR /app
COPY test_app.py .

# 容器以后台方式运行测试脚本
CMD ["python3", "test_app.py"]
