FROM ubuntu:22.04

# 避免交互式提示卡住构建
ENV DEBIAN_FRONTEND=noninteractive

# 安装 criu 和 python
RUN apt-get update && \
    apt-get install -y criu python3 && \
    apt-get clean

WORKDIR /app
COPY test_app.py .

# 默认进入 bash
CMD ["/bin/bash"]
