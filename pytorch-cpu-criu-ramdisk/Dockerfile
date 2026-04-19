# ============================================================
# 方案A：纯CPU推理 -- CRIU快照加速PyTorch冷启动
# ============================================================
# 基于本地已有的 torch-cpu 镜像 (PyTorch 2.5.1 CPU + torchvision)
# CRIU 运行在宿主机上，容器内不需要安装 CRIU
# ============================================================

FROM torch-cpu:latest

WORKDIR /app

# ---- 预下载 ResNet-50 权重 ----
# 方式1 (推荐): 从本地拷贝预下载的权重文件
COPY models/resnet50-11ad3fa6.pth /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
ENV TORCH_HOME=/root/.cache/torch

# 方式2: 构建时在线下载 (需要网络能访问 download.pytorch.org，或通过 --build-arg http_proxy 传代理)
#   如果有权重文件缓存则跳过下载，否则尝试下载
RUN if [ ! -f /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ]; then \
        python3 -c "import torchvision.models as models; models.resnet50(weights=models.ResNet50_Weights.DEFAULT); print('ResNet-50 weights downloaded')" \
        || echo "[WARN] ResNet-50 权重下载失败，运行时将回退到自定义小型CNN模型"; \
    else \
        echo "[INFO] ResNet-50 权重已存在，跳过下载"; \
    fi

# 拷贝推理脚本
COPY inference_app.py .

# 创建结果输出目录
RUN mkdir -p /app/results

# 容器以后台方式运行推理脚本
CMD ["python3", "inference_app.py"]
