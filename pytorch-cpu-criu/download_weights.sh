#!/bin/bash
# ============================================================
# 预下载 ResNet-50 权重文件（国内网络环境推荐）
# ============================================================
# 下载后权重文件放在 models/ 目录，Dockerfile 会 COPY 到镜像中
# 避免 Docker build 时因网络不通导致权重下载失败
# ============================================================

set -euo pipefail

WEIGHTS_FILE="resnet50-11ad3fa6.pth"
WEIGHTS_URL="https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
MODELS_DIR="$(dirname "$0")/models"
TARGET_PATH="${MODELS_DIR}/${WEIGHTS_FILE}"

mkdir -p "${MODELS_DIR}"

if [ -f "${TARGET_PATH}" ]; then
    echo "权重文件已存在: ${TARGET_PATH}"
    echo "如需重新下载，请先删除该文件"
    exit 0
fi

echo "下载 ResNet-50 权重文件..."
echo "  源: ${WEIGHTS_URL}"
echo "  目标: ${TARGET_PATH}"
echo ""

# 尝试直连下载
if wget -O "${TARGET_PATH}" "${WEIGHTS_URL}"; then
    echo ""
    echo "下载完成!"
    echo ""
    echo "下一步: 取消 Dockerfile 中 COPY models/ 行的注释，重新构建镜像"
else
    rm -f "${TARGET_PATH}"
    echo ""
    echo "直连下载失败，请尝试以下方式:"
    echo ""
    echo "1. 使用代理下载:"
    echo "   https_proxy=http://YOUR_PROXY:PORT wget -O ${TARGET_PATH} ${WEIGHTS_URL}"
    echo ""
    echo "2. 手动下载后放到:"
    echo "   ${TARGET_PATH}"
    echo ""
    echo "3. 跳过 ResNet-50，使用内置 SmallCNN 模型（无需下载，但模型较小）"
    echo "   直接 docker build 即可，推理脚本会自动回退"
    exit 1
fi
