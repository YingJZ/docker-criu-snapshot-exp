#!/bin/bash
# ============================================================
# 方案A（Podman 版）：PyTorch CPU推理 CRIU 快照实验
# ============================================================
# 自动化流程:
#   1. 构建 Podman 镜像
#   2. 启动容器（含 import torch + model load + 基线推理）
#   3. 执行 CRIU checkpoint
#   4. 停止容器并从 checkpoint 恢复
#   5. 触发推理验证，比较结果一致性
#   6. 输出性能对比报告
# ============================================================

set -euo pipefail

# ---- 配置 ----
IMAGE_NAME="pytorch-criu-cpu-podman"
CONTAINER_NAME="pytorch_criu_podman_demo"
CHECKPOINT_NAME="cp1"

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ---- 前置检查 ----
check_prerequisites() {
    info "检查前置条件..."

    if ! command -v podman &>/dev/null; then
        fail "未找到 podman 命令。请先安装 Podman: https://podman.io/getting-started/installation"
    fi
    ok "Podman 版本: $(podman --version)"

    if ! command -v criu &>/dev/null; then
        warn "宿主机未找到 criu 命令，Podman checkpoint 可能仍可用（Podman 可内置 CRIU）"
    else
        ok "CRIU 版本: $(criu --version 2>&1 | head -1)"
    fi

    # Podman 不需要 experimental flag，原生支持 checkpoint/restore
    # 但需要检查 Podman 版本是否支持 (>= 3.0)
    local podman_major
    podman_major=$(podman --version | grep -oP '\d+' | head -1)
    if [ "${podman_major:-0}" -lt 3 ]; then
        fail "Podman 版本过低 (需要 >= 3.0 以支持 container checkpoint)"
    fi
    ok "Podman checkpoint/restore 支持已确认"
}

# ---- 清理旧容器 ----
cleanup() {
    info "清理旧容器和检查点..."
    podman rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}

# ---- 步骤1: 构建镜像 ----
build_image() {
    info "步骤1: 构建 Podman 镜像..."
    podman build -t "${IMAGE_NAME}" .
    ok "镜像构建完成: ${IMAGE_NAME}"
}

# ---- 步骤2: 启动容器 ----
start_container() {
    info "步骤2: 启动容器..."
    podman run -d \
        --name "${CONTAINER_NAME}" \
        --security-opt seccomp=unconfined \
        "${IMAGE_NAME}"

    info "等待模型加载和基线推理完成..."
    local timeout=60
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if podman logs "${CONTAINER_NAME}" 2>&1 | grep -q "\[READY\]"; then
            ok "模型加载完成，进程进入等待状态"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "模型加载超时 (${timeout}s)"
}

# ---- 步骤3: 记录正常启动耗时 + 触发基线推理确认 ----
measure_cold_start() {
    info "步骤3: 记录冷启动信息..."

    local timing_json
    timing_json=$(podman exec "${CONTAINER_NAME}" cat /app/results/timing.json 2>/dev/null || echo "{}")
    echo "${timing_json}" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    if data:
        print(f'    模型加载耗时: {data.get(\"phase1_model_load_s\", \"N/A\")}s')
        print(f'    基线推理耗时: {data.get(\"phase2_baseline_inference_s\", \"N/A\")}s')
        print(f'    总初始化耗时: {data.get(\"total_init_s\", \"N/A\")}s')
        print(f'    模型参数量:   {data.get(\"param_count\", \"N/A\"):,}')
        print(f'    模型大小:     {data.get(\"model_size_mb\", \"N/A\")} MB')
except:
    print('    (无法解析计时数据)')
"

    ok "冷启动数据已记录"
}

# ---- 步骤4: 创建 CRIU checkpoint ----
create_checkpoint() {
    info "步骤4: 创建 CRIU checkpoint..."
    local cp_start
    cp_start=$(date +%s%N)

    # Podman checkpoint: 容器自动停止，checkpoint 保存在容器存储中
    podman container checkpoint "${CONTAINER_NAME}"

    local cp_end
    cp_end=$(date +%s%N)
    local cp_ms=$(( (cp_end - cp_start) / 1000000 ))
    ok "Checkpoint 创建完成，耗时: ${cp_ms}ms"
    CHECKPOINT_TIME_MS=${cp_ms}
}

# ---- 步骤5: 从 checkpoint 恢复容器 ----
restore_container() {
    info "步骤5: 从 checkpoint 恢复容器..."

    local restore_start
    restore_start=$(date +%s%N)

    # Podman restore: 从 checkpoint 恢复已停止的容器
    podman container restore "${CONTAINER_NAME}"

    local restore_end
    restore_end=$(date +%s%N)
    local restore_ms=$(( (restore_end - restore_start) / 1000000 ))
    ok "Container 从 checkpoint 恢复完成，耗时: ${restore_ms}ms"
    RESTORE_TIME_MS=${restore_ms}
}

# ---- 步骤6: 触发推理验证 ----
verify_inference() {
    info "步骤6: 触发推理验证（通过文件触发器）..."

    sleep 2

    podman exec "${CONTAINER_NAME}" touch /app/trigger_inference

    local timeout=30
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if podman exec "${CONTAINER_NAME}" test -f /app/results/verification.json 2>/dev/null; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ $elapsed -ge $timeout ]; then
        fail "推理验证超时"
    fi

    info "推理验证结果:"
    podman exec "${CONTAINER_NAME}" cat /app/results/verification.json | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    match = data.get('match', False)
    if match:
        print('  推理结果完全一致! CRIU restore 后模型可用')
    else:
        print('  推理结果不一致!')
        if 'error' in data:
            print(f'  错误: {data[\"error\"]}')
        else:
            print(f'  差异详情: {data.get(\"diffs\", {})}')
except Exception as e:
    print(f'  (解析验证结果失败: {e})')
"
}

# ---- 步骤7: 性能报告 ----
report() {
    info "步骤7: 生成性能报告..."

    local timing_json
    timing_json=$(podman exec "${CONTAINER_NAME}" cat /app/results/timing.json 2>/dev/null || echo "{}")
    local cold_start_s
    cold_start_s=$(echo "${timing_json}" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('total_init_s', 'N/A'))" 2>/dev/null || echo "N/A")

    local restore_s="N/A"
    if [ -n "${RESTORE_TIME_MS:-}" ]; then
        restore_s=$(python3 -c "print(${RESTORE_TIME_MS}/1000)")
    fi

    echo ""
    echo "============================================================"
    echo "  方案A（Podman 版）：PyTorch CPU推理 CRIU 冷启动加速"
    echo "============================================================"
    echo ""
    echo "  冷启动 (无 CRIU):"
    echo "    import torch + model.load() + 基线推理:  ${cold_start_s}s"
    echo ""
    echo "  快照恢复 (有 CRIU):"
    echo "    CRIU checkpoint 创建耗时:               $((CHECKPOINT_TIME_MS:-0))ms"
    echo "    CRIU restore 耗时:                       ${restore_s}s"
    echo ""
    if [ "${cold_start_s}" != "N/A" ] && [ "${restore_s}" != "N/A" ]; then
        local speedup
        speedup=$(python3 -c "print(f'{$cold_start_s / $restore_s:.1f}x')")
        echo "  加速比: ${speedup}"
    fi
    echo ""
    echo "  验证状态: $(podman exec "${CONTAINER_NAME}" cat /app/results/verification.json 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('通过' if d.get('match') else '失败')" 2>/dev/null || echo '未完成')"
    echo "============================================================"
    echo ""

    info "容器日志 (最后20行):"
    podman logs --tail 20 "${CONTAINER_NAME}"
}

# ---- 主流程 ----
main() {
    echo ""
    echo "============================================================"
    echo "  方案A（Podman 版）：PyTorch CPU推理 + CRIU 快照加速冷启动"
    echo "============================================================"
    echo ""

    check_prerequisites
    cleanup
    build_image
    start_container
    measure_cold_start
    create_checkpoint
    restore_container
    verify_inference
    report

    ok "实验完成!"
}

main "$@"
