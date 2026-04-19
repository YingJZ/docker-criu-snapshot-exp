#!/bin/bash
# ============================================================
# 方案A-优化：PyTorch CPU推理 CRIU 快照 + RAM Disk 加速
# ============================================================
# 核心思路：
#   原始实验中 CRIU restore (9.23s) 慢于冷启动 (2.48s)，
#   瓶颈在磁盘 I/O（307MB checkpoint 从磁盘读取）。
#   本实验将 checkpoint 存放在 tmpfs（内存盘）上，
#   消除磁盘 I/O 瓶颈，验证 restore 能否快于冷启动。
#
# 实验对比：
#   1. 常规冷启动（baseline）
#   2. CRIU restore — checkpoint 在普通磁盘
#   3. CRIU restore — checkpoint 在 RAM disk (tmpfs)
#   每组跑 N 次取平均值，输出统计报告
# ============================================================

set -euo pipefail

# ---- 配置 ----
IMAGE_NAME="pytorch-criu-cpu-ramdisk"
CONTAINER_NAME="pytorch_criu_ramdisk"
CHECKPOINT_NAME="cp1"

# RAM disk 路径（tmpfs 挂载点）
RAMDISK_DIR="/tmp/criu-ramdisk"
# 普通磁盘 checkpoint 路径
DISK_CHECKPOINT_DIR="/tmp/criu-disk-checkpoints"

# 每组迭代次数（默认3次，可通过参数覆盖）
ITERATIONS=${1:-3}

# 结果输出
RESULT_FILE="/tmp/criu_ramdisk_benchmark_$(date +%Y%m%d_%H%M%S).json"

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ---- 用于收集结果的数组 ----
COLD_START_TIMES=()
DISK_RESTORE_TIMES=()
RAM_RESTORE_TIMES=()
CHECKPOINT_CREATE_TIMES=()
CHECKPOINT_SIZE=""

# ---- 前置检查 ----
check_prerequisites() {
    info "检查前置条件..."

    # 检查 Docker 实验特性
    if ! docker version -f '{{.Server.Experimental}}' 2>/dev/null | grep -q "true"; then
        fail "Docker 实验特性未启用。请在 /etc/docker/daemon.json 中添加 {\"experimental\": true} 并重启 Docker"
    fi
    ok "Docker 实验特性已启用"

    # 检查 CRIU
    if ! command -v criu &>/dev/null; then
        warn "宿主机未找到 criu 命令，Docker checkpoint 可能仍可用（Docker 内置 CRIU）"
    else
        ok "CRIU 版本: $(criu --version 2>&1 | head -1)"
    fi

    # 检查是否有足够内存用于 tmpfs（建议 >= 2GB 空闲）
    local free_mem_kb
    free_mem_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local free_mem_mb=$((free_mem_kb / 1024))
    if [ "$free_mem_mb" -lt 1024 ]; then
        warn "可用内存较少 (${free_mem_mb}MB)，tmpfs 可能不够存放 checkpoint (~307MB for ResNet-50)"
    else
        ok "可用内存: ${free_mem_mb}MB，足够存放 checkpoint"
    fi
}

# ---- 准备 RAM disk (tmpfs) ----
setup_ramdisk() {
    step "准备 RAM disk (tmpfs)..."

    # 如果已挂载，先卸载
    if mountpoint -q "${RAMDISK_DIR}" 2>/dev/null; then
        info "RAM disk 已挂载，先卸载..."
        sudo umount "${RAMDISK_DIR}"
    fi

    # 创建目录
    sudo mkdir -p "${RAMDISK_DIR}"

    # 挂载 tmpfs，限制大小为 1GB（足够 ResNet-50 的 307MB checkpoint）
    sudo mount -t tmpfs -o size=1G tmpfs "${RAMDISK_DIR}"

    # 验证
    if mountpoint -q "${RAMDISK_DIR}"; then
        ok "RAM disk 已挂载: ${RAMDISK_DIR} (tmpfs, size=1G)"
        df -h "${RAMDISK_DIR}"
    else
        fail "RAM disk 挂载失败"
    fi
}

# ---- 准备普通磁盘 checkpoint 目录 ----
setup_disk_dir() {
    step "准备普通磁盘 checkpoint 目录..."
    mkdir -p "${DISK_CHECKPOINT_DIR}"
    ok "磁盘 checkpoint 目录: ${DISK_CHECKPOINT_DIR}"
}

# ---- 清理旧容器 ----
cleanup() {
    info "清理旧容器..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}

# ---- 构建 Docker 镜像 ----
build_image() {
    step "构建 Docker 镜像..."
    docker build -t "${IMAGE_NAME}" .
    ok "镜像构建完成: ${IMAGE_NAME}"
}

# ---- 等待容器就绪 ----
wait_for_ready() {
    local timeout=60
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "\[READY\]"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "容器就绪超时 (${timeout}s)"
}

# ---- 读取冷启动计时 ----
get_cold_start_time() {
    local timing_json
    timing_json=$(docker exec "${CONTAINER_NAME}" cat /app/results/timing.json 2>/dev/null || echo "{}")
    echo "${timing_json}" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    print(f'{data.get(\"total_init_s\", \"N/A\")}')
except:
    print('N/A')
"
}

# ---- 获取 checkpoint 大小 ----
get_checkpoint_size() {
    local cp_dir="$1"
    if [ -d "${cp_dir}/${CHECKPOINT_NAME}" ]; then
        du -sh "${cp_dir}/${CHECKPOINT_NAME}" 2>/dev/null | awk '{print $1}'
    else
        echo "N/A"
    fi
}

# ---- 验证推理结果 ----
verify_inference() {
    # 等待容器完全恢复
    sleep 2

    # 清除旧的验证结果
    docker exec "${CONTAINER_NAME}" rm -f /app/results/verification.json 2>/dev/null || true

    # 触发推理
    docker exec "${CONTAINER_NAME}" touch /app/trigger_inference

    # 等待验证完成
    local timeout=30
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if docker exec "${CONTAINER_NAME}" test -f /app/results/verification.json 2>/dev/null; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ $elapsed -ge $timeout ]; then
        echo "false"
        return 1
    fi

    # 检查验证结果
    local result
    result=$(docker exec "${CONTAINER_NAME}" cat /app/results/verification.json | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    print('true' if data.get('match', False) else 'false')
except:
    print('false')
" 2>/dev/null)
    echo "${result}"
}

# ============================================================
# 实验1: 冷启动基线（多次测量）
# ============================================================
benchmark_cold_start() {
    step "========== 实验1: 冷启动基线 =========="
    info "将进行 ${ITERATIONS} 次冷启动测量..."

    for i in $(seq 1 "${ITERATIONS}"); do
        info "--- 冷启动 第 ${i}/${ITERATIONS} 次 ---"

        # 清理
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

        # 启动容器并计时
        local start_ns
        start_ns=$(date +%s%N)

        docker run -d \
            --name "${CONTAINER_NAME}" \
            --security-opt seccomp=unconfined \
            "${IMAGE_NAME}"

        wait_for_ready

        local end_ns
        end_ns=$(date +%s%N)

        local elapsed_s
        elapsed_s=$(python3 -c "print(f'{(${end_ns} - ${start_ns}) / 1e9:.3f}')")

        # 也读取容器内部的计时
        local internal_time
        internal_time=$(get_cold_start_time)

        COLD_START_TIMES+=("${elapsed_s}")

        ok "冷启动 第${i}次: wall=${elapsed_s}s, internal=${internal_time}s"

        # 停掉容器，为下次准备
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    done

    ok "冷启动基线测量完成"
}

# ============================================================
# 实验2: CRIU restore — checkpoint 在普通磁盘
# ============================================================
benchmark_disk_restore() {
    step "========== 实验2: CRIU restore (普通磁盘) =========="
    info "将进行 ${ITERATIONS} 次磁盘 restore 测量..."

    # 先启动容器并创建 checkpoint
    info "准备: 启动容器 + 创建 checkpoint..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --security-opt seccomp=unconfined \
        "${IMAGE_NAME}"

    wait_for_ready

    # 记录冷启动时间
    local internal_time
    internal_time=$(get_cold_start_time)
    info "本次冷启动内部计时: ${internal_time}s"

    # 创建 checkpoint（指定磁盘目录）
    local cp_start_ns
    cp_start_ns=$(date +%s%N)

    docker checkpoint create \
        --checkpoint-dir "${DISK_CHECKPOINT_DIR}" \
        "${CONTAINER_NAME}" "${CHECKPOINT_NAME}"

    local cp_end_ns
    cp_end_ns=$(date +%s%N)
    local cp_s
    cp_s=$(python3 -c "print(f'{(${cp_end_ns} - ${cp_start_ns}) / 1e9:.3f}')")
    CHECKPOINT_CREATE_TIMES+=("${cp_s}")

    # 记录 checkpoint 大小
    CHECKPOINT_SIZE=$(get_checkpoint_size "${DISK_CHECKPOINT_DIR}")
    info "Checkpoint 大小: ${CHECKPOINT_SIZE}"
    info "Checkpoint 创建耗时: ${cp_s}s"

    # 停止容器
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true

    # 多次 restore 测量
    for i in $(seq 1 "${ITERATIONS}"); do
        info "--- 磁盘 Restore 第 ${i}/${ITERATIONS} 次 ---"

        # 确保容器已停止
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        sleep 1

        local restore_start_ns
        restore_start_ns=$(date +%s%N)

        docker start \
            --checkpoint "${CHECKPOINT_NAME}" \
            --checkpoint-dir "${DISK_CHECKPOINT_DIR}" \
            "${CONTAINER_NAME}"

        local restore_end_ns
        restore_end_ns=$(date +%s%N)
        local restore_s
        restore_s=$(python3 -c "print(f'{(${restore_end_ns} - ${restore_start_ns}) / 1e9:.3f}')")

        DISK_RESTORE_TIMES+=("${restore_s}")

        # 验证推理
        local verify
        verify=$(verify_inference)
        ok "磁盘 Restore 第${i}次: ${restore_s}s, 验证: ${verify}"

        # 停止容器准备下次 restore
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    done

    ok "磁盘 restore 测量完成"
}

# ============================================================
# 实验3: CRIU restore — checkpoint 在 RAM disk (tmpfs)
# ============================================================
benchmark_ram_restore() {
    step "========== 实验3: CRIU restore (RAM Disk / tmpfs) =========="
    info "将进行 ${ITERATIONS} 次 RAM disk restore 测量..."

    # 先启动容器并创建 checkpoint（这次用 RAM disk 路径）
    info "准备: 启动容器 + 创建 checkpoint (RAM disk)..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --security-opt seccomp=unconfined \
        "${IMAGE_NAME}"

    wait_for_ready

    # 创建 checkpoint 到 RAM disk
    local cp_start_ns
    cp_start_ns=$(date +%s%N)

    docker checkpoint create \
        --checkpoint-dir "${RAMDISK_DIR}" \
        "${CONTAINER_NAME}" "${CHECKPOINT_NAME}"

    local cp_end_ns
    cp_end_ns=$(date +%s%N)
    local cp_s
    cp_s=$(python3 -c "print(f'{(${cp_end_ns} - ${cp_start_ns}) / 1e9:.3f}')")
    CHECKPOINT_CREATE_TIMES+=("${cp_s}")

    # RAM disk 上的 checkpoint 大小
    local ram_cp_size
    ram_cp_size=$(get_checkpoint_size "${RAMDISK_DIR}")
    info "RAM Disk Checkpoint 大小: ${ram_cp_size}"
    info "RAM Disk Checkpoint 创建耗时: ${cp_s}s"

    # 停止容器
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true

    # 多次 restore 测量
    for i in $(seq 1 "${ITERATIONS}"); do
        info "--- RAM Disk Restore 第 ${i}/${ITERATIONS} 次 ---"

        # 确保容器已停止
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        sleep 1

        local restore_start_ns
        restore_start_ns=$(date +%s%N)

        docker start \
            --checkpoint "${CHECKPOINT_NAME}" \
            --checkpoint-dir "${RAMDISK_DIR}" \
            "${CONTAINER_NAME}"

        local restore_end_ns
        restore_end_ns=$(date +%s%N)
        local restore_s
        restore_s=$(python3 -c "print(f'{(${restore_end_ns} - ${restore_start_ns}) / 1e9:.3f}')")

        RAM_RESTORE_TIMES+=("${restore_s}")

        # 验证推理
        local verify
        verify=$(verify_inference)
        ok "RAM Disk Restore 第${i}次: ${restore_s}s, 验证: ${verify}"

        # 停止容器准备下次 restore
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    done

    ok "RAM disk restore 测量完成"
}

# ============================================================
# 生成统计报告
# ============================================================
generate_report() {
    step "========== 生成统计报告 =========="

    # 将 Bash 数组序列化为逗号分隔字符串，通过环境变量传给 Python
    # 避免在 heredoc 中展开 shell 变量（防止注入风险）
    local cold_str disk_str ram_str
    cold_str=$(printf '%s,' "${COLD_START_TIMES[@]}" | sed 's/,$//')
    disk_str=$(printf '%s,' "${DISK_RESTORE_TIMES[@]}" | sed 's/,$//')
    ram_str=$(printf '%s,' "${RAM_RESTORE_TIMES[@]}" | sed 's/,$//')

    COLD_TIMES="$cold_str" \
    DISK_TIMES="$disk_str" \
    RAM_TIMES="$ram_str" \
    CP_SIZE="${CHECKPOINT_SIZE}" \
    RESULT_FILE="${RESULT_FILE}" \
    python3 << 'PYEOF'
import json
import os
import statistics

def parse_times(s):
    try:
        return [float(x) for x in s.split(',') if x]
    except:
        return []

def stats(times):
    if not times:
        return {"mean": "N/A", "min": "N/A", "max": "N/A", "std": "N/A"}
    return {
        "mean": round(statistics.mean(times), 3),
        "min": round(min(times), 3),
        "max": round(max(times), 3),
        "std": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "values": times
    }

cold = stats(parse_times(os.environ.get('COLD_TIMES', '')))
disk = stats(parse_times(os.environ.get('DISK_TIMES', '')))
ram  = stats(parse_times(os.environ.get('RAM_TIMES', '')))
cp_size = os.environ.get('CP_SIZE', 'N/A')
result_path = os.environ.get('RESULT_FILE', '/tmp/criu_ramdisk_benchmark.json')

print("")
print("=" * 70)
print("  CRIU Checkpoint 存储位置对 Restore 性能的影响 — 实验报告")
print("=" * 70)
print("")

print(f"  Checkpoint 大小: {cp_size}")
print("")

print("  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
print("  │ 方案                │ 平均(s)  │ 最小(s)  │ 最大(s)  │ 标准差   │")
print("  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

for label, data in [
    ("冷启动 (baseline)", cold),
    ("CRIU restore (磁盘)", disk),
    ("CRIU restore (RAM盘)", ram),
]:
    m = data['mean'] if isinstance(data['mean'], str) else f"{data['mean']:.3f}"
    mn = data['min'] if isinstance(data['min'], str) else f"{data['min']:.3f}"
    mx = data['max'] if isinstance(data['max'], str) else f"{data['max']:.3f}"
    sd = data['std'] if isinstance(data['std'], str) else f"{data['std']:.3f}"
    print(f"  │ {label:<19} │ {m:>8} │ {mn:>8} │ {mx:>8} │ {sd:>8} │")

print("  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘")
print("")

# 计算加速比
if isinstance(cold['mean'], float) and isinstance(ram['mean'], float) and ram['mean'] > 0:
    speedup_ram = cold['mean'] / ram['mean']
    print(f"  RAM盘 CRIU restore vs 冷启动 加速比: {speedup_ram:.2f}x")
    if speedup_ram > 1.0:
        print(f"  ✅ RAM盘 restore 比冷启动更快！提升 {(speedup_ram - 1) * 100:.1f}%")
    else:
        print(f"  ❌ RAM盘 restore 仍慢于冷启动，差距 {(1 - speedup_ram) * 100:.1f}%")

if isinstance(cold['mean'], float) and isinstance(disk['mean'], float) and disk['mean'] > 0:
    speedup_disk = cold['mean'] / disk['mean']
    print(f"  磁盘 CRIU restore vs 冷启动 加速比: {speedup_disk:.2f}x")

if isinstance(disk['mean'], float) and isinstance(ram['mean'], float) and disk['mean'] > 0:
    ram_improvement = (disk['mean'] - ram['mean']) / disk['mean'] * 100
    print(f"  RAM盘 vs 磁盘 restore 提升: {ram_improvement:.1f}%")

print("")

# 保存 JSON 结果
result = {
    "checkpoint_size": cp_size,
    "cold_start": cold,
    "disk_restore": disk,
    "ram_restore": ram,
}
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"  详细结果已保存到: {result_path}")
print("")
print("=" * 70)
PYEOF
}

# ---- 清理 RAM disk ----
# 安全约束：仅删除我们明确创建的目录，且必须验证路径合法性
cleanup_ramdisk() {
    info "清理 RAM disk..."

    # 安全检查：路径必须以 /tmp/ 开头且非空，防止误删
    if [[ -z "${RAMDISK_DIR}" || "${RAMDISK_DIR}" != /tmp/* ]]; then
        warn "跳过 RAM disk 清理：路径 '${RAMDISK_DIR}' 不在 /tmp/ 下，可能有安全风险"
    else
        if mountpoint -q "${RAMDISK_DIR}" 2>/dev/null; then
            sudo umount "${RAMDISK_DIR}"
            ok "RAM disk 已卸载"
        fi
        # umount 后目录应为空，使用 rmdir 而非 rm -rf
        if [ -d "${RAMDISK_DIR}" ]; then
            sudo rmdir "${RAMDISK_DIR}" 2>/dev/null || warn "RAM disk 目录非空，已保留: ${RAMDISK_DIR}"
        fi
    fi

    # 普通磁盘目录清理（同样加路径验证）
    if [[ -n "${DISK_CHECKPOINT_DIR}" && "${DISK_CHECKPOINT_DIR}" == /tmp/* ]]; then
        rm -rf "${DISK_CHECKPOINT_DIR}"
    else
        warn "跳过磁盘 checkpoint 目录清理：路径 '${DISK_CHECKPOINT_DIR}' 不在 /tmp/ 下"
    fi
}

# ---- 完整清理 ----
full_cleanup() {
    info "清理所有资源..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    cleanup_ramdisk
}

# ---- 主流程 ----
main() {
    echo ""
    echo "============================================================"
    echo "  方案A-优化：CRIU Checkpoint RAM Disk 加速实验"
    echo "  每组迭代次数: ${ITERATIONS}"
    echo "============================================================"
    echo ""

    # 注册退出时的清理
    trap full_cleanup EXIT

    check_prerequisites
    cleanup
    setup_ramdisk
    setup_disk_dir
    build_image

    # 运行三组实验
    benchmark_cold_start
    benchmark_disk_restore
    benchmark_ram_restore

    # 生成报告
    generate_report

    ok "实验完成!"
    echo ""
    info "提示: 如需保留 RAM disk 供后续实验使用，可在下次运行前手动挂载:"
    info "  sudo mount -t tmpfs -o size=1G tmpfs ${RAMDISK_DIR}"
}

# 运行
main "$@"
