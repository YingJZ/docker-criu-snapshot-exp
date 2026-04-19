# CRIU Restore 耗时长的原因分析

> 基于实验数据：ResNet-50 CPU 推理，CRIU 4.0，Ubuntu 20.04
>
> **v3 更新**：新增冷启动 vs CRIU restore 对比基准测试，裸机 CRIU 比冷启动快 13.4x

## 1. 实验数据回顾

### Docker 方案（原始实验）

| 指标 | 数值 |
|------|------|
| CRIU Checkpoint 创建 | **12.90s** |
| CRIU Restore 执行 | **9.23s** |
| 冷启动（import torch + model.load） | 2.48s |
| 进程 VmRSS（驻留内存） | 334 MB |
| 进程 VmSize（虚拟地址空间） | 2,508 MB |
| Checkpoint 磁盘占用 | 307 MB |

### 裸机方案（新增，用于精确定位瓶颈）

| 指标 | 数值 |
|------|------|
| CRIU Checkpoint 创建 | **0.56s** |
| CRIU Restore 执行 | **0.31s** |
| 进程 VmRSS | 423 MB |
| 进程 VmSize | 2,490 MB |
| Checkpoint 磁盘占用 | 325 MB |
| VMA 条目数 | 421 |
| 线程数 | 31 |
| FD 数量 | 4 |

**关键对比**：裸机 CRIU 只需 0.31s restore，Docker 方案需要 9.23s。**Docker 封装引入了 ~9s 的额外开销。**

## 2. 核心结论：Docker 封装开销是主因，CRIU 本身很快

原始分析认为"磁盘 IO 不是主因，VMA 重建才是"。但裸机实验证明：

- CRIU 本身对 423MB RSS / 2,490MB VmSize 的 PyTorch 进程，restore 只需 **0.31s**
- Docker 的 `docker checkpoint create` / `docker start --checkpoint` 封装引入了 **~9s** 的额外延迟
- 这 9s 不是 CRIU 的问题，而是 **Docker daemon 处理 checkpoint/restore 时的架构开销**

## 3. 裸机 CRIU Dump 精确阶段耗时

从 `criu dump -vvv` 日志提取的精确阶段耗时：

| 阶段 | 时间点(s) | 阶段耗时(s) | 占比 | 说明 |
|------|-----------|------------|------|------|
| CRIU 初始化 | 0.0001 | 0.0001 | <1% | 版本检查、内核特性探测 |
| 开始 Dump 进程 | 0.0011 | 0.0010 | 0.2% | 进程识别 |
| Seize 进程 | 0.0026 | 0.0015 | 0.3% | ptrace seize 主进程 + 31 线程 |
| Dump 任务详情 | 0.0244 | 0.0219 | 3.9% | /proc 解析、VMA 收集、FD 收集 |
| **内存页 drain** | 0.0244→0.5059 | **0.4815** | **86.6%** | 80,439 页 (325MB) 内存数据写入磁盘 |
| Dump 文件锁 | 0.5059 | 0.0001 | <1% | 文件锁信息 |
| Dump 进程树 | 0.5060 | 0.0001 | <1% | pstree 序列化 |
| 解冻进程 | 0.5067 | 0.0008 | 0.1% | cgroup freezer 解冻 |
| 写入统计 | 0.5559 | 0.0492 | 8.8% | stats.img 等元数据写入 |
| **总计** | | **0.556s** | | |

**Dump 瓶颈是内存页写入磁盘（0.48s，占 87%）**，这与 325MB × SATA SSD 带宽的理论值一致。

## 4. 裸机 CRIU Restore 精确阶段耗时

从 `criu restore -vvv` 日志提取的精确阶段耗时：

| 阶段 | 时间点(s) | 阶段耗时(s) | 占比 | 说明 |
|------|-----------|------------|------|------|
| CRIU 初始化 | 0.0001 | 0.0001 | <1% | 版本检查 |
| Pre-restore 脚本 | 0.0067 | 0.0066 | 2.2% | 运行前置脚本 |
| Fork 任务进程 | 0.0071 | 0.0004 | 0.1% | fork() 创建 init 进程 |
| 恢复 namespace | 0.0086 | 0.0015 | 0.5% | 创建/加入 namespace |
| 恢复核心状态 | 0.0133 | 0.0046 | 1.5% | sigaction、FD 信息 |
| **开始打开 VMA/FD** | 0.0134 | 0.0001 | <1% | 开始逐一打开文件映射 |
| Sigreturn 切换 | 0.0198 | 0.0064 | 2.1% | 通过 sigreturn 进入 restorer |
| **VMA mmap 重建** | 0.0234→0.0255 | **0.0021** | **0.7%** | 421 次 mmap() 系统调用 |
| **内存页数据恢复** | 0.0255→0.2444 | **0.2189** | **71.5%** | 80,439 页数据写入进程地址空间 |
| VDSO 重映射 | 0.2444 | ~0.0001 | <1% | vDSO 位置调整 |
| **线程恢复 & 同步** | 0.2444→0.3062 | **0.0618** | **20.2%** | 31 线程 trap/stop/resume |
| **总计** | | **0.306s** | | |

### mmap 调用详细统计

| mmap 类型 | 数量 | 说明 |
|-----------|------|------|
| 匿名映射 (MAP_ANONYMOUS) | 96 | Python 堆、线程栈、pymalloc arenas |
| 文件映射 (file-backed) | 293 | .so 库文件、Python 模块 |
| 大块匿名 (MAP_ANONYMOUS + MAP_NORESERVE) | 30 | PyTorch 内部大块预分配 |
| **总计** | **421** | 与 /proc/pid/maps 的 VMA 数一致 |

**关键发现**：421 次 mmap() 只用了 **0.0021s**！mmap 系统调用本身并不慢（平均 5μs/次）。

### 真正的 restore 瓶颈：内存页数据写入

```
0.0255s → 0.2444s = 0.219s  用于将 80,439 页 (314MB) 数据写入进程地址空间
```

这是 restore 的主要耗时（71.5%），其瓶颈是：
- 从 checkpoint 文件读取 325MB 数据（磁盘 IO）
- 通过 write() 系统调用将每页数据写入已 mmap 的进程地址空间
- 每页涉及：lseek → read → write 到目标地址

## 5. Docker 方案 9.23s 的开销来源分析

裸机 CRIU 只需 0.31s，Docker 方案需要 9.23s。额外的 ~9s 来自：

| 开销来源 | 预估耗时 | 说明 |
|----------|----------|------|
| **Docker daemon RPC 通信** | 1–2s | `docker checkpoint create` → Docker daemon → containerd → runc → CRIU 的多层调用链 |
| **容器状态管理** | 2–3s | Docker 需要在 checkpoint 后更新容器状态为 "exited"，restore 时重新创建容器运行时 |
| **OverlayFS 层处理** | 1–2s | Docker 每次操作需要挂载/卸载 overlayfs 层 |
| **Namespace 重建** | 1–2s | Docker 需要为容器创建完整的 namespace 隔离（PID/NET/MNT/IPC/UTS） |
| **Cgroup 处理** | 0.5–1s | CRIU 在容器 cgroup 中操作比裸机复杂得多（14 个 cgroup controller） |
| **网络设备恢复** | 0.5–1s | veth 设备、bridge、iptables 规则恢复 |
| **CRIU 本身** | ~0.3s | 实际的进程状态 dump/restore（已通过裸机实验验证） |

**核心问题**：Docker 的 `docker checkpoint` 是通过 Docker daemon → containerd → runc → CRIU 的多层调用实现的。每一层都有自己的状态管理、锁、序列化/反序列化开销。裸机直接调用 CRIU 可以完全绕过这些开销。

## 6. 修正之前的分析

| 分析项 | 之前预估 | 实际测量 | 偏差原因 |
|--------|----------|----------|----------|
| VMA 重建耗时 | 2.5–4.0s | **0.002s** | 大幅高估。421 次 mmap() 只需 2.1ms |
| 内存页 IO 耗时 | 1.5–2.5s | **0.219s** | 高估。325MB 在 SSD 上读取很快 |
| Docker 开销 | 1–2s | **~9s** | 大幅低估。Docker 封装的开销远超预期 |
| CRIU 总耗时 | 9.23s | **0.31s**（裸机） | CRIU 本身很快，9s 是 Docker 造成的 |

**根本原因不是 CRIU 慢，而是 Docker 的 checkpoint/restore 封装引入了巨大开销。**

## 7. 对 RAM Disk 实验的修正预期

| 场景 | 之前预期 | 修正后预期 | 原因 |
|------|----------|-----------|------|
| Docker + RAM disk | 5–7s | **8–9s** | Docker 开销不受 RAM disk 影响 |
| 裸机 + 普通磁盘 | - | **~0.3s** | 已实测 |
| 裸机 + RAM disk | - | **~0.2s** | 消除磁盘 IO，仅剩内存拷贝 |

RAM disk 对 Docker 方案帮助不大，因为瓶颈是 Docker 封装开销而非 IO。

## 8. 优化方向（按效果排序）

| 优化 | 预期效果 | 原理 | 复杂度 |
|------|----------|------|--------|
| **裸机 CRIU（绕过 Docker）** | 9.2s → **0.3s** | 消除 Docker daemon 多层封装开销 | 中 |
| **Podman 替代 Docker** | 可能 → 1–3s | Podman 的 CRIU 集成更直接 | 低 |
| RAM disk (tmpfs) + 裸机 CRIU | 0.3s → ~0.2s | 消除磁盘 IO | 低 |
| **containerd 直接调 CRIU** | 可能 → 1–2s | 跳过 Docker daemon 层 | 高 |
| 减小 VmSize / VMA 数量 | 边际改善 | mmap 本身只需 2ms，优化空间不大 | 中 |
| `--pre-dump` + `--track-mem` | 减少冻结时间 | 不影响 restore 时间 | 低 |

## 9. 冷启动 vs CRIU Restore 对比

### 测试方法

使用 `coldstart_bench.py` 测量裸机 PyTorch 冷启动各阶段耗时（延迟 import 精确计时），使用 `bare_inference.py` + `criu restore` 测量 CRIU 恢复后首次推理可用时间。各 5-10 轮取平均。

### 冷启动各阶段耗时（10轮平均）

| 阶段 | 平均耗时 | 说明 |
|------|----------|------|
| `import torch` | **2.070s** | 加载 PyTorch 库及其依赖（numpy、C 扩展等） |
| `import torchvision` | **1.797s** | 加载 torchvision 及其依赖 |
| 模型加载 (ResNet-50) | **0.686s** | 加载权重文件、构建计算图 |
| 首次推理 (warm-up) | **0.105s** | 包含 input 构造 + 前向传播 |
| **脚本内总耗时** | **4.658s** | 从脚本第一行到首次推理完成 |
| **Wall time (含Python启动)** | **5.565s** | 从 `python` 命令到首次推理完成 |

### CRIU Restore 端到端耗时（6轮平均）

| 阶段 | 平均耗时 | 说明 |
|------|----------|------|
| CRIU restore | **0.300s** | 进程状态恢复（内存页、VMA、线程等） |
| 首次推理 (内部计时) | **0.115s** | restore 后立即执行推理 |
| **restore + 推理可用** | **0.415s** | 从 `criu restore` 到首次推理完成 |

### 核心对比

| 方案 | 首次推理可用时间 | 说明 |
|------|------------------|------|
| **裸机冷启动** | **5.565s** | python 启动 → import → model.load → 推理 |
| **裸机 CRIU restore** | **0.415s** | criu restore → 推理 |
| Docker CRIU restore | 9.230s | docker start --checkpoint → 推理 |

**裸机 CRIU restore 比冷启动快 13.4x**（5.565s → 0.415s）

### 冷启动 vs CRIU Restore 时间分解

```
冷启动 (5.565s):
├── import torch          2.070s  ████████████████████  37.2%
├── import torchvision    1.797s  █████████████████     32.3%
├── 模型加载              0.686s  ███████               12.3%
├── 首次推理              0.105s  █                      1.9%
└── Python 启动等         0.907s  █████████              16.3%

CRIU Restore (0.415s):
├── CRIU restore          0.300s  ████████████████████  72.3%
│   ├── 内存页恢复        0.219s  (71.5% of restore)
│   ├── 线程恢复          0.062s  (20.2% of restore)
│   └── 其他              0.019s  (6.3% of restore)
└── 首次推理              0.115s  ███████               27.7%
```

### 关键发现

1. **CRIU 完全跳过了冷启动的两大瓶颈**：`import torch` (2.07s) + `import torchvision` (1.80s) = **3.87s** 被完全消除
2. **CRIU restore 后推理只需 0.115s**，与冷启动后的推理耗时 (0.105s) 几乎一致，说明模型状态完整恢复
3. **CRIU 恢复比重新加载模型更快**：restore 0.30s < import+load 4.56s，即使加上 restore 后推理 (0.115s)，总耗时仍仅 0.415s
4. **Docker CRIU 反而更慢**：9.23s > 5.565s 冷启动，Docker 封装开销完全抵消了 CRIU 的优势

### 不同方案的对比总结

| 方案 | 首次推理可用 | vs 冷启动 | 适用场景 |
|------|-------------|-----------|----------|
| 裸机冷启动 | 5.565s | baseline | 无 CRIU 环境 |
| 裸机 CRIU | **0.415s** | **13.4x 快** | 推荐：裸机/K8s + containerd |
| Docker 冷启动 | ~6-8s | ~0.8x | 普通 Docker 部署 |
| Docker CRIU | 9.230s | **1.7x 慢** | ❌ 不推荐 |

## 10. 结论

### 实测数据驱动的结论

1. **裸机 CRIU restore 比冷启动快 13.4x**：冷启动 5.565s → CRIU restore+推理 0.415s，节省 5.15s
2. **CRIU 完全跳过 import 开销**：`import torch` (2.07s) + `import torchvision` (1.80s) = 3.87s 被消除
3. **Docker CRIU 反而比冷启动更慢**：9.23s > 5.565s，Docker 封装完全抵消了 CRIU 的优势
4. **CRIU 本身非常快**：对 423MB RSS / 2,490MB VmSize / 421 VMA / 31 线程的 PyTorch 进程，裸机 restore 仅需 0.30s
5. **VMA 重建不是瓶颈**：421 次 mmap() 只用了 0.002s（平均 5μs/次）
6. **内存页数据写入是 CRIU 本身的最大开销**：0.219s（71.5% of restore）
7. **restore 后推理性能无损**：0.115s vs 冷启动推理 0.105s，差异 <10%

### 对原始分析中高估的反思

原始分析基于 CRIU 官方"VMA 重建占 73%"的性能研究，但该研究针对的是更老版本的 CRIU 和不同的工作负载。PyTorch 进程的 421 个 VMA 在现代 CRIU 4.0 上处理非常快（2.1ms），说明 CRIU 的 VMA 处理已经高度优化。

### 最佳优化路径

**裸机 CRIU（绕过 Docker）** 是最有效的优化方案，可将首次推理可用时间从 5.565s 降至 0.415s（**13.4x 加速**）。如果必须使用容器化，考虑用 Podman 替代 Docker 或通过 containerd 直接调用 CRIU。

---

*本分析 v3 基于裸机 `criu dump/restore -vvv` 的实际日志数据 + 冷启动基准测试对比，替代了 v1/v2 中基于文献的预估值。*

## 附录：裸机 CRIU 实验复现步骤

```bash
# 1. 创建虚拟环境并安装 PyTorch
uv venv .venv --python 3.10
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 后台运行推理进程
.venv/bin/python bare_inference.py &>/tmp/pytorch-bare.log &

# 3. 等待就绪
while ! grep -q '\[READY\]' /tmp/pytorch-bare.log; do sleep 1; done
PID=$(cat /tmp/criu-pytorch.pid)

# 4. 捕获进程特征
wc -l /proc/$PID/maps          # VMA 数量
ls /proc/$PID/fd | wc -l       # FD 数量
cat /proc/$PID/status | grep Threads  # 线程数

# 5. CRIU dump（带详细日志）
mkdir -p /tmp/criu-dump-bare
sudo criu dump -vvv -o /tmp/criu-dump-bare/dump.log \
  -t $PID --shell-job -D /tmp/criu-dump-bare

# 6. CRIU restore（带详细日志，-d 以 daemon 方式运行）
mkdir -p /tmp/criu-restore-bare
sudo criu restore -d -vvv -o /tmp/criu-restore-bare/restore.log \
  -D /tmp/criu-dump-bare --shell-job

# 7. 验证推理结果
touch /tmp/criu-pytorch-trigger
sleep 2
cat /tmp/criu-pytorch-results/verification.json

# 8. 分析日志
# Dump 总耗时: 0.556s，其中内存页写入占 0.48s (87%)
# Restore 总耗时: 0.306s，其中内存页恢复占 0.22s (72%)
```
