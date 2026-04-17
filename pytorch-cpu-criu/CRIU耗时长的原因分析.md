# CRIU Restore 耗时长的原因分析

> 基于实验数据：ResNet-50 CPU 推理，CRIU 4.0，Ubuntu 20.04，Docker 24.0.5

## 1. 实验数据回顾

| 指标 | 数值 |
|------|------|
| CRIU Checkpoint 创建 | **12.90s** |
| CRIU Restore 执行 | **9.23s** |
| 冷启动（import torch + model.load） | 2.48s |
| 进程 VmRSS（驻留内存） | 334 MB |
| 进程 VmSize（虚拟地址空间） | 2,508 MB |
| Checkpoint 磁盘占用 | 307 MB |

**核心问题**：CRIU Restore 耗时 9.23s，是冷启动的 3.7 倍。直觉上 307MB 的数据量不应产生如此高的延迟。

## 2. 核心结论：磁盘 IO 不是主因，VMA 重建才是

307MB checkpoint 在不同存储介质上的纯 IO 时间：

| 存储 | 顺序读带宽 | 307MB 纯 IO 时间 |
|------|-----------|------------------|
| SATA SSD | ~500 MB/s | ~0.6s |
| NVMe SSD | ~3,000 MB/s | ~0.1s |
| tmpfs (RAM disk) | ~10,000 MB/s | ~0.03s |

即使使用最慢的 SATA SSD，纯 IO 也只需 0.6s。**9.23s 中超过 90% 的时间不是磁盘 IO，而是 CRIU 重建进程状态的 CPU/syscall 开销。**

## 3. 三个"大小"的差异是理解耗时的关键

```
VmSize (2,508 MB)  >>>>  VmRSS (334 MB)  ≈  Checkpoint (307 MB)
       8.2x                    1.1x
```

- **Checkpoint (307MB)**：CRIU 只**写入**驻留内存页，这就是磁盘上的数据量
- **VmRSS (334MB)**：进程实际使用的物理内存，≈ Checkpoint 大小
- **VmSize (2,508MB)**：进程的完整虚拟地址空间，**CRIU 必须在 restore 时全部重建**

CRIU 只保存了驻留页的数据，但 restore 时必须重建整个 2,508MB 的虚拟地址空间布局。这就是耗时的根本来源。

## 4. 9.23s 的逐阶段拆解

| 阶段 | 预估耗时 | 占比 | 瓶颈类型 | 说明 |
|------|----------|------|----------|------|
| **① Docker 容器重建** | 1.0–2.0s | 15–20% | RPC + 文件系统 | Docker daemon 创建容器元数据、overlayfs mount、namespace 初始化 |
| **② VMA 重建（虚拟地址空间重构）** | **2.5–4.0s** | **30–40%** | CPU / syscall | 1000+ 次 `mmap()` 系统调用 + 内核页表插入 |
| **③ 内存页读取 & 写入** | 1.5–2.5s | 20–25% | 磁盘 IO + 内存拷贝 | 读取 307MB checkpoint → write 到进程地址空间 |
| **④ 文件描述符重建** | 0.5–1.5s | 5–15% | 路径验证 + inode 校验 | 数百个 FD，每个需要验证 backing file |
| **⑤ 线程/寄存器状态恢复** | 0.3–0.5s | 3–5% | 状态序列化 | 单线程 Python 进程，开销较小 |
| **⑥ namespace/cgroup 收尾** | 0.3–1.0s | 3–10% | 内核操作 | cgroup 迁移（~80ms/task），netns 设置 |
| **⑦ 进程解冻 & 稳定化** | 0.5–1.0s | 5–10% | 调度 + page fault | 内核调度恢复的线程，处理首次 page fault |

### 为什么阶段②（VMA 重建）是最大瓶颈

CRIU 官方性能研究指出，**VMA 重建占 restore 时间的约 73%**。

#### PyTorch 进程的 VMA 数量极大

PyTorch 进程的虚拟地址空间高度碎片化，典型来源：

```
torch._C.cpython-310-x86_64-linux-gnu.so  → ~600MB 虚拟映射，多个 VMA（text/rodata/data/bss）
libtorch_cpu.so                             → 巨大共享库，多个 segment
libstdc++.so / libm.so / libc.so           → 标准 C++ 库，每个多个 VMA
Python 对象堆（pymalloc arenas）           → 大量匿名 mmap 区域
线程栈                                      → 每线程 8MB 映射
模型权重文件 mmap（safetensors）           → 文件-backed VMA
```

一个典型的 PyTorch 进程拥有 **500–2000+ 个 VMA 条目**。

#### 每个 VMA 的 restore 流程

```
读元数据 → mmap(flags, offset, prot) → 验证 backing file → 写入驻留页
```

#### `mmap()` 系统调用是昂贵的

每次 `mmap()` 涉及：
- 内核 VMA 红黑树插入
- 页表项（PTE）创建
- 对于文件-backed 映射：inode 查找 + address_space 初始化

1000+ 次 `mmap()` 的内核开销远超 307MB 的磁盘 IO。

## 5. Checkpoint 阶段为什么也慢（12.90s）

Checkpoint 的耗时结构类似：

| 阶段 | 预估耗时 | 说明 |
|------|----------|------|
| 进程冻结（cgroup freezer + ptrace seize） | 0.5–1.5s | 暂停进程所有线程 |
| **/proc 解析（收集 VMA/FD 信息）** | **2.0–4.0s** | 读取 `/proc/pid/smaps`，2000+ VMA 条目 = 数千次 `openat/fstat/read` |
| **Parasite 注入 & 内存 drain** | **4.0–6.0s** | 注入 parasite 代码，将 307MB 内存页写入磁盘 |
| 进程树 & 元数据序列化 | 0.5–1.0s | pstree, core, fds 等镜像文件生成 |
| 清理 & 解冻 | 0.3–0.5s | 移除 parasite 代码，恢复进程运行 |

`/proc/pid/smaps` 解析是 checkpoint 阶段的一个隐藏瓶颈。CRIU 性能研究记录过一个 11-task 容器中 `parse_smaps` 就花了 0.2s — 对于单进程但 VMA 数量更多的 PyTorch 进程，这个时间只长不短。

## 6. 对 RAM Disk 实验的预期修正

`pytorch-cpu-criu-ramdisk/` 实验假设 RAM disk 可以将 restore 降到 1.5–3s。**这个预期可能过于乐观**：

| 场景 | 实验计划预期 | 修正后预期 | 原因 |
|------|-------------|-----------|------|
| 乐观：CRIU 固定开销小 | 1.5–3s | **5–6s** | VMA 重建是 CPU 瓶颈，不受存储介质影响 |
| 保守：CRIU 固定开销大 | 3–5s | **6–7s** | 同上 |

RAM disk 只消除了阶段③（内存页 IO）中的磁盘读写部分，但：
- 阶段②的 VMA 重建开销完全不受 RAM disk 影响（这是 CPU/syscall 瓶颈）
- 阶段①的 Docker 开销也不受影响
- RAM disk 对 restore 的改善幅度大约是 **2–3s**（从 9.2s 降到 6–7s），而非实验计划预期的 6–7s 改善

## 7. 如何验证本分析

当前实验的**最大缺失**是没有 CRIU 详细日志。要确认上述拆解，需要：

### 7.1 捕获 CRIU 详细日志

```bash
# 方法1：使用 CRIU 直接（绕过 Docker）
# 需要 Docker 的 checkpoint 目录位置
CRDIR=$(docker inspect --format='{{.CheckpointDir}}' <container>)

# 手动运行 CRIU restore，开启 verbose 日志
criu restore -vvv -o /tmp/criu-restore.log -d --shell-job

# 方法2：查看 CRIU stats（如果 Docker 保留了）
crit show --stats /var/lib/docker/containers/<ID>/checkpoints/cp1/stats.img
```

CRIU verbose 日志会输出每个阶段的毫秒级耗时，包括 `frozen_time`、`memdump_time`、`memwrite_time` 等关键字段。

### 7.2 在 checkpoint 前捕获进程特征数据

```bash
# VMA 条目数（最关键 — 直接影响 restore 耗时）
docker exec <container> sh -c "wc -l /proc/1/maps"

# FD 数量（影响 FD 重建阶段）
docker exec <container> sh -c "ls /proc/1/fd | wc -l"

# 线程数
docker exec <container> sh -c "cat /proc/1/status | grep Threads"

# 内存映射概要
docker exec <container> sh -c "cat /proc/1/smaps_rollup"
```

### 7.3 对比实验

| 实验 | 目的 | 预期结果 |
|------|------|----------|
| RAM disk restore | 分离 IO 和 CPU 开销 | ~6–7s（证明 IO 只占 2–3s） |
| 简单 Python 进程 restore | 基准对比 | <<1s（证明 PyTorch 的 VMA 碎片化是关键） |
| `madvise(MADV_DONTNEED)` 释放非必要映射后 restore | 验证 VMA 数量影响 | 显著变快 |

## 8. 优化方向

按预期效果排序：

| 优化 | 预期效果 | 原理 | 复杂度 |
|------|----------|------|--------|
| **减小 VmSize / VMA 数量** | **最有效** | 减少需要重建的 VMA 数量，从根本降低 syscall 开销 | 中 |
| RAM disk (tmpfs) | 9.2s → ~6–7s | 消除磁盘 IO | 低 |
| 跳过 Docker，直接用 CRIU | 节省 1–2s | 避免 Docker daemon 的容器创建开销 | 中 |
| CRIU lazy-pages | restore 延迟加载 | 只在 page fault 时加载页面，restore 快但首次推理慢 | 高 |
| 用 Podman 替代 Docker | 可能更快 | Podman 的 CRIU 集成更原生 | 低 |
| `--pre-dump` + `--track-mem` | 减少 freeze 时间 | 不减少 restore 时间 | 低 |

### 减小 VmSize / VMA 数量的具体方法

1. **`torch.device("meta")` 延迟初始化**：模型权重使用 meta device，不分配实际内存，restore 后再 materialize
2. **safetensors mmap 模式**：权重以文件-backed mmap 加载，CRIU 不需要 dump 这些页面（只需验证文件路径不变），但需验证 CRIU 对 mmap 的恢复行为
3. **释放 PyTorch 内部缓存**：checkpoint 前调用 `torch.cuda.empty_cache()`（GPU）和 `gc.collect()` 减少 RSS
4. **精简 Python 运行时**：减少 import 的模块数量，每个 `.so` 模块都增加 VMA 数量

## 9. 结论

> **CRIU Restore 9.23s ≈ VMA 重建(3–4s) + 内存页 IO(1.5–2.5s) + Docker 开销(1–2s) + 其他(1–2s)**

- 磁盘 IO 仅占 **15–25%**，不是主要瓶颈
- **VMA 重建（虚拟地址空间重构）占 30–40%**，是最大瓶颈，由 PyTorch 进程内存布局高度碎片化导致
- Docker 容器创建开销占 **15–20%**，是可消除的固定成本
- 对于冷启动 >10s 的大模型，VMA 重建的固定开销（~3–4s）相对于模型加载时间变得可接受
- 对于小模型（冷启动 <3s），CRIU restore 不可能快于冷启动，因为 VMA 重建和 Docker 开销的固定成本 >3s

---

*本分析基于实验数据、CRIU 官方性能研究、CRIU 源码架构理解。要获得精确的逐阶段耗时，需要运行 `criu restore -vvv` 并分析详细日志。*
